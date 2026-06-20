"""GPU validation: does Rank-1 EWC retain old knowledge on a real image diffusion model?

Self-contained, single-file, run-and-go on a single GPU (tuned for a Colab T4, ~16 GB).
It trains a custom UNet noise-prediction diffusion model on a REAL dataset (CIFAR-10, auto
downloaded) and exercises the actual shipped anti-forgetting code paths:

    library.anti_forgetting.create_ewc_regularizer / EWCRegularizer   (Rank-1 EWC)
    library.anti_forgetting.create_adaptive_lambda_controller          (adaptive lambda)
    library.distillation.distillation_loss                             (output distillation)

Protocol (the catastrophic-forgetting setup):
  1. Pre-train the UNet on an "old" class (e.g. CIFAR automobile) -> theta*.
  2. Fine-tune on a "new" class (e.g. CIFAR horse) under several configurations.
  3. Report the noise-prediction loss on BOTH classes. Lower old loss after fine-tuning = the
     old knowledge was retained. The baseline forgets; Rank-1 EWC should keep old loss low
     while still learning the new class.

Colab T4 usage:
    !git clone <this repo> && cd sd-scripts
    !pip install torchvision        # torch + torchvision are preinstalled on Colab
    !python tools/dev/gpu_validate_anti_forgetting.py

CPU self-test (no dataset download, a few seconds):
    python tools/dev/gpu_validate_anti_forgetting.py --smoke

Nothing is hardcoded to a path; everything is a CLI flag with a sensible default. Run
``python tools/dev/gpu_validate_anti_forgetting.py --help`` for the full list.
"""

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from library import distillation
from library.anti_forgetting import (
    add_adaptive_penalty,
    create_adaptive_lambda_controller,
    create_ewc_regularizer,
)


def group_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class UNet(nn.Module):
    """A small but real two-scale UNet eps-predictor with time conditioning."""

    def __init__(self, base: int, in_ch: int = 3):
        super().__init__()
        ch2 = base * 2
        temb_dim = base * 4
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down1_blocks = nn.ModuleList([ResBlock(base, base, temb_dim), ResBlock(base, base, temb_dim)])
        self.down1 = Downsample(base)
        self.down2_blocks = nn.ModuleList([ResBlock(base, ch2, temb_dim), ResBlock(ch2, ch2, temb_dim)])
        self.down2 = Downsample(ch2)
        self.mid_blocks = nn.ModuleList([ResBlock(ch2, ch2, temb_dim), ResBlock(ch2, ch2, temb_dim)])
        self.up2 = Upsample(ch2)
        self.up2_blocks = nn.ModuleList([ResBlock(ch2 + ch2, ch2, temb_dim), ResBlock(ch2, ch2, temb_dim)])
        self.up1 = Upsample(ch2)
        self.up1_blocks = nn.ModuleList([ResBlock(ch2 + base, base, temb_dim), ResBlock(base, base, temb_dim)])
        self.out_norm = group_norm(base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(sinusoidal_embedding(t, self.base))
        h = self.stem(x)
        for b in self.down1_blocks:
            h = b(h, temb)
        s1 = h
        h = self.down1(h)
        for b in self.down2_blocks:
            h = b(h, temb)
        s2 = h
        h = self.down2(h)
        for b in self.mid_blocks:
            h = b(h, temb)
        h = self.up2(h)
        h = torch.cat([h, s2], dim=1)
        for b in self.up2_blocks:
            h = b(h, temb)
        h = self.up1(h)
        h = torch.cat([h, s1], dim=1)
        for b in self.up1_blocks:
            h = b(h, temb)
        return self.out_conv(F.silu(self.out_norm(h)))


def make_schedule(steps: int, device: torch.device) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    return torch.cumprod(1.0 - betas, dim=0)  # abar


def add_noise(x0, t, noise, abar):
    ab = abar[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def infinite_images(loader):
    while True:
        for batch in loader:
            yield batch[0] if isinstance(batch, (list, tuple)) else batch


def get_data(args, device):
    """Return (old_loader, new_loader, old_eval, new_eval). In --smoke mode use random tensors."""
    if args.smoke:
        def random_loader():
            while True:
                yield torch.randn(args.batch, 3, 32, 32)
        old_eval = torch.randn(args.eval_images, 3, 32, 32, device=device)
        new_eval = torch.randn(args.eval_images, 3, 32, 32, device=device)
        return random_loader(), random_loader(), old_eval, new_eval

    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])
    train = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=tf)
    test = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=tf)

    def subset_loader(ds, cls, shuffle):
        idx = [i for i, y in enumerate(ds.targets) if y == cls]
        return DataLoader(Subset(ds, idx), batch_size=args.batch, shuffle=shuffle, drop_last=True, num_workers=args.workers)

    def eval_tensor(ds, cls):
        idx = [i for i, y in enumerate(ds.targets) if y == cls][: args.eval_images]
        return torch.stack([ds[i][0] for i in idx]).to(device)

    old_loader = infinite_images(subset_loader(train, args.old_class, True))
    new_loader = infinite_images(subset_loader(train, args.new_class, True))
    old_eval = eval_tensor(test, args.old_class)
    new_eval = eval_tensor(test, args.new_class)
    return old_loader, new_loader, old_eval, new_eval


def build_args_namespace(args, ewc_lambda=0.0, distill_high=0.0, distill_low=0.0, adaptive=False):
    """Mirror the real training args the shipped helpers read, with the repo defaults."""
    return SimpleNamespace(
        loss_type="l2",
        ewc_lambda=ewc_lambda,
        ewc_fisher_samples=args.fisher_samples,
        ewc_buffers_on_cpu=args.ewc_buffers_on_cpu,
        distillation_weight_high=distill_high,
        distillation_weight_low=distill_low,
        adaptive_lambda=adaptive,
        adaptive_lambda_ema=0.99,
        adaptive_lambda_base=1.0,
        adaptive_lambda_min=0.0,
        adaptive_lambda_max=10.0,
    )


def fisher_phase(model, reg, loader, abar, device):
    """Estimate u over the first num_fisher_samples micro-batches at theta* (fp32, no optimizer step)."""
    model.train()
    for _ in range(reg.num_fisher_samples):
        x0 = next(loader).to(device)
        t = torch.randint(0, abar.shape[0], (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        model.zero_grad(set_to_none=True)
        F.mse_loss(model(add_noise(x0, t, noise, abar), t), noise).backward()
        reg.accumulate()
        if reg.maybe_finalize():
            break
    model.zero_grad(set_to_none=True)


@torch.no_grad()
def evaluate(model, x0, abar, reps, device, seed):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    for _ in range(reps):
        t = torch.randint(0, abar.shape[0], (x0.shape[0],), generator=gen, device=device)
        noise = torch.randn(x0.shape, generator=gen, device=device)
        total += float(F.mse_loss(model(add_noise(x0, t, noise, abar), t), noise))
    model.train()
    return total / reps


def drift_geometry(model, reg_ref):
    params = dict(model.named_parameters())
    s = u_sq = total_sq = 0.0
    for n, u in reg_ref.u.items():
        delta = params[n].detach().float() - reg_ref.theta_star[n].to(params[n].device).float()
        s += float(torch.sum(u.to(params[n].device).float() * delta))
        u_sq += float(torch.sum(u.float() ** 2))
        total_sq += float(torch.sum(delta ** 2))
    u_norm = math.sqrt(u_sq) if u_sq > 0 else 1.0
    along = abs(s / u_norm)
    total = math.sqrt(total_sq)
    return along, math.sqrt(max(total * total - along * along, 0.0))


def train_config(cfg, theta_star, args, new_loader, abar, device, reg_ref, old_eval, new_eval):
    torch.manual_seed(args.seed)
    model = UNet(args.base_channels).to(device)
    model.load_state_dict(theta_star)

    reg = teacher = controller = None
    na = cfg["args"]
    if na.ewc_lambda > 0.0:
        reg = create_ewc_regularizer(na, [model], accelerator=None)
        fisher_phase(model, reg, new_loader, abar, device)
    if distillation.is_enabled(na):
        teacher = UNet(args.base_channels).to(device)
        teacher.load_state_dict(theta_star)
        teacher.eval()
        teacher.requires_grad_(False)
        controller = create_adaptive_lambda_controller(na)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for _ in range(args.finetune_steps):
        x0 = next(new_loader).to(device)
        t = torch.randint(0, abar.shape[0], (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        x_t = add_noise(x0, t, noise, abar)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x_t, t)
            task = F.mse_loss(pred, noise)
        total = task.float()
        if reg is not None:
            total = total + reg.penalty()
        elif teacher is not None:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
                teacher_pred = teacher(x_t, t)
            level = (1 - abar[t]).float()
            weights = torch.ones(x0.shape[0], device=device)
            dterm = distillation.distillation_loss(pred.float(), teacher_pred.float(), level, weights, na)
            total = add_adaptive_penalty(task.float(), dterm, controller, accelerator=None)
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

    old_loss = evaluate(model, old_eval, abar, args.eval_reps, device, args.seed + 50)
    new_loss = evaluate(model, new_eval, abar, args.eval_reps, device, args.seed + 50)
    along, perp = drift_geometry(model, reg_ref)
    return new_loss, old_loss, along, perp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="./data", help="where CIFAR-10 is downloaded/cached")
    parser.add_argument("--old_class", type=int, default=1, help="CIFAR-10 class index pre-trained as 'old' (1=automobile)")
    parser.add_argument("--new_class", type=int, default=7, help="CIFAR-10 class index fine-tuned as 'new' (7=horse)")
    parser.add_argument("--base_channels", type=int, default=128, help="UNet base width (model size)")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pretrain_steps", type=int, default=3000)
    parser.add_argument("--finetune_steps", type=int, default=1500)
    parser.add_argument("--fisher_samples", type=int, default=50)
    parser.add_argument("--ewc_lambdas", type=float, nargs="+", default=[20.0, 200.0])
    parser.add_argument("--distill_high", type=float, default=1.0)
    parser.add_argument("--distill_low", type=float, default=0.2)
    parser.add_argument("--ewc_buffers_on_cpu", action="store_true")
    parser.add_argument("--eval_images", type=int, default=512)
    parser.add_argument("--eval_reps", type=int, default=40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="random data + tiny config to self-test on CPU")
    args = parser.parse_args()

    if args.smoke:
        args.base_channels = 16
        args.timesteps = 20
        args.batch = 8
        args.pretrain_steps = 4
        args.finetune_steps = 4
        args.fisher_samples = 3
        args.eval_images = 8
        args.eval_reps = 2
        args.ewc_lambdas = [10.0]
        args.workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    abar = make_schedule(args.timesteps, device)
    old_loader, new_loader, old_eval, new_eval = get_data(args, device)

    model = UNet(args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device = {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"UNet base_channels={args.base_channels}  parameters = {n_params/1e6:.2f}M")
    print(f"old class = {args.old_class}   new class = {args.new_class}   timesteps = {args.timesteps}")

    # Pre-train on the old class -> theta*.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    t0 = time.time()
    model.train()
    for step in range(args.pretrain_steps):
        x0 = next(old_loader).to(device)
        t = torch.randint(0, abar.shape[0], (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = F.mse_loss(model(add_noise(x0, t, noise, abar), t), noise)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    theta_star = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print(f"pre-train done in {time.time()-t0:.0f}s")

    old0 = evaluate(model, old_eval, abar, args.eval_reps, device, args.seed + 50)
    new0 = evaluate(model, new_eval, abar, args.eval_reps, device, args.seed + 50)

    # Reference u (new-task Fisher at theta*), reused to project every run's drift.
    ref = UNet(args.base_channels).to(device)
    ref.load_state_dict(theta_star)
    reg_ref = create_ewc_regularizer(build_args_namespace(args, ewc_lambda=1.0), [ref], accelerator=None)
    fisher_phase(ref, reg_ref, new_loader, abar, device)

    configs = [{"name": "baseline (none)", "args": build_args_namespace(args)}]
    for lam in args.ewc_lambdas:
        configs.append({"name": f"Rank-1 EWC lambda={lam:g}", "args": build_args_namespace(args, ewc_lambda=lam)})
    configs.append({"name": "distillation", "args": build_args_namespace(args, distill_high=args.distill_high, distill_low=args.distill_low)})
    configs.append({"name": "distillation + adaptive", "args": build_args_namespace(args, distill_high=args.distill_high, distill_low=args.distill_low, adaptive=True)})

    print()
    print(f"theta* fit:  old loss = {old0:.4f}   new loss = {new0:.4f}")
    print("(low old loss = knows the old class; high new loss = the new class is not learned yet)")
    header = f"{'configuration':<28}{'new loss':>11}{'old loss':>11}{'drift|u':>11}{'drift_T u':>11}{'sec':>8}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        t0 = time.time()
        new_loss, old_loss, along, perp = train_config(cfg, theta_star, args, new_loader, abar, device, reg_ref, old_eval, new_eval)
        print(f"{cfg['name']:<28}{new_loss:>11.4f}{old_loss:>11.4f}{along:>11.4f}{perp:>11.4f}{time.time()-t0:>8.0f}")

    print()
    print("Lower old loss after fine-tuning = the old class was retained. Baseline forgets it;")
    print("Rank-1 EWC should keep old loss well below baseline while new loss stays close to baseline,")
    print("at no teacher cost. drift|u collapsing to ~0 confirms the penalty is doing its job.")


if __name__ == "__main__":
    main()
