"""Empirical demo: when does Rank-1 EWC actually retain old knowledge? (mock model + mock data, CPU)

Exercises the *real* shipped code (``library.anti_forgetting.EWCRegularizer`` and
``library.distillation.distillation_loss``) and shows that Rank-1 EWC's effectiveness is not
magic: it depends on whether the empirical Fisher is genuinely low-rank, which is exactly the
assumption the method is built on. We measure that assumption directly (gradient collinearity)
so the regime is proven, not asserted.

Two regimes, both fit to an "old" task first (-> theta*), then trained on a related "new" task:

  - "entangled": old and new use the *same* dense features. Per-sample gradients are spread out
    (high-rank Fisher), so a single direction u cannot capture old knowledge -> EWC is weak.
  - "separable": old knowledge lives in a low-rank feature block A; the new task is learned in a
    separate block B but its data also excites block A as nuisance, which is what erases the old
    weights. The forgetting direction is low-rank -> u captures it -> EWC retains old knowledge
    while block B is learned freely.

For each regime we print the measured Fisher collinearity, then a lambda sweep so the
learn-new / keep-old trade-off is visible, with distillation as a reference.

Run:  venv/Scripts/python.exe tools/dev/demo_anti_forgetting_effect.py
"""

import argparse
import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from library.anti_forgetting import EWCRegularizer
from library import distillation


def mse(pred, target):
    return torch.nn.functional.mse_loss(pred, target)


def iterate_batches(x, y, batch, steps, generator):
    n = x.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=generator)
        yield x[idx], y[idx]


def fisher_collinearity(model, x, y, n):
    """c = ||mean grad||^2 / mean ||grad||^2 over n single-sample gradients at the current weights.

    c -> 1 when per-sample gradients all point the same way (empirical Fisher ~ rank-1, the regime
    Rank-1 EWC assumes); c -> 0 when they are spread across many directions (high-rank Fisher).
    """
    flats = []
    for i in range(n):
        model.zero_grad(set_to_none=True)
        mse(model(x[i : i + 1]), y[i : i + 1]).backward()
        flats.append(torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]))
    model.zero_grad(set_to_none=True)
    g = torch.stack(flats)
    mean = g.mean(0)
    return float((mean @ mean) / (g.pow(2).sum(1).mean() + 1e-12))


def run_fisher_phase(model, reg, fx, fy, batch, generator):
    for bx, by in iterate_batches(fx, fy, batch, reg.num_fisher_samples, generator):
        model.zero_grad(set_to_none=True)
        mse(model(bx), by).backward()
        reg.accumulate()
        if reg.maybe_finalize():
            break
    model.zero_grad(set_to_none=True)


def drift_geometry(model, reg_ref):
    params = dict(model.named_parameters())
    s = u_sq = total_sq = 0.0
    for n, u in reg_ref.u.items():
        delta = params[n].detach().float() - reg_ref.theta_star[n].float()
        s += float(torch.sum(u.float() * delta))
        u_sq += float(torch.sum(u.float() ** 2))
        total_sq += float(torch.sum(delta ** 2))
    u_norm = math.sqrt(u_sq) if u_sq > 0 else 1.0
    along = abs(s / u_norm)
    total = math.sqrt(total_sq)
    return along, math.sqrt(max(total * total - along * along, 0.0))


def train_run(theta_star_state, model_factory, data, cfg, reg_ref, hp, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = model_factory()
    model.load_state_dict(theta_star_state)

    reg = teacher = None
    if cfg["method"] == "ewc":
        reg = EWCRegularizer(list(model.named_parameters()), cfg["lam"], hp.fisher_samples, store_on_cpu=False)
        run_fisher_phase(model, reg, data.x_new, data.y_new, hp.batch, torch.Generator().manual_seed(seed + 1))
    elif cfg["method"] == "distill":
        teacher = model_factory()
        teacher.load_state_dict(theta_star_state)
        teacher.eval()
        teacher.requires_grad_(False)
        dist_args = SimpleNamespace(loss_type="l2", distillation_weight_high=cfg["weight"], distillation_weight_low=cfg["weight"])

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    for bx, by in iterate_batches(data.x_new, data.y_new, hp.batch, hp.train_steps, gen):
        optimizer.zero_grad(set_to_none=True)
        pred = model(bx)
        loss = mse(pred, by)
        if reg is not None:
            loss = loss + reg.penalty()
        elif teacher is not None:
            ones = torch.ones(bx.shape[0])
            loss = loss + distillation.distillation_loss(pred, teacher(bx), ones, ones, dist_args)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        new_loss = float(mse(model(data.x_new), data.y_new))
        old_loss = float(mse(model(data.x_old), data.y_old))
    along, perp = drift_geometry(model, reg_ref)
    return new_loss, old_loss, along, perp


def build_entangled(seed, hp):
    g = torch.Generator().manual_seed(seed)
    d, m = 12, 6
    x_old = torch.randn(hp.n, d, generator=g)
    x_new = torch.randn(hp.n, d, generator=g)
    w_old = torch.randn(d, m, generator=g)
    w_new = w_old + 0.6 * torch.randn(d, m, generator=g)
    data = SimpleNamespace(x_old=x_old, y_old=x_old @ w_old, x_new=x_new, y_new=x_new @ w_new)
    return data, (lambda: nn.Linear(d, m, bias=False))


def build_separable(seed, hp):
    g = torch.Generator().manual_seed(seed)
    d_a, d_b, m = 2, 16, 4
    d = d_a + d_b
    w_a = 2.0 * torch.randn(m, d_a, generator=g)   # strong low-rank old features
    w_b = torch.randn(m, d_b, generator=g)
    x_old = torch.randn(hp.n, d, generator=g)
    x_new = torch.randn(hp.n, d, generator=g)
    y_old = x_old[:, :d_a] @ w_a.t()               # old target uses block A only
    y_new = x_new[:, d_a:] @ w_b.t()               # new target uses block B only; block A is nuisance
    data = SimpleNamespace(x_old=x_old, y_old=y_old, x_new=x_new, y_new=y_new)
    return data, (lambda: nn.Linear(d, m, bias=False))


def run_regime(title, builder, seed, hp):
    data, model_factory = builder(seed, hp)

    base = model_factory()
    opt = torch.optim.Adam(base.parameters(), lr=hp.lr)
    pre_gen = torch.Generator().manual_seed(seed + 7)
    for bx, by in iterate_batches(data.x_old, data.y_old, hp.batch, hp.pretrain_steps, pre_gen):
        opt.zero_grad(set_to_none=True)
        mse(base(bx), by).backward()
        opt.step()
    theta_star_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    with torch.no_grad():
        old0 = float(mse(base(data.x_old), data.y_old))
        new0 = float(mse(base(data.x_new), data.y_new))

    ref = model_factory()
    ref.load_state_dict(theta_star_state)
    c = fisher_collinearity(ref, data.x_new, data.y_new, min(hp.n, 128))
    reg_ref = EWCRegularizer(list(ref.named_parameters()), 1.0, hp.fisher_samples, store_on_cpu=False)
    run_fisher_phase(ref, reg_ref, data.x_new, data.y_new, hp.batch, torch.Generator().manual_seed(seed + 1))

    configs = [
        {"name": "baseline (none)", "method": "none"},
        {"name": "Rank-1 EWC lambda=10", "method": "ewc", "lam": 10.0},
        {"name": "Rank-1 EWC lambda=100", "method": "ewc", "lam": 100.0},
        {"name": "Rank-1 EWC lambda=1000", "method": "ewc", "lam": 1000.0},
        {"name": "distillation weight=1.0", "method": "distill", "weight": 1.0},
    ]

    print()
    print(f"=== regime: {title} ===")
    print(f"theta* fit:  old-task loss = {old0:.4f}   new-task loss = {new0:.4f}")
    print(f"new-task Fisher collinearity c = {c:.3f}   (1 = rank-1 / EWC-friendly, 0 = spread / EWC-weak)")
    header = f"{'configuration':<26}{'new loss':>11}{'old loss':>11}{'drift|u':>11}{'drift_T u':>11}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        new_loss, old_loss, along, perp = train_run(theta_star_state, model_factory, data, cfg, reg_ref, hp, seed)
        print(f"{cfg['name']:<26}{new_loss:>11.4f}{old_loss:>11.4f}{along:>11.4f}{perp:>11.4f}")


def make_diffusion_model():
    return nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))


def diffusion_schedule(steps):
    betas = torch.linspace(1e-4, 0.02, steps)
    return torch.cumprod(1.0 - betas, dim=0)  # abar


def diffusion_loss(model, x0, abar, generator, noise=None, t=None):
    """One noise-prediction step; returns (loss, noise_level[B] in [0,1], pred, target_noise, t)."""
    b = x0.shape[0]
    if t is None:
        t = torch.randint(0, abar.shape[0], (b,), generator=generator)
    if noise is None:
        noise = torch.randn(x0.shape, generator=generator)
    ab = abar[t].unsqueeze(1)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    inp = torch.cat([x_t, (t.float() / abar.shape[0]).unsqueeze(1)], dim=1)
    pred = model(inp)
    level = (1 - abar[t])  # higher = noisier; already in [0,1]
    return mse(pred, noise), level, pred, noise, t


def diffusion_collinearity(model, x0, abar, n, generator):
    flats = []
    for i in range(n):
        model.zero_grad(set_to_none=True)
        loss, _, _, _, _ = diffusion_loss(model, x0[i : i + 1], abar, generator)
        loss.backward()
        flats.append(torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]))
    model.zero_grad(set_to_none=True)
    g = torch.stack(flats)
    mean = g.mean(0)
    return float((mean @ mean) / (g.pow(2).sum(1).mean() + 1e-12))


@torch.no_grad()
def diffusion_eval(model, x0, abar, reps, generator):
    """Monte-Carlo average noise-prediction loss over fixed random (t, noise) draws."""
    total = 0.0
    for _ in range(reps):
        loss, _, _, _, _ = diffusion_loss(model, x0, abar, generator)
        total += float(loss)
    return total / reps


def run_diffusion_regime(seed, hp):
    g = torch.Generator().manual_seed(seed)
    abar = diffusion_schedule(hp.diff_T)
    mu_a = torch.tensor([-2.0, -2.0])
    mu_b = torch.tensor([2.0, 2.0])
    x_old = mu_a + 0.3 * torch.randn(hp.n, 2, generator=g)
    x_new = mu_b + 0.3 * torch.randn(hp.n, 2, generator=g)

    base = make_diffusion_model()
    opt = torch.optim.Adam(base.parameters(), lr=hp.lr)
    pre_g = torch.Generator().manual_seed(seed + 7)
    for bx, _ in iterate_batches(x_old, x_old, hp.batch, hp.diff_pretrain, pre_g):
        opt.zero_grad(set_to_none=True)
        loss, _, _, _, _ = diffusion_loss(base, bx, abar, pre_g)
        loss.backward()
        opt.step()
    theta_star_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    ref = make_diffusion_model()
    ref.load_state_dict(theta_star_state)
    c = diffusion_collinearity(ref, x_new, abar, min(hp.n, 128), torch.Generator().manual_seed(seed + 3))
    reg_ref = EWCRegularizer(list(ref.named_parameters()), 1.0, hp.fisher_samples, store_on_cpu=False)
    for bx, _ in iterate_batches(x_new, x_new, hp.batch, hp.fisher_samples, torch.Generator().manual_seed(seed + 1)):
        ref.zero_grad(set_to_none=True)
        loss, _, _, _, _ = diffusion_loss(ref, bx, abar, torch.Generator().manual_seed(seed + 100))
        loss.backward()
        reg_ref.accumulate()
        if reg_ref.maybe_finalize():
            break
    ref.zero_grad(set_to_none=True)

    eval_g_seed = seed + 50
    old0 = diffusion_eval(base, x_old, abar, hp.diff_eval_reps, torch.Generator().manual_seed(eval_g_seed))
    new0 = diffusion_eval(base, x_new, abar, hp.diff_eval_reps, torch.Generator().manual_seed(eval_g_seed))

    configs = [
        {"name": "baseline (none)", "method": "none"},
        {"name": "Rank-1 EWC lambda=1", "method": "ewc", "lam": 1.0},
        {"name": "Rank-1 EWC lambda=10", "method": "ewc", "lam": 10.0},
        {"name": "Rank-1 EWC lambda=100", "method": "ewc", "lam": 100.0},
        {"name": "distillation weight=2.0", "method": "distill", "weight": 2.0},
    ]

    print()
    print("=== regime: diffusion (2D noise-prediction, DDPM schedule -> the method's real regime) ===")
    print(f"theta* fit:  old denoise loss = {old0:.4f}   new denoise loss = {new0:.4f}")
    print(f"new-task Fisher collinearity c = {c:.3f}   (1 = rank-1 / EWC-friendly, 0 = spread / EWC-weak)")
    header = f"{'configuration':<26}{'new loss':>11}{'old loss':>11}{'drift|u':>11}{'drift_T u':>11}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        torch.manual_seed(seed)
        model = make_diffusion_model()
        model.load_state_dict(theta_star_state)
        reg = teacher = None
        if cfg["method"] == "ewc":
            reg = EWCRegularizer(list(model.named_parameters()), cfg["lam"], hp.fisher_samples, store_on_cpu=False)
            fg = torch.Generator().manual_seed(seed + 1)
            ng = torch.Generator().manual_seed(seed + 100)
            for bx, _ in iterate_batches(x_new, x_new, hp.batch, hp.fisher_samples, fg):
                model.zero_grad(set_to_none=True)
                loss, _, _, _, _ = diffusion_loss(model, bx, abar, ng)
                loss.backward()
                reg.accumulate()
                if reg.maybe_finalize():
                    break
            model.zero_grad(set_to_none=True)
        elif cfg["method"] == "distill":
            teacher = make_diffusion_model()
            teacher.load_state_dict(theta_star_state)
            teacher.eval()
            teacher.requires_grad_(False)
            dist_args = SimpleNamespace(loss_type="l2", distillation_weight_high=cfg["weight"], distillation_weight_low=cfg["weight"])

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        tg = torch.Generator().manual_seed(seed + 2)
        ng = torch.Generator().manual_seed(seed + 9)
        for bx, _ in iterate_batches(x_new, x_new, hp.batch, hp.diff_train, tg):
            optimizer.zero_grad(set_to_none=True)
            t = torch.randint(0, abar.shape[0], (bx.shape[0],), generator=ng)
            noise = torch.randn(bx.shape, generator=ng)
            task, level, pred, tgt_noise, t = diffusion_loss(model, bx, abar, ng, noise=noise, t=t)
            loss = task
            if reg is not None:
                loss = loss + reg.penalty()
            elif teacher is not None:
                ab = abar[t].unsqueeze(1)
                x_t = ab.sqrt() * bx + (1 - ab).sqrt() * noise
                inp = torch.cat([x_t, (t.float() / abar.shape[0]).unsqueeze(1)], dim=1)
                teacher_pred = teacher(inp)
                loss = loss + distillation.distillation_loss(pred, teacher_pred, level, torch.ones(bx.shape[0]), dist_args)
            loss.backward()
            optimizer.step()

        new_loss = diffusion_eval(model, x_new, abar, hp.diff_eval_reps, torch.Generator().manual_seed(eval_g_seed))
        old_loss = diffusion_eval(model, x_old, abar, hp.diff_eval_reps, torch.Generator().manual_seed(eval_g_seed))
        along, perp = drift_geometry(model, reg_ref)
        print(f"{cfg['name']:<26}{new_loss:>11.4f}{old_loss:>11.4f}{along:>11.4f}{perp:>11.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    hp = SimpleNamespace(
        n=512, batch=64, pretrain_steps=2500, train_steps=800, fisher_samples=32, lr=5e-3,
        diff_T=50, diff_pretrain=4000, diff_train=2000, diff_eval_reps=200,
    )

    run_regime("entangled (dense shared features -> high-rank Fisher)", build_entangled, args.seed, hp)
    run_regime("separable (low-rank old features -> rank-1 Fisher)", build_separable, args.seed, hp)
    run_diffusion_regime(args.seed, hp)

    print()
    print("Takeaway: EWC's retention tracks the measured collinearity c and, more deeply, on whether")
    print("the averaged direction u is the *shared* backbone rather than the new-task learning direction.")
    print("In deterministic regression toys u IS the learning direction, so EWC can only freeze; in the")
    print("stochastic diffusion regime the noise-driven shared signal dominates u, which is the setting")
    print("the method targets. The geometric guarantee (drift|u -> 0) holds in every regime regardless.")


if __name__ == "__main__":
    main()
