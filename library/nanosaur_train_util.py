# NanoSaur training utilities for sd-scripts
# Provides rectified-flow loss, sampling, argument parsing, and saving helpers.

import argparse
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from accelerate import Accelerator, PartialState
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm

from library import nanosaur_models, strategy_base, train_util
from library.device_utils import clean_memory_on_device
from library.nanosaur_models import NanoSaurTransformer2DModel
from library.nanosaur_utils import NanoSaurVAEWrapper
from library.safetensors_utils import mem_eff_save_file
import logging

logger = logging.getLogger(__name__)

MODEL_VERSION_NANOSAUR = "nanosaur"


# Timestep sampling


def sample_timesteps(
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    alpha: float = 2.0,
) -> torch.Tensor:
    """
    Sample diffusion timesteps t ∈ (0, 1] biased toward the middle.

    Uses the logistic-normal trick from the original NanoSaur training code:
        t = sigmoid(randn() + log(alpha))
    where alpha=2.0 biases toward t≈0.5 and avoids near-zero values.
    """
    mu = math.log(alpha)
    return torch.sigmoid(torch.randn((batch,), device=device, dtype=dtype) + mu)


# Noisy model input


def get_noisy_model_input_and_timesteps(
    args: argparse.Namespace,
    latents: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a noisy latent and sample diffusion timesteps.

    Rectified flow interpolation:
        zt = (1 - t) * x0 + t * z1

    Args:
        args: Training arguments (may contain ``time_sampling_alpha``).
        latents: Clean latents x0 (B, C, H, W).
        noise: Pure Gaussian noise z1 (B, C, H, W).
        device: Device for newly created tensors.
        dtype: Target dtype.

    Returns:
        (noisy_model_input, timesteps)
            noisy_model_input: zt, shape (B, C, H, W).
            timesteps: Sampled t values in (0, 1], shape (B,).
    """
    batch = latents.size(0)
    alpha = getattr(args, "time_sampling_alpha", 2.0)
    t = sample_timesteps(batch, device, dtype, alpha)  # (B,)
    shape = [batch] + [1] * (latents.dim() - 1)
    noisy = (1.0 - t.view(shape)) * latents + t.view(shape) * noise
    return noisy.to(dtype), t.to(dtype)


# Loss utilities


def get_flow_matching_loss(
    x0_pred: torch.Tensor,
    latents: torch.Tensor,
    zt: torch.Tensor,
    t: torch.Tensor,
    t_epsilon: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rectified-flow velocity loss.

    The model predicts x0; we convert to velocities with a clamped denominator
    to avoid instability at small t.

    velocity_pred   = (zt - x0_pred) / (t + epsilon)
    velocity_target = (zt - x0)      / (t + epsilon)

    Args:
        x0_pred: Predicted clean latent, shape (B, C, H, W).
        latents: Ground-truth clean latent x0, shape (B, C, H, W).
        zt: Noisy latent, shape (B, C, H, W).
        t: Timesteps, shape (B,).
        t_epsilon: Denominator clamp for numerical stability.

    Returns:
        (velocity_pred, velocity_target) — the loss caller does MSE.
    """
    shape = [t.size(0)] + [1] * (latents.dim() - 1)
    t_clamped = (t + t_epsilon).view(shape).to(latents.dtype)
    velocity_pred = (zt - x0_pred) / t_clamped
    velocity_target = (zt - latents) / t_clamped
    return velocity_pred, velocity_target


# Sampling / inference


def get_sampling_timesteps(
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    sample_shift: Optional[float] = None,
) -> torch.Tensor:
    """
    Build a linear schedule from t=1 down to t≈0 with optional shift.

    shift > 1 biases toward high-signal timesteps (recommended ~4.0 for 1024px).
    """
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)[:-1]
    if sample_shift is not None and sample_shift > 0:
        ts = sample_shift * ts / (1.0 + (sample_shift - 1.0) * ts)
    return ts


@torch.no_grad()
def rectified_flow_sample(
    model: NanoSaurTransformer2DModel,
    z: torch.Tensor,
    cond: torch.Tensor,
    null_cond: Optional[torch.Tensor],
    steps: int,
    guidance_scale: float,
    sample_shift: Optional[float] = 4.0,
    cfg_start: float = 0.03,
    cfg_end: float = 0.80,
    path_drop_guidance: bool = True,
    use_momentum_guidance: bool = True,
    mg_alpha: float = 0.5,
    mg_beta: float = 0.6,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Euler ODE sampler with optional classifier-free guidance and SPRINT.

    SPRINT optimisation: every other uncond CFG step uses ``uncond=True`` to
    skip the 22 global FlattenDiT blocks → ~2× faster uncond evaluation.

    Args:
        model: NanoSaurTransformer2DModel.
        z: Initial noise latent (B, C, H, W).
        cond: Conditional text embeddings (B, L, D).
        null_cond: Unconditional text embeddings, or None to disable CFG.
        steps: Number of Euler steps.
        guidance_scale: CFG scale.
        sample_shift: Timestep schedule shift (higher → bias toward large t).
        cfg_start: Fraction of steps from which CFG is applied.
        cfg_end: Fraction of steps until which CFG is applied.
        path_drop_guidance: If True, use SPRINT on odd uncond steps.
        use_momentum_guidance: Enable momentum-based guidance correction.
        mg_alpha / mg_beta: Momentum hyperparameters.
        show_progress: Show tqdm progress bar.

    Returns:
        Denoised latent tensor (B, C, H, W).
    """
    latents = z.clone()
    batch = latents.size(0)
    device = latents.device
    dtype = latents.dtype
    latent_shape = [1] + [1] * (latents.dim() - 1)
    timesteps = get_sampling_timesteps(steps, device, dtype, sample_shift)
    momentum: Optional[torch.Tensor] = None

    iterator = (
        tqdm(range(steps), desc="NanoSaur sample", leave=False)
        if show_progress
        else range(steps)
    )

    for idx in iterator:
        t_curr = timesteps[idx]
        t_next = (
            timesteps[idx + 1]
            if idx + 1 < steps
            else torch.tensor(0.0, device=device, dtype=dtype)
        )
        dt = t_curr - t_next
        t = t_curr.expand(batch)

        # Conditional prediction (always dense)
        model.prepare_block_swap_before_forward()
        x0_pred = model._forward(latents, t, cond)
        guided = (latents - x0_pred) / t_curr

        step_fraction = idx / steps
        apply_cfg = (
            null_cond is not None
            and cfg_start < step_fraction < cfg_end
        )
        if apply_cfg:
            # SPRINT: use sparse uncond path every other step
            use_sprint = path_drop_guidance and (idx % 2 == 1)
            model.prepare_block_swap_before_forward()
            x0_uncond = model._forward(latents, t, null_cond, uncond=use_sprint)
            unguided = (latents - x0_uncond) / t_curr
            guided = unguided + guidance_scale * (guided - unguided)

            if use_momentum_guidance:
                if momentum is None:
                    momentum = guided.clone()
                effective = guided + mg_alpha * (guided - momentum)
                momentum = (1.0 - mg_beta) * guided + mg_beta * momentum
                guided = effective

        latents = latents - dt.view(latent_shape) * guided

    return latents


# Sample image generation


@torch.no_grad()
def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: Optional[int],
    global_step: int,
    model: NanoSaurTransformer2DModel,
    vae: NanoSaurVAEWrapper,
    text_encoders: List,
    sample_prompts_te_outputs: Optional[Dict],
    prompt_replacement: Optional[Tuple[str, str]] = None,
):
    """Generate sample images during NanoSaur training."""
    if global_step == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if global_step % args.sample_every_n_steps != 0 or epoch is not None:
                return

    if args.sample_prompts is None:
        logger.warning("sample_prompts not set, skipping sample generation")
        return

    logger.info(f"Generating sample images at step {global_step}")

    distributed_state = PartialState()
    model = accelerator.unwrap_model(model)

    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    for prompt_dict in prompts:
        prompt: str = prompt_dict.get("prompt", "")
        negative_prompt: str = prompt_dict.get("negative_prompt", "")
        height = int(prompt_dict.get("height", 1024))
        width = int(prompt_dict.get("width", 1024))
        steps = int(prompt_dict.get("sample_steps", getattr(args, "sample_steps", 40)))
        guidance_scale = float(prompt_dict.get("scale", getattr(args, "sample_cfg", 7.0)))
        sample_shift = float(prompt_dict.get("sample_shift", getattr(args, "sample_shift", 4.0)))
        seed = prompt_dict.get("seed", None)
        seed = int(seed) if seed is not None else None

        if prompt_replacement is not None:
            prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

        # Retrieve or encode text conditioning
        cond_te = None
        if sample_prompts_te_outputs and prompt in sample_prompts_te_outputs:
            cond_te = sample_prompts_te_outputs[prompt]
        elif text_encoders is not None and text_encoders[0] is not None:
            tokens = tokenize_strategy.tokenize(prompt)
            cond_te = encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens)

        neg_te = None
        if sample_prompts_te_outputs and negative_prompt in sample_prompts_te_outputs:
            neg_te = sample_prompts_te_outputs[negative_prompt]
        elif text_encoders is not None and text_encoders[0] is not None:
            tokens = tokenize_strategy.tokenize(negative_prompt)
            neg_te = encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens)

        if cond_te is None:
            logger.error(f"Cannot encode prompt, skipping: {prompt}")
            continue

        # Unpack: (hidden_states, input_ids, attention_mask)
        cond_hidden = cond_te[0].to(accelerator.device)
        neg_hidden = neg_te[0].to(accelerator.device) if neg_te is not None else None

        # Latent shape: 96ch, 16x spatial downscale
        lat_h = height // 16
        lat_w = width // 16
        lat_c = nanosaur_models.MODEL_CHANNELS
        weight_dtype = next(model.parameters()).dtype

        generator = torch.Generator(device=accelerator.device)
        if seed is not None:
            generator.manual_seed(seed)
        z = torch.randn(1, lat_c, lat_h, lat_w, device=accelerator.device, dtype=weight_dtype, generator=generator)

        with accelerator.autocast():
            denoised = rectified_flow_sample(
                model=model,
                z=z,
                cond=cond_hidden,
                null_cond=neg_hidden,
                steps=steps,
                guidance_scale=guidance_scale,
                sample_shift=sample_shift,
                show_progress=distributed_state.is_local_main_process,
            )

        # Decode to image
        org_vae_device = vae.device
        vae.to(accelerator.device)
        decoded = vae.decode(denoised)  # (1, 3, H, W) in [-1, 1]
        vae.to(org_vae_device)
        clean_memory_on_device(accelerator.device)

        decoded = decoded.clamp(-1.0, 1.0).float().cpu()
        img_np = ((decoded[0].permute(1, 2, 0) + 1.0) * 127.5).numpy().astype(np.uint8)
        image = Image.fromarray(img_np)

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
        seed_suffix = f"_{seed}" if seed is not None else ""
        img_filename = f"{args.output_name + '_' if args.output_name else ''}{num_suffix}_{ts_str}{seed_suffix}.png"
        image.save(os.path.join(save_dir, img_filename))
        logger.info(f"Saved sample image: {img_filename}")

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    clean_memory_on_device(accelerator.device)


# Model saving


def save_nanosaur_model(
    ckpt_path: str,
    model: NanoSaurTransformer2DModel,
    sai_metadata: Optional[Dict],
    save_dtype: Optional[torch.dtype] = None,
    use_mem_eff_save: bool = False,
) -> None:
    """Save only the diffusion model weights to a safetensors file."""
    state_dict = {}
    for k, v in model.state_dict().items():
        if save_dtype is not None and v.dtype != save_dtype:
            v = v.detach().clone().to("cpu").to(save_dtype)
        else:
            v = v.detach().clone().to("cpu")
        state_dict[k] = v

    if use_mem_eff_save:
        mem_eff_save_file(state_dict, ckpt_path, metadata=sai_metadata)
    else:
        save_file(state_dict, ckpt_path, metadata=sai_metadata)
    logger.info(f"Saved NanoSaur model to {ckpt_path}")


def save_nanosaur_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    model: NanoSaurTransformer2DModel,
) -> None:
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None, args, False, True, False, is_stable_diffusion_ckpt=True
        )
        save_nanosaur_model(ckpt_file, model, sai_metadata, save_dtype, args.mem_eff_save)

    train_util.save_sd_model_on_train_end_common(
        args, True, True, epoch, global_step, sd_saver, None
    )


def save_nanosaur_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator: Accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    model: NanoSaurTransformer2DModel,
) -> None:
    def sd_saver(ckpt_file: str, epoch_no: int, global_step: int):
        sai_metadata = train_util.get_sai_model_spec(
            {}, args, False, True, False, is_stable_diffusion_ckpt=True
        )
        save_nanosaur_model(ckpt_file, model, sai_metadata, save_dtype, args.mem_eff_save)

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


# Argument parser additions


def add_nanosaur_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Add NanoSaur-specific command-line arguments to the parser."""
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=False,
        help="Path to the NanoSaur text encoder safetensors (contains Gemma3 weights + spiece_model). "
        "/ NanoSauのテキストエンコーダsafetensorsのパス (Gemma3の重みとspiece_modelを含む)。",
    )
    parser.add_argument(
        "--vae",
        type=str,
        required=False,
        help="Path to the NanoSaur VAE safetensors. "
        "/ NanoSaur VAE safetensorsのパス。",
    )
    parser.add_argument(
        "--time_sampling_alpha",
        type=float,
        default=2.0,
        help="Alpha parameter for logistic-normal timestep sampling. Higher values bias toward t=0.5. Default: 2.0 "
        "/ ロジスティック正規分布タイムステップサンプリングのアルファパラメータ。デフォルト: 2.0",
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=4.0,
        help="Timestep schedule shift for sampling. Higher values bias toward high-signal steps. Default: 4.0 "
        "/ サンプリングのタイムステップスケジュールシフト。デフォルト: 4.0",
    )
    parser.add_argument(
        "--sample_cfg",
        type=float,
        default=7.0,
        help="CFG guidance scale for sample generation. Default: 7.0 "
        "/ サンプル生成のCFGガイダンススケール。デフォルト: 7.0",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=40,
        help="Number of Euler steps for sample generation. Default: 40 "
        "/ サンプル生成のオイラーステップ数。デフォルト: 40",
    )
    parser.add_argument(
        "--cfg_start",
        type=float,
        default=0.03,
        help="Fraction of steps from which CFG is applied during sampling. Default: 0.03 "
        "/ サンプリング中にCFGが適用されるステップの割合。デフォルト: 0.03",
    )
    parser.add_argument(
        "--cfg_end",
        type=float,
        default=0.80,
        help="Fraction of steps until which CFG is applied during sampling. Default: 0.80 "
        "/ サンプリング中にCFGが適用されるステップの割合の終点。デフォルト: 0.80",
    )
    parser.add_argument(
        "--disable_sprint",
        action="store_true",
        default=False,
        help="Disable SPRINT (path drop guidance) optimization during sampling. "
        "/ サンプリング中のSPRINT (パスドロップガイダンス) 最適化を無効にする。",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=False,
        help="Use Flash Attention (requires flash_attn package). Falls back to SDPA if unavailable. "
        "/ Flash Attentionを使用する (flash_attnパッケージが必要)。",
    )
    parser.add_argument(
        "--use_sage_attn",
        action="store_true",
        default=False,
        help="Use SageAttention for faster inference (requires sageattention package). Inference only — not recommended for training. "
        "/ SageAttentionを使用して高速推論を行う (sageattentionパッケージが必要)。推論のみ推奨。",
    )
