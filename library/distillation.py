"""Output prediction-matching distillation against the frozen base model.

This implements an LwF-style (Learning without Forgetting) regularizer for
diffusion / flow-matching training. The total loss becomes

    L = L_task(student, real_target) + lambda(noise_level) * dist(student, teacher)

where the teacher is the base model (for LoRA/network training, the same model
with the adapter disabled via ``network.set_multiplier(0.0)``; for full
fine-tuning, a frozen copy of the initial weights). The distillation term pulls
the student's prediction toward the base prediction so base knowledge is kept
while the task term still adapts the model.

``lambda`` depends on the per-sample normalized noise level (1 = pure noise,
0 = clean), not on SNR: SNR / ``--min_snr_gamma`` is only defined for the
DDPM-schedule families (SD1.x, SDXL), whereas the flow-matching families
(SD3, FLUX, Lumina, Anima, Hunyuan) use rectified-flow ``sigmas``. Every family
already computes a noise level in [0, 1] (``sigmas`` for flow models,
``timesteps / num_train_timesteps`` for DDPM), so the schedule is shared while
each trainer feeds its own noise level.

Diffusion is coarse-to-fine: high-noise steps carry global structure / semantics
(the knowledge worth preserving), low-noise steps carry texture / style (what we
usually want to adapt). Weighting distillation higher at high noise therefore
anchors concepts to the base while leaving detail learning free.
"""

import argparse
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from library import loss as loss_util

logger = logging.getLogger(__name__)


def is_enabled(args: argparse.Namespace) -> bool:
    """Distillation is active when either end of the lambda schedule is positive."""
    high = getattr(args, "distillation_weight_high", 0.0) or 0.0
    low = getattr(args, "distillation_weight_low", 0.0) or 0.0
    return high > 0.0 or low > 0.0


def lambda_for_noise_level(noise_level: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Per-sample distillation weight, linearly interpolated over the noise level.

    Args:
        noise_level: tensor of shape [B] in [0, 1], where 1 is pure noise.
    Returns:
        tensor of shape [B] with the lambda weight for each sample.
    """
    low = float(getattr(args, "distillation_weight_low", 0.0) or 0.0)
    high = float(getattr(args, "distillation_weight_high", 0.0) or 0.0)
    level = noise_level.clamp(0.0, 1.0)
    return low + (high - low) * level


def distillation_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    noise_level: torch.Tensor,
    loss_weights: torch.Tensor,
    args: argparse.Namespace,
    huber_c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scalar distillation loss for one batch.

    The distance between the student and teacher predictions uses the same
    ``--loss_type`` (and Huber threshold ``huber_c``) as the task loss, so the
    two terms are always consistent. The teacher prediction is detached so
    gradients only flow through the student. The per-element distance is reduced
    over all non-batch dims, then weighted per sample by ``lambda(noise_level)``
    and the dataset's ``loss_weights`` before averaging over the batch. This term
    is intentionally kept out of ``post_process_loss`` (min-SNR etc.) because it
    carries its own noise-dependent weighting.
    """
    student = student_pred.float()
    teacher = teacher_pred.float().detach()

    per_element = loss_util.conditional_loss(student, teacher, args.loss_type, "none", huber_c)
    per_sample = per_element.mean(dim=list(range(1, per_element.ndim)))  # [B]

    lam = lambda_for_noise_level(noise_level.to(per_sample.device).float(), args)
    weighted = per_sample * lam * loss_weights.to(per_sample.device).float()
    return weighted.mean()


def normalized_noise_level_from_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
    """Flow-matching noise level: ``sigmas`` already lives in [0, 1] (1 = noise).

    ``sigmas`` is broadcast-shaped (e.g. [B, 1, 1, 1]); collapse it to [B].
    """
    return sigmas.detach().float().reshape(sigmas.shape[0], -1)[:, 0]


def normalized_noise_level_from_timesteps(timesteps: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    """DDPM noise level: normalize discrete timesteps to [0, 1] (1 = most noise)."""
    return timesteps.detach().float().reshape(timesteps.shape[0]) / float(num_train_timesteps)


def teacher_path(args: argparse.Namespace) -> str:
    """Where the full fine-tune teacher is loaded from (defaults to the base being fine-tuned)."""
    return getattr(args, "distillation_teacher_path", None) or args.pretrained_model_name_or_path


def prepare_teacher(
    teacher: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    *,
    supports_block_swap: bool,
    supports_fp8: bool,
) -> Tuple[nn.Module, bool]:
    """Freeze a full fine-tune teacher denoiser and apply optional fp8 / block-swap for VRAM.

    The teacher is used only for no-grad forward passes, so it is set to eval and has grad
    disabled. fp8 (generic per-Linear) and block swap are opt-in VRAM savers; both fall back to a
    plain device placement (with a warning) for models that do not support them. Only the denoiser
    is duplicated here; the VAE and text encoders are already frozen and shared by the caller.

    Returns the (frozen) teacher and whether block swapping is active.
    """
    teacher.eval()
    teacher.requires_grad_(False)

    if getattr(args, "distillation_teacher_fp8", False):
        if not supports_fp8:
            logger.warning("--distillation_teacher_fp8 is not supported for this model; keeping the teacher in its loaded dtype")
        else:
            from library import fp8_optimization_utils

            sd = teacher.state_dict()
            sd = fp8_optimization_utils.optimize_state_dict_with_fp8(sd, device, None, None, move_to_device=True)
            fp8_optimization_utils.apply_fp8_monkey_patch(teacher, sd, use_scaled_mm=False)
            teacher.load_state_dict(sd, strict=False, assign=True)

    blocks = int(getattr(args, "distillation_teacher_blocks_to_swap", 0) or 0)
    is_swapping = False
    if blocks > 0 and supports_block_swap:
        logger.info(f"distillation teacher: enable block swap, blocks_to_swap={blocks}")
        teacher.enable_block_swap(blocks, device)
        teacher.move_to_device_except_swap_blocks(device)
        is_swapping = True
    else:
        if blocks > 0 and not supports_block_swap:
            logger.warning(
                "--distillation_teacher_blocks_to_swap is not supported for this model; placing the whole teacher on the device"
            )
        teacher.to(device)
    return teacher, is_swapping


def before_teacher_forward(teacher: nn.Module, is_swapping: bool) -> None:
    """Call before each teacher forward when block swapping is active."""
    if is_swapping:
        teacher.prepare_block_swap_before_forward()
