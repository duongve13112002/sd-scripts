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
from typing import Optional

import torch
import torch.nn.functional as F


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
) -> torch.Tensor:
    """Scalar distillation loss for one batch.

    The teacher prediction is detached so gradients only flow through the
    student. The per-element distance is reduced over all non-batch dims, then
    weighted per sample by ``lambda(noise_level)`` and the dataset's
    ``loss_weights`` before averaging over the batch. This term is intentionally
    kept out of ``post_process_loss`` (min-SNR etc.) because it carries its own
    noise-dependent weighting.
    """
    student = student_pred.float()
    teacher = teacher_pred.float().detach()

    loss_type = getattr(args, "distillation_loss_type", "l2")
    if loss_type == "huber":
        delta = float(getattr(args, "distillation_huber_c", 1.0))
        per_element = F.huber_loss(student, teacher, reduction="none", delta=delta)
    else:
        per_element = (student - teacher) ** 2

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
