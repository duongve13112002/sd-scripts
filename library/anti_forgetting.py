"""Shared anti-forgetting utilities.

Hosts the cross-cutting pieces of the anti-forgetting feature set:
- the adaptive-lambda controller (a thermostat for a soft penalty's strength), and
- the argument validator that resolves conflicts between the methods.

The methods constrain different spaces: replay (data), output distillation (output),
and — in later steps — Rank-1 EWC and OPLoRA (parameter). The controller and validator
are model-agnostic and CPU-testable. The validator auto-resolves quality conflicts toward
the better method (with a warning) and fails fast on mode mismatches.
"""

import argparse
import logging
from typing import Optional

import torch

from library import distillation
from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def is_adaptive_lambda_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "adaptive_lambda", False))


def _has_soft_penalty(args: argparse.Namespace) -> bool:
    # Adaptive lambda scales a soft penalty's strength, so it needs one to act on.
    # Output distillation is the current soft penalty; Rank-1 EWC (a later step) also qualifies.
    return distillation.is_enabled(args) or getattr(args, "ewc_lambda", 0.0) > 0.0


def verify_anti_forgetting_args(args: argparse.Namespace) -> None:
    """Resolve conflicts between anti-forgetting methods.

    Quality conflicts (both valid, one strictly better) are auto-resolved toward the
    better method with a warning; mode mismatches fail fast. Called from
    ``verify_training_args`` so it runs for every trainer.

    Mode-specific precedence (Rank-1 EWC vs distillation, OPLoRA vs distillation) is added
    together with those features, since their arguments do not exist yet.
    """
    if is_adaptive_lambda_enabled(args) and not _has_soft_penalty(args):
        logger.warning(
            "adaptive_lambda is enabled but no soft-penalty method (output distillation) is active, "
            "so it has nothing to scale; disabling adaptive_lambda."
        )
        args.adaptive_lambda = False


def _reduce_scalar(value, accelerator) -> float:
    """Reduce a per-rank scalar loss to a single float, averaged across ranks so every
    process computes the same adaptive coefficient (DDP-consistent)."""
    if isinstance(value, torch.Tensor):
        v = value.detach()
        if accelerator is not None:
            v = accelerator.reduce(v, reduction="mean")
        return float(v.item())
    return float(value)


class AdaptiveLambdaController:
    """EMA thermostat for a soft-penalty coefficient.

    Tracks the ratio of the preservation (penalty) loss to the task loss and scales the
    penalty so it grows when forgetting grows and relaxes when the new task is hard to
    learn. The coefficient multiplies the existing (noise-dependent) penalty, so the noise
    profile from --distillation_weight_high/low is preserved; the controller only modulates
    the overall strength over time.
    """

    def __init__(self, args: argparse.Namespace):
        self.ema = float(args.adaptive_lambda_ema)
        self.base = float(args.adaptive_lambda_base)
        self.min_coeff = float(args.adaptive_lambda_min)
        self.max_coeff = float(args.adaptive_lambda_max)
        self.eps = 1e-8
        self.r_bar: Optional[float] = None

    def update(self, preserve_loss, task_loss, accelerator=None) -> float:
        preserve = _reduce_scalar(preserve_loss, accelerator)
        task = _reduce_scalar(task_loss, accelerator)
        r = preserve / (task + self.eps)
        if self.r_bar is None:
            self.r_bar = r
        else:
            self.r_bar = self.ema * self.r_bar + (1.0 - self.ema) * r
        coeff = self.base * self.r_bar
        return float(min(max(coeff, self.min_coeff), self.max_coeff))


def create_adaptive_lambda_controller(args: argparse.Namespace) -> Optional[AdaptiveLambdaController]:
    return AdaptiveLambdaController(args) if is_adaptive_lambda_enabled(args) else None


def add_adaptive_penalty(task_loss, penalty_term, controller: Optional[AdaptiveLambdaController], accelerator=None):
    """Add a soft penalty to the task loss, optionally scaled by the adaptive controller.

    ``task_loss`` is the task loss; ``penalty_term`` is the (already noise-weighted) penalty
    such as the distillation distance. When a controller is present, the penalty is scaled by
    the adaptive coefficient derived from the penalty/task ratio.
    """
    if controller is not None:
        coeff = controller.update(penalty_term, task_loss, accelerator)
        penalty_term = coeff * penalty_term
    return task_loss + penalty_term
