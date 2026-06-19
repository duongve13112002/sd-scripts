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


def is_ewc_enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "ewc_lambda", 0.0) > 0.0


def _has_soft_penalty(args: argparse.Namespace) -> bool:
    # Adaptive lambda scales a soft penalty's strength, so it needs one to act on.
    # Output distillation is the current soft penalty; Rank-1 EWC (a later step) also qualifies.
    return distillation.is_enabled(args) or getattr(args, "ewc_lambda", 0.0) > 0.0


def verify_anti_forgetting_args(args: argparse.Namespace) -> None:
    """Resolve conflicts between anti-forgetting methods.

    Quality conflicts (both valid, one strictly better) are auto-resolved toward the
    better method with a warning; mode mismatches fail fast. Called from
    ``verify_training_args`` so it runs for every trainer.

    OPLoRA-vs-distillation precedence is added together with that feature, since its
    arguments do not exist yet. The LoRA-mode rejection of EWC lives in the LoRA trainer,
    which knows it is a network trainer.
    """
    # EWC configuration validation (full fine-tune feature; LoRA is rejected in the LoRA trainer).
    if is_ewc_enabled(args):
        if getattr(args, "ewc_fisher_samples", 0) <= 0:
            raise ValueError("--ewc_lambda > 0 requires --ewc_fisher_samples > 0 to estimate the Fisher direction.")
        if getattr(args, "fused_backward_pass", False) or getattr(args, "blockwise_fused_optimizers", False):
            raise ValueError(
                "Rank-1 EWC is incompatible with --fused_backward_pass / --blockwise_fused_optimizers: the optimizer "
                "steps inside the backward hook, which would update weights during the EWC Fisher phase. "
                "Disable EWC or the fused optimizer."
            )

    # Quality conflict: full fine-tune EWC supersedes distillation (parameter-space, no
    # resident GPU teacher), so keep EWC and disable distillation.
    if is_ewc_enabled(args) and distillation.is_enabled(args):
        logger.warning(
            "Rank-1 EWC and output distillation are both enabled; EWC supersedes distillation "
            "(parameter-space penalty, no resident teacher), so disabling distillation."
        )
        args.distillation_weight_high = 0.0
        args.distillation_weight_low = 0.0

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


class EWCRegularizer:
    """Rank-1 Elastic Weight Consolidation for full fine-tuning.

    Penalizes drift of the trainable weights along the dominant Fisher direction, which in
    diffusion is well-approximated by the mean gradient ``u`` (per-sample gradients are
    strongly collinear at low SNR, so the empirical Fisher is ~rank-1, F ~ u u^T). The
    penalty is the global scalar ``lambda * (u^T (theta - theta*))^2``, where ``theta*`` is
    the initial weight snapshot. It needs no teacher model and no extra forward pass at
    train time, only one inner product.

    ``u`` is estimated over the first ``num_fisher_samples`` micro-batches (the Fisher phase)
    using the normal training loss and noise distribution, then averaged across ranks. The
    reference buffers (``u`` and ``theta*``) can live on CPU to save VRAM, at the cost of a
    per-step host-device transfer.
    """

    def __init__(self, named_params, lam: float, num_fisher_samples: int, store_on_cpu: bool):
        self.lam = float(lam)
        self.num_fisher_samples = int(num_fisher_samples)
        self.store_on_cpu = bool(store_on_cpu)
        self.params = [(n, p) for n, p in named_params if p.requires_grad]
        self.theta_star = {}
        self.u = {}
        for n, p in self.params:
            dev = torch.device("cpu") if self.store_on_cpu else p.device
            self.theta_star[n] = p.detach().to(dev, dtype=torch.float32, copy=True)
            self.u[n] = torch.zeros_like(self.theta_star[n])
        self.count = 0
        self.collecting = self.num_fisher_samples > 0
        self.ready = False

    def accumulate(self) -> None:
        """Add the current per-batch gradients into the running Fisher sum. Call after
        ``backward`` and before ``zero_grad`` during the Fisher phase."""
        for n, p in self.params:
            if p.grad is not None:
                self.u[n] += p.grad.detach().to(self.u[n].device, dtype=torch.float32)
        self.count += 1

    def maybe_finalize(self, accelerator=None) -> bool:
        """Once enough samples are collected, average the Fisher sum, reduce it across ranks,
        and mark the penalty ready. Returns True on the step it finalizes."""
        if not self.collecting or self.count < self.num_fisher_samples:
            return False
        multi = accelerator is not None and accelerator.num_processes > 1
        for n in self.u:
            self.u[n] /= self.count
            if multi:
                reduced = accelerator.reduce(self.u[n].to(accelerator.device), reduction="mean")
                self.u[n] = reduced.to("cpu") if self.store_on_cpu else reduced
        self.collecting = False
        self.ready = True
        return True

    def penalty(self):
        """Global EWC penalty tensor ``lambda * (u^T (theta - theta*))^2`` on the param device,
        differentiable through the live weights."""
        s = None
        for n, p in self.params:
            u = self.u[n].to(device=p.device, dtype=torch.float32)
            theta_star = self.theta_star[n].to(device=p.device, dtype=torch.float32)
            term = torch.sum(u * (p.float() - theta_star))
            s = term if s is None else s + term
        return self.lam * (s * s)


def create_ewc_regularizer(args: argparse.Namespace, models, accelerator=None) -> Optional[EWCRegularizer]:
    """Build the EWC regularizer over the trainable params of ``models`` (the denoiser, and
    any other fine-tuned modules). Snapshot ``theta*`` here, before any optimizer step."""
    if not is_ewc_enabled(args):
        return None
    named = []
    for m in models:
        named.extend(m.named_parameters())
    return EWCRegularizer(named, args.ewc_lambda, args.ewc_fisher_samples, getattr(args, "ewc_buffers_on_cpu", False))
