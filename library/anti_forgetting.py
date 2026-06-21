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

    Mode rejections that need to know the training mode live where the mode is known:
    EWC-on-LoRA in the LoRA trainer, and OPLoRA-on-full-fine-tune via argparse (the OPLoRA
    arguments are only registered by the LoRA parser). The OPLoRA precedence here is
    getattr-guarded, so it is a no-op when those arguments are absent (full fine-tune).
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
        if int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
            raise ValueError(
                "Rank-1 EWC is incompatible with --blocks_to_swap: block swapping leaves the trainable weights "
                "split across CPU and GPU, so the EWC penalty cannot be summed across them. Disable one of them."
            )
    elif getattr(args, "ewc_reference_model_path", None):
        logger.warning(
            "--ewc_reference_model_path is set but Rank-1 EWC is disabled (--ewc_lambda is 0), so the reference "
            "model is ignored. Set --ewc_lambda > 0 to anchor training to it."
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

    # Quality conflict: LoRA OPLoRA supersedes distillation (hard top-k guarantee, no teacher
    # forward), so keep OPLoRA and disable distillation. getattr-guarded so it is a no-op on
    # full fine-tune runs, where --oplora is not even a registered argument.
    if getattr(args, "oplora", False) and distillation.is_enabled(args):
        logger.warning(
            "OPLoRA and output distillation are both enabled; OPLoRA supersedes distillation "
            "(hard orthogonal guarantee, no teacher forward), so disabling distillation."
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

    def __init__(self, named_params, lam: float, num_fisher_samples: int, store_on_cpu: bool,
                 accelerator=None, buffer_dtype: torch.dtype = torch.float32, reference_params=None):
        self.lam = float(lam)
        self.num_fisher_samples = int(num_fisher_samples)
        self.store_on_cpu = bool(store_on_cpu)
        self.buffer_dtype = buffer_dtype
        self.params = [(n, p) for n, p in named_params if p.requires_grad]
        # theta* defaults to the trained weights themselves. When reference_params is given it is the
        # weights of a separate model (matched by name), so the penalty anchors training to that model
        # instead of the initial weights (e.g. continue fine-tuning B while staying anchored to base A).
        reference = {n: p for n, p in reference_params} if reference_params is not None else None
        self.theta_star = {}
        self.u = {}
        for n, p in self.params:
            dev = torch.device("cpu") if self.store_on_cpu else p.device
            if reference is not None:
                if n not in reference:
                    raise ValueError(
                        f"--ewc_reference_model_path is missing the trained parameter '{n}'. The reference model "
                        f"must share the architecture of the trained model; it anchors the denoiser, so disable "
                        f"text-encoder training or omit --ewc_reference_model_path."
                    )
                src = reference[n]
                if src.shape != p.shape:
                    raise ValueError(
                        f"--ewc_reference_model_path parameter '{n}' has shape {tuple(src.shape)} but the trained "
                        f"parameter has shape {tuple(p.shape)}; the reference model architecture does not match."
                    )
            else:
                src = p
            # theta* is a snapshot of the anchor weight, so storing it at the weight's own precision
            # (buffer_dtype) loses nothing while halving VRAM for bf16/fp16 weights; the penalty
            # difference is still computed in fp32 (see penalty()). It is placed on the trained
            # parameter's device so the per-step penalty needs no extra transfer.
            self.theta_star[n] = src.detach().to(dev, dtype=self.buffer_dtype, copy=True)
            # u is accumulated and stored at buffer_dtype on theta*'s device: fast and low-VRAM.
            # With bf16/fp16 the running sum slightly underestimates u's magnitude over many
            # micro-batches (swamping), but EWC's gradients are near-collinear so u's *direction*
            # is preserved and lambda absorbs the magnitude; the per-step penalty difference is still
            # computed in fp32 (penalty()), which is the precision that matters most. Use fp32 buffers
            # for strict fp32 accumulation (more VRAM), or --ewc_buffers_on_cpu to keep them off GPU.
            self.u[n] = torch.zeros_like(self.theta_star[n])
        self.count = 0
        self.collecting = self.num_fisher_samples > 0
        self.ready = False
        # The Fisher phase runs no optimizer step, so nothing else logs while it collects u.
        # Announce it (main process only, so a multi-GPU run does not repeat the line per rank)
        # and report progress, so a silent multi-minute phase is not mistaken for a hang.
        self._log_enabled = accelerator is None or accelerator.is_main_process
        self._log_every = max(1, self.num_fisher_samples // 10)
        if self.collecting and self._log_enabled:
            logger.info(
                f"Rank-1 EWC: Fisher phase started, estimating the dominant gradient direction over "
                f"{self.num_fisher_samples} micro-batches per process (reference buffers in {self.buffer_dtype}). "
                f"Weights stay frozen and the progress bar does not advance until this finishes."
            )

    def accumulate(self) -> None:
        """Add the current per-batch gradients into the running Fisher sum. Call after
        ``backward`` and before ``zero_grad`` during the Fisher phase."""
        for n, p in self.params:
            if p.grad is not None:
                self.u[n] += p.grad.detach().to(self.u[n].device, dtype=self.buffer_dtype)
        self.count += 1
        if self._log_enabled and self.collecting and self.count < self.num_fisher_samples and self.count % self._log_every == 0:
            logger.info(f"Rank-1 EWC: Fisher phase {self.count}/{self.num_fisher_samples} micro-batches")

    def maybe_finalize(self, accelerator=None) -> bool:
        """Once enough samples are collected, average the Fisher sum, reduce it across ranks,
        and mark the penalty ready. Returns True on the step it finalizes."""
        if not self.collecting or self.count < self.num_fisher_samples:
            return False
        multi = accelerator is not None and accelerator.num_processes > 1
        for n in self.u:
            avg = self.u[n] / self.count
            if multi:
                avg = accelerator.reduce(avg.to(accelerator.device), reduction="mean")
            # store the finalized direction back on the buffer device at buffer_dtype (the fp32
            # accumulator is dropped here, freeing the extra Fisher-phase memory)
            self.u[n] = avg.to(device=self.theta_star[n].device, dtype=self.buffer_dtype)
        self.collecting = False
        self.ready = True
        if self._log_enabled:
            logger.info(
                f"Rank-1 EWC: Fisher phase done, u estimated over {self.num_fisher_samples} micro-batches "
                f"per process; training and the progress bar start now."
            )
        return True

    def penalty(self):
        """Global EWC penalty tensor ``lambda * (u^T (theta - theta*))^2`` on the param device,
        differentiable through the live weights.

        ``p.float()`` makes the ``theta - theta*`` difference fp32 (so bf16/fp16 buffers do not lose
        it to cancellation), and that fp32 result promotes ``u`` in the product. ``u`` and ``theta*``
        are therefore left at the buffer dtype: upcasting them to fp32 would only allocate a full
        fp32 copy of the model every step (slow) without changing the result. They are only moved
        when the buffers live on CPU (``--ewc_buffers_on_cpu``)."""
        s = None
        for n, p in self.params:
            u = self.u[n]
            theta_star = self.theta_star[n]
            if theta_star.device != p.device:
                u = u.to(p.device)
                theta_star = theta_star.to(p.device)
            term = torch.sum(u * (p.float() - theta_star))
            s = term if s is None else s + term
        return self.lam * (s * s)


_EWC_BUFFER_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def create_ewc_regularizer(args: argparse.Namespace, models, accelerator=None,
                           reference_named_params=None) -> Optional[EWCRegularizer]:
    """Build the EWC regularizer over the trainable params of ``models`` (the denoiser, and
    any other fine-tuned modules). Snapshot ``theta*`` here, before any optimizer step.

    ``reference_named_params`` optionally provides the (name, param) pairs of a separate anchor
    model loaded by the caller (``--ewc_reference_model_path``); when given, ``theta*`` is taken
    from it instead of the trained weights, so the penalty pulls training toward that model."""
    if not is_ewc_enabled(args):
        return None
    named = []
    for m in models:
        named.extend(m.named_parameters())
    buffer_dtype = _EWC_BUFFER_DTYPES[getattr(args, "ewc_buffer_dtype", "fp32")]
    if buffer_dtype != torch.float32:
        trainable = next((p for _, p in named if p.requires_grad), None)
        if trainable is not None and trainable.dtype == torch.float32:
            logger.warning(
                "--ewc_buffer_dtype is reduced precision but the trainable weights are fp32; this can "
                "quantize away small weight drift and weaken EWC. Use fp32 buffers when training in fp32; "
                "reduced-precision buffers are intended for bf16/fp16 weights (e.g. --full_bf16)."
            )
    if reference_named_params is not None and (accelerator is None or accelerator.is_main_process):
        logger.info(
            f"Rank-1 EWC: anchoring theta* to the reference model at {args.ewc_reference_model_path} "
            f"(instead of the initial trained weights)."
        )
    return EWCRegularizer(
        named, args.ewc_lambda, args.ewc_fisher_samples,
        getattr(args, "ewc_buffers_on_cpu", False), accelerator, buffer_dtype,
        reference_named_params,
    )
