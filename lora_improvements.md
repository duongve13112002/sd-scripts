# LoRA Training Precision Improvements

## Overview

This document describes two precision-oriented features added to the Anima/FLUX LoRA training pipeline: **`lora_fp32_accumulation`** and **`attn_softmax_scale`**. Both target a common root cause — numerical precision loss in bfloat16 training — but they operate at different levels of the forward pass and carry different risk/reward profiles.

---

## The Problem: bf16 Precision Loss in LoRA Training

bfloat16 has only ~3 significant decimal digits (7.8 mantissa bits vs. fp32's 23). Two specific failure modes arise:

1. **LoRA delta vanishing**: The LoRA contribution (`down → up` matmul) produces a small delta relative to the base model output. When added in bf16, this delta can be quantized to zero — effectively making the LoRA layer a no-op for that forward pass.

2. **Softmax saturation**: When attention logits (Q·K scaled by 1/√d) are compressed into a narrow numerical range, bf16 softmax can't differentiate between tokens. Attention distributions become near-uniform, losing the model's ability to focus.

These issues compound: imprecise attention → noisy gradients → imprecise LoRA updates → slower or degraded convergence.

---

## Feature 1: `lora_fp32_accumulation`

**Flag:** `--lora_fp32_accumulation`
**Files:** `networks/lora_flux.py` (implementation), `networks/lora_anima.py` (inherits), `library/anima_train_utils.py` (arg definition)

### What It Does

Upcasts LoRA's down→up matrix multiplications to fp32, applies scaling in fp32, then casts the result back to the model's native dtype before adding to the base output.

```python
# Standard path (bf16 throughout)
lx = self.lora_down(x)                    # bf16 matmul
lx = self.lora_up(lx)                     # bf16 matmul
return org_forwarded + lx * multiplier     # bf16 add — delta may vanish

# FP32 accumulation path
lx = F.linear(x.float(), self.lora_down.weight.float())   # fp32 matmul
lx = F.linear(lx, self.lora_up.weight.float())            # fp32 matmul
lx = (lx * multiplier * scale).to(org_forwarded.dtype)    # scale in fp32, then cast
return org_forwarded + lx                                  # add in bf16
```

### Why It Matters

Consider a rank-4 LoRA on a hidden_dim=3072 linear layer. The delta is the product of a (3072×4) and (4×3072) matrix — it's inherently low-magnitude because it passes through a 4-dimensional bottleneck. In bf16, values smaller than ~1e-2 relative to the base activation get rounded away.

FP32 accumulation preserves these deltas with ~7 significant digits, ensuring the LoRA contribution actually reaches the residual stream.

### Cost

| Metric | Impact |
|--------|--------|
| VRAM | Negligible. Temporary fp32 tensors are rank-sized (e.g., 4×3072), not model-sized |
| Speed | Near-zero. The fp32 matmuls are on tiny matrices; base model forward/backward dominates |
| Correctness | Strictly more precise — no behavioral change, just higher fidelity |

### Recommendation

**Always enable.** There is no downside. The precision gain is most impactful for low ranks (≤32) and bf16 training, but even at higher ranks the cost is zero so there's no reason to leave it off.

---

## Feature 2: `attn_softmax_scale`

**Flag:** `--attn_softmax_scale <float>`
**Files:** `library/attention.py` (propagation to all backends), `library/anima_models.py` (model integration), `library/anima_train_utils.py` (arg definition)

### What It Does

Overrides the default attention scale factor (1/√head_dim) with a custom value. This scale is applied to Q·K dot products before softmax, across all supported backends:

| Backend | Parameter |
|---------|-----------|
| PyTorch SDPA | `scale=` |
| xFormers | `scale=` |
| SageAttention | `sm_scale=` |
| Flash Attention | `softmax_scale=` |

### Why It Matters

For head_dim=128, the default scale is 1/√128 ≈ 0.088. This compresses attention logits into a narrow range. In bf16:

- **Small scale (default 0.088):** Logits clustered near zero → softmax outputs near-uniform → attention can't differentiate tokens
- **Larger scale (e.g., 0.12):** Logits spread wider → softmax produces sharper distributions → better token discrimination in bf16

This is motivated by research on low-precision training instability showing that attention softmax is often the first component to degrade under reduced precision.

### Cost

| Metric | Impact |
|--------|--------|
| VRAM | Zero |
| Speed | Zero (single scalar multiply) |
| Correctness | **This changes model behavior.** Larger scale = sharper attention. Too large can cause divergence |

### Caveats

- **Flash Attention already accumulates softmax in fp32 internally.** If you're using `--attn_mode flash`, the precision benefit of a custom scale is reduced (though the sharpness effect remains).
- **This is a hyperparameter, not a free lunch.** Wrong values can hurt convergence. The recommended range is 0.10–0.15 for head_dim=128.
- **Inference mismatch risk:** If you train with a non-default scale, inference must use the same scale for consistent results.

### Recommendation

**Worth experimenting with on non-Flash backends (torch, sageattn) and bf16 training.** Start at 0.10 and compare loss curves against the default. If you're already using Flash Attention, the marginal benefit is small.

---

## Comparison with Existing Training Scripts

### Feature Availability Matrix

| Feature | `lora.py` (SD1/2) | `lora_flux.py` (FLUX) | `lora_anima.py` (Anima) |
|---------|:---:|:---:|:---:|
| FP32 accumulation | - | **Yes** | **Yes** (inherited) |
| Softmax scale | - | - | **Yes** |
| Split QKV dims | - | **Yes** | **Yes** (inherited) |
| GGPO | - | **Yes** | **Yes** (inherited) |
| Rank dropout | Yes | Yes | Yes |
| Module dropout | Yes | Yes | Yes |
| LoRA+ | Yes | Yes | Yes |
| Regex-based LR | - | - | **Yes** |

### Precision Handling by Training Script

| Feature | `train_network.py` | `sdxl_train_network.py` | `flux_train_network.py` | `anima_train_network.py` |
|---------|:---:|:---:|:---:|:---:|
| mixed_precision (fp16/bf16) | Yes | Yes | Yes | Yes |
| full_fp16 / full_bf16 | - | Yes | Yes | Yes |
| FP8 base model | - | - | Yes | - |
| LoRA fp32 accumulation | - | - | - | **Yes** |
| Attention softmax scale | - | - | - | **Yes** |
| Unsloth offload checkpointing | - | - | - | **Yes** |

### What the Standard Scripts Are Missing

The base `train_network.py` and `sdxl_train_network.py` have no precision-aware LoRA computation — the forward pass runs entirely in the model's dtype. This is fine for fp32 training but becomes a silent quality degradation in bf16:

- **No fp32 accumulation:** LoRA deltas computed and added in bf16. At low ranks, contributions are effectively discarded during rounding.
- **No softmax scale control:** Locked to 1/√head_dim. No way to compensate for bf16's limited dynamic range in attention.
- **No GGPO:** No gradient-guided perturbation for robustness. Standard LoRA relies entirely on dropout for regularization.

The `flux_train_network.py` has FP8 support and the `lora_flux.py` module supports fp32 accumulation and GGPO, but the training script doesn't expose `--lora_fp32_accumulation` or `--attn_softmax_scale` as arguments — these are only wired up in the Anima training path.

---

## Practical Guidance

### When to use what

| Scenario | fp32 accumulation | softmax scale | Notes |
|----------|:-:|:-:|-------|
| bf16 + low rank (≤32) | **Yes** | Try 0.10–0.12 | Maximum benefit — small deltas most vulnerable to bf16 rounding |
| bf16 + high rank (≥64) | Yes (free) | Optional | Deltas are larger, less likely to vanish, but still no cost to enable |
| bf16 + Flash Attention | **Yes** | Low priority | Flash already does fp32 softmax internally |
| bf16 + SDPA/sageattn | **Yes** | **Yes** | Both precision fixes complement each other |
| fp32 training | Unnecessary | Unnecessary | Full precision already, no rounding issues |
| fp16 training | **Yes** | Try 0.10–0.12 | fp16 has more mantissa bits than bf16 but still benefits |

### Example command

```bash
python anima_train_network.py \
    --mixed_precision bf16 \
    --full_bf16 \
    --lora_fp32_accumulation \
    --attn_softmax_scale 0.11 \
    --attn_mode sageattn \
    --network_dim 16 \
    ...
```

---

## Summary

| | `lora_fp32_accumulation` | `attn_softmax_scale` |
|---|---|---|
| **Mechanism** | Upcasts LoRA matmuls to fp32 | Widens attention logit range |
| **Type** | Precision fix (transparent) | Hyperparameter (changes behavior) |
| **Cost** | ~0 | 0 |
| **Risk** | None | Can affect convergence if too large |
| **When most useful** | Low-rank LoRA + bf16 | Non-Flash attention + bf16 |
| **Verdict** | Always enable | Experiment carefully |

`lora_fp32_accumulation` is a strict improvement with no tradeoffs — enable it by default. `attn_softmax_scale` is a useful knob for squeezing more precision out of bf16 attention, but it requires tuning and its benefit overlaps with Flash Attention's built-in fp32 softmax.
