"""
DARE-TIES LoRA merge script.

Merges multiple LoRA adapters using the DARE-TIES algorithm:
- DARE (Drop And REscale): Random pruning of task vectors with L1 norm rescaling
- TIES (Trim, Elect Sign, Merge): Sign consensus voting to resolve parameter conflicts

Supports three merge methods:
- dare_ties: DARE random pruning + TIES sign consensus (default)
- dare_linear: DARE random pruning + simple weighted sum (no sign consensus)
- ties: TIES magnitude pruning + sign consensus (no DARE random pruning)

References:
- DARE: "Language Models are Super Mario" (https://arxiv.org/abs/2311.03099)
- TIES: "Resolving Interference When Merging Models" (https://arxiv.org/abs/2306.01708)
"""

import argparse
from collections import defaultdict
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Add project root to path so `library` can be imported when running as a standalone script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from library import train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_state_dict(file_name: str, dtype: torch.dtype, device: torch.device = torch.device("cpu")) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Load a LoRA state dict from file."""
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if isinstance(sd[key], torch.Tensor):
            sd[key] = sd[key].to(dtype=dtype, device=device)

    return sd, metadata


def expand_lora_to_delta(sd: Dict[str, torch.Tensor], module_name: str) -> torch.Tensor:
    """Expand a LoRA module (up @ down * scale) to a full delta tensor.

    For LoRAs, the weights are already task vectors (deltas from base model),
    so we just need to compute: up @ down * (alpha / rank)
    """
    down_key = f"{module_name}.lora_down.weight"
    up_key = f"{module_name}.lora_up.weight"
    alpha_key = f"{module_name}.alpha"

    down = sd[down_key]
    up = sd[up_key]
    rank = down.shape[0]
    alpha = sd.get(alpha_key, torch.tensor(float(rank)))
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    scale = alpha / rank

    # Handle different weight shapes
    if len(down.shape) == 2 and len(up.shape) == 2:
        # Linear: down [rank, in], up [out, rank]
        delta = (up @ down) * scale
    elif len(down.shape) == 4 and down.shape[2:] == (1, 1):
        # Conv2d 1x1: squeeze spatial dims, matmul, unsqueeze back
        delta = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scale
    elif len(down.shape) == 4:
        # Conv2d 3x3: use conv2d
        delta = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3) * scale
    else:
        raise ValueError(f"Unsupported weight shape: down={down.shape}, up={up.shape}")

    return delta


def dare_prune_rescale(tensor: torch.Tensor, density: float, seed: Optional[int] = None) -> torch.Tensor:
    """Apply DARE: random pruning with Bernoulli mask, then L1 norm rescaling.

    Args:
        tensor: Input task vector
        density: Fraction of weights to keep (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Pruned and rescaled tensor
    """
    if density >= 1.0:
        return tensor.clone()
    if density <= 0.0:
        return torch.zeros_like(tensor)

    # Generate Bernoulli mask (generator must be on CPU for deterministic seeding)
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    mask = torch.bernoulli(torch.full(tensor.shape, density, dtype=torch.float32, device="cpu"), generator=generator).to(dtype=tensor.dtype, device=tensor.device)

    masked = tensor * mask

    # L1 rescaling: preserve expected magnitude
    original_l1 = tensor.abs().sum()
    masked_l1 = masked.abs().sum()

    if masked_l1 > 0:
        masked = masked * (original_l1 / masked_l1)

    return masked


def ties_magnitude_prune(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Apply TIES magnitude-based pruning: keep top-k% by absolute value.

    Args:
        tensor: Input task vector
        density: Fraction of weights to keep (0.0 to 1.0)

    Returns:
        Pruned tensor (no rescaling for TIES)
    """
    if density >= 1.0:
        return tensor.clone()
    if density <= 0.0:
        return torch.zeros_like(tensor)

    flat = tensor.abs().flatten()
    k = max(1, int(flat.numel() * density))
    threshold = torch.topk(flat, k).values[-1]
    mask = (tensor.abs() >= threshold).to(tensor.dtype)

    return tensor * mask


def ties_sign_consensus(
    deltas: List[torch.Tensor],
    weights: List[float],
) -> List[torch.Tensor]:
    """Apply TIES sign consensus: compute majority sign and mask disagreeing values.

    For each parameter position, the majority sign is determined by the weighted
    sum of all task vectors. Values whose sign disagrees with the majority are
    masked to zero.

    Args:
        deltas: List of task vector tensors (one per model)
        weights: Weight for each model

    Returns:
        List of masked task vectors
    """
    # Compute weighted sum to determine majority sign
    weighted_sum = torch.zeros_like(deltas[0])
    for delta, w in zip(deltas, weights):
        weighted_sum += delta * w

    # Majority sign: +1 where sum >= 0, -1 where sum < 0
    majority_sign = (weighted_sum >= 0).float() * 2 - 1

    # Mask each delta: zero out values that disagree with majority sign
    result = []
    for delta in deltas:
        delta_sign = delta.sign()
        # Keep values where sign matches majority OR where delta is zero
        agree_mask = (delta_sign == majority_sign) | (delta == 0)
        result.append(delta * agree_mask.float())

    return result


def svd_decompose_delta(
    delta: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose a full delta tensor back into LoRA format using SVD.

    delta ≈ up @ down, where:
        down: [rank, in_dim]
        up: [out_dim, rank]

    The singular values are distributed into both matrices (sqrt split).

    Args:
        delta: Full delta tensor [out_dim, in_dim] or [out_dim, in_dim, h, w]
        rank: Target rank for decomposition

    Returns:
        Tuple of (down, up) weight tensors
    """
    is_conv = len(delta.shape) == 4
    if is_conv:
        out_dim, in_dim, h, w = delta.shape
        # Reshape for SVD: [out_dim, in_dim * h * w]
        delta_2d = delta.reshape(out_dim, -1)
    else:
        delta_2d = delta

    # SVD decomposition
    U, S, Vh = torch.linalg.svd(delta_2d.float(), full_matrices=False)

    # Truncate to target rank
    rank = min(rank, min(delta_2d.shape[0], delta_2d.shape[1]))
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    # Split singular values between up and down (sqrt split)
    sqrt_S = S.sqrt()
    up = U * sqrt_S.unsqueeze(0)     # [out_dim, rank]
    down = sqrt_S.unsqueeze(1) * Vh  # [rank, in_dim] or [rank, in_dim * h * w]

    if is_conv:
        down = down.reshape(rank, in_dim, h, w)

    return down.to(delta.dtype).contiguous(), up.to(delta.dtype).contiguous()


def get_module_names(sd: Dict[str, torch.Tensor]) -> List[str]:
    """Extract unique LoRA module names from a state dict."""
    names = set()
    for key in sd.keys():
        if ".lora_down.weight" in key:
            name = key.replace(".lora_down.weight", "")
            names.add(name)
    return sorted(names)


def dare_ties_merge(
    model_paths: List[str],
    weights: List[float],
    density: float,
    output_rank: int,
    merge_dtype: torch.dtype,
    seed: Optional[int] = None,
    method: str = "dare_ties",
    no_rescale: bool = False,
    device: torch.device = torch.device("cpu"),
    num_shards: int = 1,
) -> Dict[str, torch.Tensor]:
    """Perform DARE-TIES merge on multiple LoRA models.

    Args:
        model_paths: Paths to LoRA safetensors files
        weights: Weight for each model
        density: Fraction of weights to keep (0.0 to 1.0)
        output_rank: Rank for the output LoRA
        merge_dtype: Data type for merge computation
        seed: Random seed for DARE pruning reproducibility
        method: Merge method - "dare_ties", "dare_linear", or "ties"
        no_rescale: If True, skip L1 rescaling in DARE
        num_shards: Number of shards to split modules into for VRAM management

    Returns:
        Merged LoRA state dict
    """
    assert len(model_paths) == len(weights), "Number of models must equal number of weights"
    assert method in ("dare_ties", "dare_linear", "ties"), f"Unknown method: {method}"

    # Load all models
    all_sds = []
    for path in model_paths:
        logger.info(f"Loading: {path}")
        sd, _ = load_state_dict(path, merge_dtype, device=device)
        all_sds.append(sd)

    # Collect all module names across all models
    all_module_names = set()
    for sd in all_sds:
        all_module_names.update(get_module_names(sd))
    all_module_names = sorted(all_module_names)
    logger.info(f"Total unique modules: {len(all_module_names)}")

    base_seed = seed if seed is not None else 0
    merged_sd = {}

    # Process modules in shards to limit VRAM usage
    # Each shard: compute deltas on GPU -> batched SVD -> store results on CPU -> free VRAM
    num_shards = max(1, num_shards)
    shard_size = math.ceil(len(all_module_names) / num_shards)

    for shard_idx in range(num_shards):
        shard_start = shard_idx * shard_size
        shard_end = min(shard_start + shard_size, len(all_module_names))
        shard_modules = all_module_names[shard_start:shard_end]

        if num_shards > 1:
            desc = f"Shard {shard_idx + 1}/{num_shards}: computing deltas"
        else:
            desc = "Computing deltas"

        # Compute merged deltas for this shard
        shard_deltas = []  # list of (module_name, merged_delta)

        for module_idx_in_shard, module_name in enumerate(tqdm(shard_modules, desc=desc)):
            module_idx = shard_start + module_idx_in_shard
            deltas = []
            delta_weights = []
            delta_indices = []

            for model_idx, (sd, w) in enumerate(zip(all_sds, weights)):
                down_key = f"{module_name}.lora_down.weight"
                if down_key not in sd:
                    continue

                delta = expand_lora_to_delta(sd, module_name)
                deltas.append(delta)
                delta_weights.append(w)
                delta_indices.append(model_idx)

            if len(deltas) == 0:
                continue

            # Prune (DARE random or TIES magnitude)
            pruned_deltas = []
            for i, delta in enumerate(deltas):
                if method in ("dare_ties", "dare_linear"):
                    prune_seed = base_seed + module_idx * 1000 + delta_indices[i] if seed is not None else None
                    pruned = dare_prune_rescale(delta, density, seed=prune_seed)
                    if no_rescale:
                        generator = torch.Generator(device="cpu")
                        if prune_seed is not None:
                            generator.manual_seed(prune_seed)
                        mask = torch.bernoulli(
                            torch.full(delta.shape, density, dtype=torch.float32, device="cpu"), generator=generator
                        ).to(dtype=delta.dtype, device=delta.device)
                        pruned = delta * mask
                elif method == "ties":
                    pruned = ties_magnitude_prune(delta, density)
                pruned_deltas.append(pruned)

            # Sign consensus
            if method in ("dare_ties", "ties"):
                pruned_deltas = ties_sign_consensus(pruned_deltas, delta_weights)

            # Weighted sum
            merged_delta = torch.zeros_like(pruned_deltas[0])
            for delta, w in zip(pruned_deltas, delta_weights):
                merged_delta += delta * w

            shard_deltas.append((module_name, merged_delta))

        # Batched SVD for this shard - group by shape
        shape_groups: Dict[tuple, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
        for module_name, delta in shard_deltas:
            if len(delta.shape) == 4:
                svd_key = (delta.shape[0], delta.shape[1] * delta.shape[2] * delta.shape[3], True, delta.shape[1], delta.shape[2], delta.shape[3])
            else:
                svd_key = (delta.shape[0], delta.shape[1], False)
            shape_groups[svd_key].append((module_name, delta))

        if num_shards > 1:
            svd_desc = f"Shard {shard_idx + 1}/{num_shards}: batched SVD"
        else:
            svd_desc = "Batched SVD"

        for svd_key, group in tqdm(shape_groups.items(), desc=svd_desc):
            is_conv = svd_key[2]
            names = [g[0] for g in group]
            deltas = [g[1] for g in group]

            if is_conv:
                in_dim, kh, kw = svd_key[3], svd_key[4], svd_key[5]
                deltas_2d = [d.reshape(d.shape[0], -1) for d in deltas]
            else:
                deltas_2d = deltas

            batched = torch.stack(deltas_2d)

            U, S, Vh = torch.linalg.svd(batched.float(), full_matrices=False)

            r = min(output_rank, min(batched.shape[1], batched.shape[2]))
            U = U[:, :, :r]
            S = S[:, :r]
            Vh = Vh[:, :r, :]

            sqrt_S = S.sqrt()
            up_batch = U * sqrt_S.unsqueeze(1)
            down_batch = sqrt_S.unsqueeze(2) * Vh

            for i, name in enumerate(names):
                down = down_batch[i]
                up = up_batch[i]

                if is_conv:
                    down = down.reshape(r, in_dim, kh, kw)

                # Store on CPU immediately to free VRAM
                merged_sd[f"{name}.lora_down.weight"] = down.to(merge_dtype).cpu().contiguous()
                merged_sd[f"{name}.lora_up.weight"] = up.to(merge_dtype).cpu().contiguous()
                merged_sd[f"{name}.alpha"] = torch.tensor(float(output_rank))

        # Free shard tensors
        del shard_deltas, shape_groups
        if device.type != "cpu":
            torch.cuda.empty_cache()

    logger.info(f"Merge complete: {len(all_module_names)} modules merged")
    return merged_sd


def merge(args):
    assert len(args.models) == len(args.ratios), "Number of models must equal number of ratios"

    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    device = torch.device(args.device)

    merged_sd = dare_ties_merge(
        model_paths=args.models,
        weights=args.ratios,
        density=args.density,
        output_rank=args.output_rank,
        merge_dtype=merge_dtype,
        seed=args.seed,
        method=args.method,
        no_rescale=args.no_rescale,
        device=device,
        num_shards=args.num_shards,
    )

    # Convert to save dtype
    if save_dtype is not None:
        for key in list(merged_sd.keys()):
            if isinstance(merged_sd[key], torch.Tensor):
                merged_sd[key] = merged_sd[key].to(save_dtype)

    # Build metadata
    logger.info("Calculating hashes and creating metadata...")
    metadata = {}
    metadata["ss_network_module"] = "networks.lora"
    metadata["ss_network_dim"] = str(args.output_rank)
    metadata["ss_network_alpha"] = str(float(args.output_rank))
    metadata["dare_ties_method"] = args.method
    metadata["dare_ties_density"] = str(args.density)
    metadata["dare_ties_weights"] = str(args.ratios)
    metadata["dare_ties_models"] = str([os.path.basename(m) for m in args.models])
    if args.seed is not None:
        metadata["dare_ties_seed"] = str(args.seed)

    model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(merged_sd, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    logger.info(f"Saving model to: {args.save_to}")
    save_file(merged_sd, args.save_to, metadata=metadata)
    logger.info("Done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge multiple LoRA models using DARE-TIES algorithm")

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="LoRA model files to merge (safetensors)",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        required=True,
        help="Weight ratio for each model (e.g., 1.0 1.0 0.8)",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        required=True,
        help="Output file path (safetensors)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="Fraction of weights to keep after pruning (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--output_rank",
        type=int,
        default=None,
        help="Rank for the output LoRA (default: use the max rank from input models)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dare_ties",
        choices=["dare_ties", "dare_linear", "ties"],
        help="Merge method: dare_ties (default), dare_linear (no sign consensus), ties (magnitude pruning)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="Precision for merge computation (default: float)",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="Precision for saving (default: same as merge precision)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for DARE pruning reproducibility",
    )
    parser.add_argument(
        "--no_rescale",
        action="store_true",
        help="Skip L1 rescaling in DARE (not recommended)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for merge computation, e.g. 'cuda' or 'cpu' (default: cpu)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split modules into N shards to limit VRAM usage (default: 1, no sharding)",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    # Auto-detect output rank from input models if not specified
    if args.output_rank is None:
        max_rank = 0
        for model_path in args.models:
            sd = load_file(model_path) if model_path.endswith(".safetensors") else torch.load(model_path, map_location="cpu")
            for key in sd:
                if ".lora_down.weight" in key:
                    max_rank = max(max_rank, sd[key].shape[0])
        args.output_rank = max_rank
        logger.info(f"Auto-detected output rank: {max_rank}")

    merge(args)
