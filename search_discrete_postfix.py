#!/usr/bin/env python3
"""Discrete postfix token search for Anima LLM Adapter.

Instead of learning continuous postfix vectors via gradient descent, this script
searches for the best discrete tokens from the Qwen3 and T5 vocabularies using:
  Phase 1: Pre-compute vocab hidden states
  Phase 2: Gradient-based candidate selection (narrow to ~1% of vocab)
  Phase 3: Greedy position-by-position search using training loss

Outputs a .safetensors file compatible with PostfixNetwork dual mode.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm

# Add sd-scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from library import (
    anima_utils,
    flux_train_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Discrete postfix token search for Anima")

    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="DiT model path")
    parser.add_argument("--vae", type=str, required=True, help="VAE model path (for latent caching)")
    parser.add_argument("--qwen3", type=str, required=True, help="Qwen3 text encoder path")
    parser.add_argument("--t5_tokenizer_path", type=str, default=None, help="T5 tokenizer path")

    # Dataset
    parser.add_argument("--dataset_config", type=str, required=True, help="TOML dataset config")

    # Search params
    parser.add_argument("--num_postfix_tokens", type=int, default=16, help="Number of Qwen3-side postfix tokens")
    parser.add_argument("--num_t5_postfix_tokens", type=int, default=16, help="Number of T5-side postfix tokens")
    parser.add_argument("--candidate_ratio", type=float, default=0.01, help="Fraction of vocab to keep as candidates")
    parser.add_argument("--eval_images", type=int, default=8, help="Number of images per candidate evaluation")
    parser.add_argument("--grad_batches", type=int, default=4, help="Number of batches for gradient scoring")

    # Output
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="anima_discrete_postfix")

    # Training-compatible args (for noise sampling)
    parser.add_argument("--timestep_sampling", type=str, default="sigmoid")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0)
    parser.add_argument("--discrete_flow_shift", type=float, default=1.0)
    parser.add_argument("--weighting_scheme", type=str, default=None)
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--ip_noise_gamma", type=float, default=0)
    parser.add_argument("--ip_noise_gamma_random_strength", action="store_true")

    # Hardware
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn_mode", type=str, default="flash")
    parser.add_argument("--split_attn", action="store_true")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Cache location
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to store cached latents/TE outputs (default: cache_dataset/ next to image_dir)")

    # Dataset-compatible args
    parser.add_argument("--qwen3_max_token_length", type=int, default=256)
    parser.add_argument("--t5_max_token_length", type=int, default=256)
    parser.add_argument("--vae_batch_size", type=int, default=1)
    parser.add_argument("--vae_chunk_size", type=int, default=None)
    parser.add_argument("--vae_disable_cache", action="store_true")
    parser.add_argument("--text_encoder_batch_size", type=int, default=None)
    parser.add_argument("--caption_shuffle_variants", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def get_weight_dtype(args):
    if args.mixed_precision == "bf16":
        return torch.bfloat16
    elif args.mixed_precision == "fp16":
        return torch.float16
    return torch.float32


@torch.no_grad()
def precompute_qwen3_hidden_states(qwen3_model, device, dtype, batch_size=512):
    """Run all vocab tokens through Qwen3 to get output hidden states.

    Assumes qwen3_model is already on the target device.
    Returns: (vocab_size, hidden_dim) tensor on CPU.
    """
    vocab_size = qwen3_model.config.vocab_size
    hidden_dim = qwen3_model.config.hidden_size
    logger.info(f"Pre-computing Qwen3 hidden states for {vocab_size} tokens...")

    all_hidden_states = torch.empty(vocab_size, hidden_dim, dtype=dtype)

    for start in tqdm(range(0, vocab_size, batch_size), desc="Qwen3 vocab forward"):
        end = min(start + batch_size, vocab_size)
        input_ids = torch.arange(start, end, device=device).unsqueeze(1)  # (B, 1)
        outputs = qwen3_model(input_ids=input_ids, output_hidden_states=False)
        hidden = outputs.last_hidden_state[:, 0, :].to(dtype)
        all_hidden_states[start:end] = hidden.cpu()

    logger.info(f"Qwen3 hidden states shape: {all_hidden_states.shape}, "
                f"mean norm: {all_hidden_states.norm(dim=-1).mean():.2f}")
    return all_hidden_states


@torch.no_grad()
def get_t5_embeddings(adapter, device, dtype):
    """Extract T5 embedding matrix projected through adapter.in_proj.

    Returns: (vocab_size, model_dim) tensor.
    """
    embed_weight = adapter.embed.weight.to(device=device, dtype=dtype)  # (32128, 1024)
    # Project through in_proj (may be Identity)
    if isinstance(adapter.in_proj, nn.Identity):
        projected = embed_weight
    else:
        projected = adapter.in_proj(embed_weight)
    logger.info(f"T5 embeddings shape: {projected.shape}, "
                f"mean norm: {projected.norm(dim=-1).mean():.2f}")
    return projected.cpu()


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def resolve_cache_dir(image_dir, cache_dir_arg):
    """Resolve cache directory for a given image directory."""
    if cache_dir_arg:
        return Path(cache_dir_arg)
    # Default: cache_dataset/ sibling to image_dir
    return Path(image_dir).parent / "cache_dataset"


def find_uncached_images(image_dirs, cache_dir_arg):
    """Scan image directories and return lists of images needing TE and latent caches."""
    images_need_latents = []
    images_need_te = []

    for image_dir in image_dirs:
        image_dir = Path(image_dir)
        cache_dir = resolve_cache_dir(image_dir, cache_dir_arg)
        if not image_dir.exists():
            continue
        for img_file in sorted(image_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = img_file.stem

            te_path = cache_dir / f"{stem}_anima_te.npz"
            if not te_path.exists():
                caption_file = image_dir / f"{stem}.txt"
                if caption_file.exists():
                    images_need_te.append((img_file, te_path, caption_file))

            latent_files = list(cache_dir.glob(f"{stem}_*_anima.npz"))
            if not latent_files:
                images_need_latents.append((img_file, cache_dir))

    return images_need_te, images_need_latents


@torch.no_grad()
def cache_te_outputs(qwen3_model, qwen3_tokenizer, t5_tokenizer, images_need_te, args, device):
    """Cache TE outputs. Assumes Qwen3 is already on the target device."""
    max_qwen3_len = args.qwen3_max_token_length
    max_t5_len = args.t5_max_token_length

    for _img_file, te_path, caption_file in tqdm(images_need_te, desc="TE cache"):
        te_path.parent.mkdir(parents=True, exist_ok=True)
        caption = caption_file.read_text().strip()

        qwen3_enc = qwen3_tokenizer(
            caption, return_tensors="pt", truncation=True, padding="max_length", max_length=max_qwen3_len
        )
        t5_enc = t5_tokenizer(
            caption, return_tensors="pt", truncation=True, padding="max_length", max_length=max_t5_len
        )

        outputs = qwen3_model(
            input_ids=qwen3_enc["input_ids"].to(device),
            attention_mask=qwen3_enc["attention_mask"].to(device),
        )
        prompt_embeds = outputs.last_hidden_state[0].cpu().float().numpy()

        np.savez(
            te_path,
            prompt_embeds=prompt_embeds,
            attn_mask=qwen3_enc["attention_mask"][0].numpy().astype(np.int32),
            t5_input_ids=t5_enc["input_ids"][0].numpy().astype(np.int32),
            t5_attn_mask=t5_enc["attention_mask"][0].numpy().astype(np.int32),
            caption_dropout_rate=np.array(0.0, dtype=np.float32),
        )

    logger.info(f"  Cached {len(images_need_te)} TE outputs")


def build_bucket_manager(config):
    """Build a BucketManager from the TOML dataset config."""
    from library.train_util import BucketManager

    general = config.get("general", {})
    resolution = general.get("resolution", 1024)
    if isinstance(resolution, list):
        max_reso = tuple(resolution)
    else:
        max_reso = (resolution, resolution)
    min_size = general.get("min_bucket_reso", 256)
    max_size = general.get("max_bucket_reso", 1024)
    reso_steps = general.get("bucket_reso_steps", 64)

    bucket_manager = BucketManager(False, max_reso, min_size, max_size, reso_steps)
    bucket_manager.make_buckets()
    return bucket_manager


@torch.no_grad()
def cache_latents(images_need_latents, args, device, weight_dtype, bucket_manager):
    """Load VAE, cache latents with bucket-based resizing, then free VAE."""
    from library.train_util import IMAGE_TRANSFORMS, trim_and_resize_if_required

    logger.info(f"Caching latents for {len(images_need_latents)} images, loading VAE...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae, device="cpu", disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size, disable_cache=args.vae_disable_cache,
    )
    vae.to(device=device, dtype=weight_dtype)
    vae.eval()

    for img_file, cache_dir in tqdm(images_need_latents, desc="Latent cache"):
        cache_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(img_file).convert("RGB")
        w, h = img.size

        # Use bucket manager to determine target resolution (same as training pipeline)
        bucket_reso, resized_size, _ = bucket_manager.select_bucket(w, h)

        # Resize and crop using the same logic as the training pipeline
        img_array = np.array(img, dtype=np.uint8)
        img_array, original_size, crop_ltrb = trim_and_resize_if_required(
            False, img_array, bucket_reso, resized_size
        )

        # Apply same transforms as training pipeline: ToTensor + Normalize([0.5], [0.5]) → [-1, 1]
        img_tensor = IMAGE_TRANSFORMS(img_array).unsqueeze(0).to(device=device, dtype=weight_dtype)
        latents = vae.encode_pixels_to_latents(img_tensor)

        lat_np = latents[0].cpu().float().numpy()
        stem = img_file.stem
        bw, bh = bucket_reso
        np.savez(
            cache_dir / f"{stem}_{bw:04d}x{bh:04d}_anima.npz",
            latents=lat_np,
            original_size=np.array(original_size),
            crop_ltrb=np.array(crop_ltrb),
        )

    del vae
    torch.cuda.empty_cache()
    logger.info(f"  Cached {len(images_need_latents)} latents")


def load_dataset_and_cache(image_dirs, cache_dir_arg):
    """Load cached latents and text encoder outputs from the cache directory.

    Returns list of (latents, te_conds) tuples where:
      latents: (C, 1, H/8, W/8) tensor
      te_conds: [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
    """
    all_data = []
    for image_dir in image_dirs:
        cache_dir = resolve_cache_dir(image_dir, cache_dir_arg)
        if not cache_dir.exists():
            logger.warning(f"Cache directory not found: {cache_dir}")
            continue

        te_cache_files = sorted(cache_dir.glob("*_anima_te.npz"))
        if not te_cache_files:
            logger.warning(f"No TE cache files in {cache_dir}, skipping")
            continue

        for te_file in te_cache_files:
            stem = te_file.name.replace("_anima_te.npz", "")

            latent_files = list(cache_dir.glob(f"{stem}_*_anima.npz"))
            if not latent_files:
                logger.warning(f"No latent cache found for {stem}, skipping")
                continue

            latent_file = latent_files[0]

            try:
                lat_data = np.load(latent_file)
                latents = torch.from_numpy(lat_data["latents"])

                te_data = np.load(te_file)

                if "num_variants" in te_data:
                    prompt_embeds = torch.from_numpy(te_data["prompt_embeds_v0"])
                    attn_mask = torch.from_numpy(te_data["attn_mask_v0"])
                    t5_input_ids = torch.from_numpy(te_data["t5_input_ids_v0"])
                    t5_attn_mask = torch.from_numpy(te_data["t5_attn_mask_v0"])
                else:
                    prompt_embeds = torch.from_numpy(te_data["prompt_embeds"])
                    attn_mask = torch.from_numpy(te_data["attn_mask"])
                    t5_input_ids = torch.from_numpy(te_data["t5_input_ids"])
                    t5_attn_mask = torch.from_numpy(te_data["t5_attn_mask"])

                all_data.append((latents, [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]))

            except Exception as e:
                logger.warning(f"Failed to load cache for {stem}: {e}")
                continue

    logger.info(f"Loaded {len(all_data)} cached examples")
    return all_data


def compute_loss_batch(
    anima_model,
    latents,
    text_encoder_conds,
    noise_scheduler,
    args,
    device,
    weight_dtype,
    fixed_noise=None,
):
    """Compute rectified flow MSE loss for a batch.

    Returns scalar loss value.
    """
    latents = latents.to(device=device, dtype=weight_dtype)

    if latents.ndim == 5:
        latents = latents.squeeze(2)

    noise = fixed_noise if fixed_noise is not None else torch.randn_like(latents)

    noisy_model_input, timesteps, _sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
        args, noise_scheduler, latents, noise, device, weight_dtype
    )
    timesteps = timesteps / 1000.0

    # Unpack text encoder conditions: [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds[:4]

    prompt_embeds = prompt_embeds.to(device=device, dtype=weight_dtype)
    attn_mask = attn_mask.to(device=device)
    t5_input_ids = t5_input_ids.to(device=device, dtype=torch.long)
    t5_attn_mask = t5_attn_mask.to(device=device)

    bs = latents.shape[0]
    h_latent = latents.shape[-2]
    w_latent = latents.shape[-1]
    padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=weight_dtype, device=device)

    noisy_model_input = noisy_model_input.unsqueeze(2)  # 4D to 5D

    with torch.autocast(device_type="cuda", dtype=weight_dtype):
        model_pred = anima_model(
            noisy_model_input,
            timesteps,
            prompt_embeds,
            padding_mask=padding_mask,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )

    model_pred = model_pred.squeeze(2)
    target = noise - latents

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss


def prepare_eval_batch(all_data, num_images, weight_dtype, seed=42):
    """Prepare a fixed evaluation batch from cached data.

    Handles variable-resolution latents by grouping by shape and picking
    the largest group (most common resolution).

    Returns: (latents, text_encoder_conds, fixed_noise)
    """
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(all_data), generator=rng).tolist()

    # Group by latent shape to find compatible batches
    shape_groups = defaultdict(list)
    for idx in indices:
        lat, te = all_data[idx]
        shape_groups[lat.shape].append(idx)

    # Pick the largest group (most common resolution)
    best_shape = max(shape_groups, key=lambda s: len(shape_groups[s]))
    group_indices = shape_groups[best_shape][:num_images]

    if len(group_indices) < num_images:
        logger.warning(f"Only {len(group_indices)} images at resolution {best_shape}, "
                       f"requested {num_images}")

    # Collect and stack
    latents_list = []
    te_conds_list = []

    for idx in group_indices:
        lat, te = all_data[idx]
        latents_list.append(lat)
        te_conds_list.append(te)

    # Stack latents
    latents = torch.stack(latents_list)  # (B, C, 1, H, W)

    # Stack text encoder conds: [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
    stacked_conds = []
    for c in range(4):
        tensors = [te[c] for te in te_conds_list]
        stacked_conds.append(torch.stack(tensors))

    # Pre-generate fixed noise for consistent evaluation
    if latents.ndim == 5:
        noise_shape = latents.squeeze(2).shape
    else:
        noise_shape = latents.shape
    fixed_noise = torch.randn(noise_shape, generator=rng, dtype=weight_dtype)

    return latents, stacked_conds, fixed_noise


def score_candidates_by_gradient(
    anima_model,
    all_data,
    qwen3_hidden_states,
    t5_embeddings,
    noise_scheduler,
    args,
    device,
    weight_dtype,
    num_postfix_tokens,
    num_t5_postfix_tokens,
    num_batches=4,
    eval_images=8,
):
    """Gradient projection scoring to select top-K candidates from each vocab.

    Creates placeholder postfix vectors, computes loss gradients, then scores
    each vocab token by dot product with negative gradient.

    Returns: (qwen3_candidate_ids, t5_candidate_ids)
    """
    logger.info("Phase 2: Gradient-based candidate selection...")
    embed_dim = qwen3_hidden_states.shape[1]

    # Create placeholder postfix vectors (requires grad)
    placeholder_qwen3 = torch.zeros(num_postfix_tokens, embed_dim, device=device, dtype=torch.float32, requires_grad=True)
    placeholder_t5 = torch.zeros(num_t5_postfix_tokens, embed_dim, device=device, dtype=torch.float32, requires_grad=True)

    # Attach to adapter
    adapter = anima_model.llm_adapter
    adapter.postfix_embeds = placeholder_qwen3
    adapter.postfix_t5_embeds = placeholder_t5

    # Accumulate gradients over multiple batches, micro-batching to save VRAM
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(total=num_batches * eval_images, desc="Gradient scoring")
    for batch_idx in range(num_batches):
        latents, te_conds, fixed_noise = prepare_eval_batch(
            all_data, eval_images, weight_dtype, seed=args.seed + batch_idx
        )

        bs = latents.shape[0]
        for i in range(bs):
            lat_i = latents[i:i+1]
            te_i = [c[i:i+1] for c in te_conds]
            fn_i = fixed_noise[i:i+1]

            loss = compute_loss_batch(
                anima_model, lat_i, te_i, noise_scheduler, args, device, weight_dtype,
                fixed_noise=fn_i.to(device),
            )
            loss.backward()
            total_loss += loss.item()
            total_samples += 1
            pbar.set_postfix(loss=f"{total_loss / total_samples:.6f}")
            pbar.update(1)
    pbar.close()

    avg_loss = total_loss / total_samples
    logger.info(f"  Average loss with zero placeholder: {avg_loss:.6f}")

    # Get gradients (accumulated over batches)
    grad_qwen3 = placeholder_qwen3.grad.float().cpu()  # (num_postfix, embed_dim)
    grad_t5 = placeholder_t5.grad.float().cpu()  # (num_t5_postfix, embed_dim)

    # Score each vocab token: score = -grad . hidden_state
    # Average gradient across postfix positions for initial screening
    avg_grad_qwen3 = grad_qwen3.mean(dim=0)  # (embed_dim,)
    avg_grad_t5 = grad_t5.mean(dim=0)  # (embed_dim,)

    qwen3_scores = -torch.mv(qwen3_hidden_states.float(), avg_grad_qwen3)  # (vocab_size,)
    t5_scores = -torch.mv(t5_embeddings.float(), avg_grad_t5)  # (vocab_size,)

    # Select top candidates
    num_qwen3_candidates = max(1, int(qwen3_hidden_states.shape[0] * args.candidate_ratio))
    num_t5_candidates = max(1, int(t5_embeddings.shape[0] * args.candidate_ratio))

    qwen3_candidate_ids = qwen3_scores.topk(num_qwen3_candidates).indices
    t5_candidate_ids = t5_scores.topk(num_t5_candidates).indices

    logger.info(f"  Selected {num_qwen3_candidates} Qwen3 candidates (top scores: "
                f"{qwen3_scores[qwen3_candidate_ids[:5]].tolist()})")
    logger.info(f"  Selected {num_t5_candidates} T5 candidates (top scores: "
                f"{t5_scores[t5_candidate_ids[:5]].tolist()})")

    # Clean up
    adapter.postfix_embeds = None
    adapter.postfix_t5_embeds = None
    placeholder_qwen3.grad = None
    placeholder_t5.grad = None

    # Also return per-position gradients for position-specific scoring
    return qwen3_candidate_ids, t5_candidate_ids, grad_qwen3, grad_t5


def greedy_search_side(
    anima_model,
    all_data,
    candidate_ids,
    candidate_hidden_states,
    noise_scheduler,
    args,
    device,
    weight_dtype,
    num_tokens,
    side="qwen3",
    other_side_postfix=None,
):
    """Greedy position-by-position search for one side (Qwen3 or T5).

    Returns: (selected_ids, selected_hidden_states, per_position_losses)
    """
    embed_dim = candidate_hidden_states.shape[1]
    num_candidates = len(candidate_ids)

    adapter = anima_model.llm_adapter

    # Initialize: selected tokens so far
    selected_ids = []
    selected_hidden = []
    per_position_losses = []

    # Current postfix state (fill unselected positions with zeros)
    current_postfix = torch.zeros(num_tokens, embed_dim, device=device, dtype=weight_dtype)

    # Prepare fixed eval data (same across all candidates for consistency)
    latents, te_conds, fixed_noise = prepare_eval_batch(
        all_data, args.eval_images, weight_dtype, seed=args.seed + 1000
    )
    fixed_noise = fixed_noise.to(device)

    for pos in range(num_tokens):
        logger.info(f"  Searching {side} position {pos}/{num_tokens} over {num_candidates} candidates...")
        best_loss = float("inf")
        best_id = -1
        best_hidden = None

        # Evaluate each candidate at this position
        for cand_idx in tqdm(range(num_candidates), desc=f"{side} pos {pos}", leave=False):
            token_id = candidate_ids[cand_idx].item()
            hidden = candidate_hidden_states[cand_idx].to(device=device, dtype=weight_dtype)

            # Set current position to candidate
            current_postfix[pos] = hidden

            # Attach postfix
            if side == "qwen3":
                adapter.postfix_embeds = current_postfix
                adapter.postfix_t5_embeds = other_side_postfix
            else:
                adapter.postfix_embeds = other_side_postfix
                adapter.postfix_t5_embeds = current_postfix

            with torch.no_grad():
                # Micro-batch to save VRAM
                mb_loss = 0.0
                mb_bs = latents.shape[0]
                for mi in range(mb_bs):
                    l = compute_loss_batch(
                        anima_model, latents[mi:mi+1], [c[mi:mi+1] for c in te_conds],
                        noise_scheduler, args, device, weight_dtype,
                        fixed_noise=fixed_noise[mi:mi+1],
                    )
                    mb_loss += l.item()

            loss_val = mb_loss / mb_bs
            if loss_val < best_loss:
                best_loss = loss_val
                best_id = token_id
                best_hidden = hidden.clone()

        # Fix this position
        selected_ids.append(best_id)
        selected_hidden.append(best_hidden.cpu())
        current_postfix[pos] = best_hidden
        per_position_losses.append(best_loss)

        logger.info(f"  Position {pos}: token_id={best_id}, loss={best_loss:.6f}")

    # Clean up
    adapter.postfix_embeds = None
    adapter.postfix_t5_embeds = None

    return selected_ids, torch.stack(selected_hidden), per_position_losses


def save_results(
    output_path,
    qwen3_hidden_states,
    t5_hidden_states,
    qwen3_token_ids,
    t5_token_ids,
    qwen3_losses,
    t5_losses,
    qwen3_tokenizer,
    t5_tokenizer,
):
    """Save dual-mode .safetensors and JSON log."""
    # Save safetensors
    state_dict = {
        "postfix": qwen3_hidden_states.to(torch.bfloat16),
        "postfix_t5": t5_hidden_states.to(torch.bfloat16),
    }

    metadata = {
        "ss_network_module": "networks.postfix_anima",
        "ss_num_postfix_tokens": str(qwen3_hidden_states.shape[0]),
        "ss_embed_dim": str(qwen3_hidden_states.shape[1]),
        "ss_mode": "dual",
        "ss_search_method": "discrete_greedy",
    }

    from library import train_util as tu
    model_hash, legacy_hash = tu.precalculate_safetensors_hashes(state_dict, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    safetensors_path = output_path + ".safetensors"
    save_file(state_dict, safetensors_path, metadata)
    logger.info(f"Saved postfix weights to {safetensors_path}")

    # Decode token IDs to text
    qwen3_tokens_text = [qwen3_tokenizer.decode([tid]) for tid in qwen3_token_ids]
    t5_tokens_text = [t5_tokenizer.decode([tid]) for tid in t5_token_ids]

    # Save JSON log
    log = {
        "qwen3_side": {
            "token_ids": qwen3_token_ids,
            "token_text": qwen3_tokens_text,
            "per_position_loss": qwen3_losses,
        },
        "t5_side": {
            "token_ids": t5_token_ids,
            "token_text": t5_tokens_text,
            "per_position_loss": t5_losses,
        },
    }

    json_path = output_path + ".json"
    with open(json_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved search log to {json_path}")

    # Print summary
    logger.info("=== Search Results ===")
    logger.info(f"Qwen3 tokens: {qwen3_tokens_text}")
    logger.info(f"T5 tokens: {t5_tokens_text}")


def main():
    args = parse_args()
    weight_dtype = get_weight_dtype(args)
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # Parse image dirs from TOML config (used for caching and loading)
    config = toml.load(args.dataset_config)
    image_dirs = []
    for section in config.get("datasets", []):
        for subset in section.get("subsets", []):
            if "image_dir" in subset:
                image_dirs.append(subset["image_dir"])
    if not image_dirs:
        raise RuntimeError(f"No image_dir found in {args.dataset_config}")

    # Build bucket manager for resolution-consistent latent caching
    bucket_manager = build_bucket_manager(config)
    logger.info(f"Bucket resolutions: {len(bucket_manager.resos)} buckets, "
                f"reso_steps={bucket_manager.reso_steps}")

    # Check what caches are needed before loading any models
    images_need_te, images_need_latents = find_uncached_images(image_dirs, args.cache_dir)
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)

    # ---- Phase 1: Qwen3 on GPU (vocab hidden states + TE caches) ----
    logger.info("=" * 60)
    logger.info("Phase 1: Qwen3 → vocab hidden states + TE caches")
    logger.info("=" * 60)

    qwen3_model, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        args.qwen3, dtype=weight_dtype, device="cpu"
    )
    # Move to GPU once, do both tasks, then free
    qwen3_model.to(device)

    # 1a. Pre-compute vocab hidden states (Qwen3 already on GPU)
    qwen3_hidden_states = precompute_qwen3_hidden_states(qwen3_model, device, weight_dtype)

    # 1b. Cache TE outputs for uncached images (Qwen3 still on GPU)
    if images_need_te:
        logger.info(f"Caching TE outputs for {len(images_need_te)} images...")
        cache_te_outputs(qwen3_model, qwen3_tokenizer, t5_tokenizer, images_need_te, args, device)
    else:
        logger.info("All TE output caches exist, skipping")

    # Free Qwen3 — done with text encoder
    del qwen3_model
    torch.cuda.empty_cache()

    # ---- Latent caches (VAE on GPU, then freed) ----
    if images_need_latents:
        cache_latents(images_need_latents, args, device, weight_dtype, bucket_manager)
    else:
        logger.info("All latent caches exist, skipping")

    # ---- Load DiT (with integrated LLM adapter) ----
    logger.info("Loading Anima DiT model...")
    loading_device = "cpu" if args.blocks_to_swap > 0 else device
    anima_model = anima_utils.load_anima_model(
        device,
        args.pretrained_model_name_or_path,
        args.attn_mode,
        args.split_attn,
        loading_device,
        weight_dtype,
        fp8_scaled=False,
    )
    if args.blocks_to_swap > 0:
        anima_model.enable_block_swap(args.blocks_to_swap, device)
    # Freeze all model parameters — only postfix placeholders need gradients.
    # Without this, backward() allocates ~4GB for model gradients alone.
    for p in anima_model.parameters():
        p.requires_grad_(False)
    if args.gradient_checkpointing:
        anima_model.enable_gradient_checkpointing()
    # Keep model in train mode so gradient checkpointing is active
    # (DiT blocks skip checkpointing when self.training is False)

    # Get T5 embeddings from adapter
    t5_embeddings = get_t5_embeddings(anima_model.llm_adapter, device, weight_dtype)

    # ---- Load dataset ----
    logger.info("Loading cached dataset...")
    all_data = load_dataset_and_cache(image_dirs, args.cache_dir)
    if len(all_data) == 0:
        raise RuntimeError("No cached data found. Check that image_dir has images with .txt captions.")

    # ---- Phase 2: Gradient-based candidate selection ----
    logger.info("=" * 60)
    logger.info("Phase 2: Gradient-based candidate selection")
    logger.info("=" * 60)

    noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=args.discrete_flow_shift
    )

    qwen3_candidate_ids, t5_candidate_ids, _, _ = score_candidates_by_gradient(
        anima_model,
        all_data,
        qwen3_hidden_states,
        t5_embeddings,
        noise_scheduler,
        args,
        device,
        weight_dtype,
        args.num_postfix_tokens,
        args.num_t5_postfix_tokens,
        num_batches=args.grad_batches,
        eval_images=args.eval_images,
    )

    # Gather candidate hidden states
    qwen3_candidate_hidden = qwen3_hidden_states[qwen3_candidate_ids]  # (num_candidates, embed_dim)
    t5_candidate_hidden = t5_embeddings[t5_candidate_ids]  # (num_candidates, embed_dim)

    # Free full vocab tensors
    del qwen3_hidden_states, t5_embeddings
    torch.cuda.empty_cache()

    # Log candidate tokens for debugging
    qwen3_cand_text = [qwen3_tokenizer.decode([qwen3_candidate_ids[i].item()]) for i in range(min(20, len(qwen3_candidate_ids)))]
    t5_cand_text = [t5_tokenizer.decode([t5_candidate_ids[i].item()]) for i in range(min(20, len(t5_candidate_ids)))]
    logger.info(f"Top Qwen3 candidates: {qwen3_cand_text}")
    logger.info(f"Top T5 candidates: {t5_cand_text}")

    # ---- Phase 3: Greedy search ----
    logger.info("=" * 60)
    logger.info("Phase 3: Greedy search")
    logger.info("=" * 60)

    # Search Qwen3 side first
    logger.info("--- Searching Qwen3 side (KV) ---")
    qwen3_selected_ids, qwen3_selected_hidden, qwen3_losses = greedy_search_side(
        anima_model,
        all_data,
        qwen3_candidate_ids,
        qwen3_candidate_hidden,
        noise_scheduler,
        args,
        device,
        weight_dtype,
        args.num_postfix_tokens,
        side="qwen3",
        other_side_postfix=None,
    )

    # Search T5 side with Qwen3 postfix fixed
    logger.info("--- Searching T5 side (Q) ---")
    qwen3_postfix_fixed = qwen3_selected_hidden.to(device=device, dtype=weight_dtype)
    t5_selected_ids, t5_selected_hidden, t5_losses = greedy_search_side(
        anima_model,
        all_data,
        t5_candidate_ids,
        t5_candidate_hidden,
        noise_scheduler,
        args,
        device,
        weight_dtype,
        args.num_t5_postfix_tokens,
        side="t5",
        other_side_postfix=qwen3_postfix_fixed,
    )

    # ---- Phase 4: Save results ----
    logger.info("=" * 60)
    logger.info("Phase 4: Saving results")
    logger.info("=" * 60)

    output_path = os.path.join(args.output_dir, args.output_name)
    save_results(
        output_path,
        qwen3_selected_hidden,
        t5_selected_hidden,
        qwen3_selected_ids,
        t5_selected_ids,
        qwen3_losses,
        t5_losses,
        qwen3_tokenizer,
        t5_tokenizer,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
