"""
NanoSaur full fine-tuning script.

Trains all (or selected) parameters of the NanoSaur diffusion transformer using
rectified flow matching.  Only the diffusion model weights are saved.

Usage:
    accelerate launch --mixed_precision bf16 nanosaur_train.py \\
      --pretrained_model_name_or_path nanosaur_diffusion_model.safetensors \\
      --text_encoder nanosaur_text_encoder.safetensors \\
      --vae nanosaur_vae_decoder.safetensors \\
      --dataset_config dataset.toml \\
      --output_dir output/ \\
      --output_name my_nanosaur \\
      --max_train_steps 10000 \\
      --learning_rate 5e-6 \\
      --mixed_precision bf16 \\
      --blocks_to_swap 10 \\
      --gradient_checkpointing

Per-attribute learning rates (optional):
    --attr_lr "blocks.0-5=1e-5,blocks.6-25=5e-6,dec_net=2e-5"

Full argument reference: docs/nanosaur_train.md
"""

import argparse
import math
import os
import re
from multiprocessing import Value

import torch
from tqdm import tqdm
from accelerate.utils import set_seed

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    config_util,
    deepspeed_utils,
    nanosaur_train_util,
    nanosaur_utils,
    sai_model_spec,
    strategy_base,
    strategy_nanosaur,
    train_util,
)
from library.config_util import BlueprintGenerator, ConfigSanitizer
from library.custom_train_functions import add_custom_train_arguments, apply_masked_loss
from library.utils import add_logging_arguments, setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Per-attribute learning rate parsing


def parse_attr_lr(attr_lr_str: str) -> dict:
    """
    Parse a string like ``"blocks.0-5=1e-5,dec_net=2e-5"`` into a list of
    (regex, lr) pairs.  Ranges like ``0-5`` are expanded to
    ``(0|1|2|3|4|5)``.

    Returns:
        dict mapping regex pattern string to float lr.
    """
    if not attr_lr_str:
        return {}

    result = {}
    for token in attr_lr_str.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        attr_part, lr_str = token.rsplit("=", 1)
        attr_part = attr_part.strip()
        lr = float(lr_str.strip())

        # Expand range notation like "blocks.0-5" → regex matching blocks.{0..5}
        def expand_ranges(s):
            def replace_range(m):
                lo, hi = int(m.group(1)), int(m.group(2))
                return "(" + "|".join(str(i) for i in range(lo, hi + 1)) + ")"

            return re.sub(r"(\d+)-(\d+)", replace_range, s)

        result[expand_ranges(attr_part)] = lr

    return result


def build_param_groups(model: torch.nn.Module, attr_lr_map: dict, default_lr: float) -> list:
    """
    Build optimizer parameter groups with per-attribute learning rates.

    For each named parameter, the FIRST matching pattern in ``attr_lr_map``
    determines the learning rate.  Unmatched parameters use ``default_lr``.
    """
    if not attr_lr_map:
        return [{"params": list(model.parameters()), "lr": default_lr}]

    # Compile all patterns
    compiled = [(re.compile(pat), lr) for pat, lr in attr_lr_map.items()]

    groups: dict = {}  # lr → list of params
    for name, param in model.named_parameters():
        assigned_lr = default_lr
        for pattern, lr in compiled:
            if pattern.search(name):
                assigned_lr = lr
                break
        groups.setdefault(assigned_lr, []).append(param)

    param_groups = [{"params": params, "lr": lr} for lr, params in groups.items()]
    logger.info(f"Per-attribute LR groups: {[g['lr'] for g in param_groups]}")
    return param_groups


def _compute_uncond_hidden(args, accelerator, tokenize_strategy, text_encoding_strategy, gemma3):
    """Encode the empty prompt once for classifier-free guidance dropout.

    Returns a CPU tensor (1, L, D), or None when dropout is disabled. Must be
    called while ``gemma3`` is on the compute device.
    """
    if float(getattr(args, "cond_dropout_rate", 0.0)) <= 0.0:
        return None
    with accelerator.autocast(), torch.no_grad():
        tokens = tokenize_strategy.tokenize("")
        uncond = text_encoding_strategy.encode_tokens(tokenize_strategy, [gemma3], tokens)
    logger.info(f"NanoSaur: cached unconditional embedding for CFG dropout (rate={args.cond_dropout_rate})")
    return uncond[0].detach().to("cpu")


# Main training function


def train(args: argparse.Namespace) -> None:
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    if not args.skip_cache_check:
        args.skip_cache_check = getattr(args, "skip_latents_validity_check", False)

    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning("Enabling cache_text_encoder_outputs because cache_text_encoder_outputs_to_disk is set.")
        args.cache_text_encoder_outputs = True

    if getattr(args, "cpu_offload_checkpointing", False) and not args.gradient_checkpointing:
        args.gradient_checkpointing = True

    if args.seed is not None:
        set_seed(args.seed)

    cache_latents = args.cache_latents

    # Dataset preparation

    if args.cache_latents:
        latents_caching_strategy = strategy_nanosaur.NanoSaurLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(
            ConfigSanitizer(True, True, getattr(args, "masked_loss", False), True)
        )
        if args.dataset_config is not None:
            user_config = config_util.load_user_config(args.dataset_config)
        else:
            use_dreambooth = args.in_json is None
            if use_dreambooth:
                user_config = {
                    "datasets": [{
                        "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                            args.train_data_dir, args.reg_data_dir
                        )
                    }]
                }
            else:
                user_config = {
                    "datasets": [{
                        "subsets": [{"image_dir": args.train_data_dir, "metadata_file": args.in_json}]
                    }]
                }
        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = (
            config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        )
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None  # noqa: F841

    train_dataset_group.verify_bucket_reso_steps(16)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return

    if len(train_dataset_group) == 0:
        logger.error("No data found. Please check your dataset config.")
        return

    # Accelerator

    logger.info("Preparing accelerator")
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # Load VAE for latent caching

    vae = None
    if cache_latents:
        vae = nanosaur_utils.load_nanosaur_vae(args.vae, weight_dtype, "cpu")
        vae.vae.to(accelerator.device, dtype=weight_dtype)
        vae.vae.requires_grad_(False)
        vae.vae.eval()
        train_dataset_group.new_cache_latents(vae, accelerator)
        vae.vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    # Tokenize strategy

    # Load text encoder (need tokenizer)
    logger.info("Loading NanoSaur text encoder for caching")
    gemma3_tokenizer, gemma3 = nanosaur_utils.load_nanosaur_text_encoder(
        args.text_encoder, weight_dtype, "cpu"
    )
    gemma3.eval()
    gemma3.requires_grad_(False)

    tokenize_strategy = strategy_nanosaur.NanoSaurTokenizeStrategy(
        tokenizer=gemma3_tokenizer,
        max_length=nanosaur_utils.TEXT_MAX_LENGTH,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

    text_encoding_strategy = strategy_nanosaur.NanoSaurTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # Cache text encoder outputs

    sample_prompts_te_outputs = None
    if args.cache_text_encoder_outputs:
        gemma3.to(accelerator.device)
        te_caching_strategy = strategy_nanosaur.NanoSaurTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            False,
            False,
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(te_caching_strategy)

        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([gemma3], accelerator)

        if args.sample_prompts is not None:
            logger.info(f"Caching text encoder outputs for sample prompts: {args.sample_prompts}")
            prompts = train_util.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for key in ("prompt", "negative_prompt"):
                        p = prompt_dict.get(key, "")
                        if p not in sample_prompts_te_outputs:
                            tokens = tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy, [gemma3], tokens
                            )

        # Pre-compute empty-prompt embedding for CFG dropout before freeing the TE.
        uncond_hidden = _compute_uncond_hidden(
            args, accelerator, tokenize_strategy, text_encoding_strategy, gemma3
        )

        accelerator.wait_for_everyone()
        gemma3 = None
        clean_memory_on_device(accelerator.device)
    else:
        # Online encoding: keep Gemma3 on device throughout training
        logger.info("Online text encoding enabled — Gemma3 will encode each batch on-the-fly")
        gemma3.to(accelerator.device, dtype=weight_dtype)
        uncond_hidden = _compute_uncond_hidden(
            args, accelerator, tokenize_strategy, text_encoding_strategy, gemma3
        )

    # Wire the configured strategies into the dataset (tokenize / latents / TE-cache)
    # so dataset __getitem__ can tokenize captions (online) or load cached TE outputs.
    train_dataset_group.set_current_strategies()

    # Load diffusion model

    logger.info("Loading NanoSaur diffusion model")
    model = nanosaur_utils.load_nanosaur_model(
        args.pretrained_model_name_or_path,
        weight_dtype,
        torch.device("cpu"),
        disable_mmap=args.disable_mmap_load_safetensors,
        use_flash_attn=getattr(args, "use_flash_attn", False),
        use_sage_attn=getattr(args, "use_sage_attn", False),
    )

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    model.requires_grad_(True)

    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        logger.info(f"NanoSaur: enabling block swap for {args.blocks_to_swap} blocks")
        model.enable_block_swap(args.blocks_to_swap, accelerator.device)

    # Load VAE for decoding during sampling

    if not cache_latents:
        vae = nanosaur_utils.load_nanosaur_vae(args.vae, weight_dtype, "cpu")
        vae.vae.requires_grad_(False)
        vae.vae.eval()

    # Optimizer

    # Parse per-attribute LR if provided
    attr_lr_str = getattr(args, "attr_lr", None)
    attr_lr_map = parse_attr_lr(attr_lr_str) if attr_lr_str else {}
    params_to_optimize = build_param_groups(model, attr_lr_map, args.learning_rate)

    _, _, optimizer = train_util.get_optimizer(args, params_to_optimize)

    # DataLoader

    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=n_workers > 0,
    )

    # LR scheduler

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if args.max_train_epochs is not None:
        num_train_epochs = args.max_train_epochs
        args.max_train_steps = num_update_steps_per_epoch * num_train_epochs

    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, num_train_epochs * len(train_dataloader))

    # Prepare with accelerator

    if is_swapping_blocks:
        model = accelerator.prepare(model, device_placement=[False])
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()
    else:
        model = accelerator.prepare(model)

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    if not cache_latents:
        vae.vae.to(accelerator.device)

    # Training loop

    train_util.save_sd_model_on_train_end  # just to ensure import

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"NanoSaur full FT: {num_trainable_params:,} trainable parameters")
    logger.info(f"  Total steps: {args.max_train_steps}, epochs: {num_train_epochs}")

    global_step = 0
    first_epoch = 0

    # Resume from checkpoint if requested
    if args.resume is not None and os.path.isdir(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        accelerator.load_state(args.resume)
        global_step = int(args.resume.split("-")[-1])
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(first_epoch * num_update_steps_per_epoch, args.max_train_steps),
        desc="steps",
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_postfix({"loss": "N/A", "lr": "N/A"})

    for epoch in range(first_epoch, num_train_epochs):
        current_epoch.value = epoch
        model.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            with accelerator.accumulate(model):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    imgs = batch["images"].to(accelerator.device, dtype=weight_dtype)
                    with torch.no_grad():
                        latents = vae.encode(imgs)

                # Get text encoder conditioning
                if args.cache_text_encoder_outputs:
                    # Cached TE outputs are exposed by the dataset under
                    # "text_encoder_outputs_list" = [hidden_states, input_ids, attn_mask].
                    te_outputs = batch.get("text_encoder_outputs_list", None)
                    if te_outputs is not None and te_outputs[0] is not None:
                        hidden_states = te_outputs[0].to(accelerator.device, dtype=weight_dtype)
                    else:
                        # This should not happen if caching worked
                        logger.error("No cached text encoder outputs found in batch!")
                        continue
                else:
                    # Online text encoding: tokenize and encode captions per step
                    captions = batch["captions"]
                    with torch.no_grad():
                        tokens = tokenize_strategy.tokenize(captions)
                        te_outputs = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [gemma3], tokens
                        )
                    hidden_states = te_outputs[0].to(weight_dtype)

                # Classifier-free guidance dropout: swap conditioning for the
                # empty-prompt embedding on a fraction of samples (matches reference).
                cond_dropout_rate = float(getattr(args, "cond_dropout_rate", 0.0))
                if cond_dropout_rate > 0.0 and uncond_hidden is not None:
                    drop_mask = torch.rand(hidden_states.size(0), device=hidden_states.device) < cond_dropout_rate
                    if drop_mask.any():
                        uncond = uncond_hidden.to(device=hidden_states.device, dtype=hidden_states.dtype)
                        uncond = uncond.expand(hidden_states.size(0), -1, -1)
                        hidden_states = torch.where(drop_mask.view(-1, 1, 1), uncond, hidden_states)

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                noisy_input, t = nanosaur_train_util.get_noisy_model_input_and_timesteps(
                    args, latents, noise, accelerator.device, weight_dtype
                )

                # Forward pass
                if is_swapping_blocks:
                    accelerator.unwrap_model(model).prepare_block_swap_before_forward()

                with accelerator.autocast():
                    x0_pred = accelerator.unwrap_model(model)._forward(
                        noisy_input, t, hidden_states
                    )

                # Compute rectified flow velocity loss
                velocity_pred, velocity_target = nanosaur_train_util.get_flow_matching_loss(
                    x0_pred, latents, noisy_input, t
                )

                loss = torch.nn.functional.mse_loss(
                    velocity_pred.float(), velocity_target.float(), reduction="none"
                )
                # Masked loss support (needs per-pixel loss, not a scalar)
                if getattr(args, "masked_loss", False) or (
                    "alpha_masks" in batch and batch["alpha_masks"] is not None
                ):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean(dim=list(range(1, loss.ndim)))  # per-sample mean
                # Per-sample weighting (e.g. prior_loss_weight for regularization images)
                if "loss_weights" in batch and batch["loss_weights"] is not None:
                    loss = loss * batch["loss_weights"].to(loss.device)
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                current_step.value = global_step
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })

                # Sample generation
                if args.sample_prompts and accelerator.is_main_process:
                    if vae is None and args.vae:
                        _vae = nanosaur_utils.load_nanosaur_vae(args.vae, weight_dtype, accelerator.device)
                    else:
                        _vae = vae
                    nanosaur_train_util.sample_images(
                        accelerator=accelerator,
                        args=args,
                        epoch=epoch,
                        global_step=global_step,
                        model=accelerator.unwrap_model(model),
                        vae=_vae,
                        text_encoders=[gemma3],  # None when cached, model when online
                        sample_prompts_te_outputs=sample_prompts_te_outputs,
                    )

                # Step-wise checkpoint
                if args.save_every_n_steps and global_step % args.save_every_n_steps == 0:
                    if accelerator.is_main_process:
                        _save_model_checkpoint(args, accelerator, model, save_dtype, epoch, global_step)

                if global_step >= args.max_train_steps:
                    break

        # Epoch-end operations
        if accelerator.is_main_process:
            if args.save_every_n_epochs and (epoch + 1) % args.save_every_n_epochs == 0:
                _save_model_checkpoint(args, accelerator, model, save_dtype, epoch + 1, global_step)

    accelerator.wait_for_everyone()

    # Final save
    if accelerator.is_main_process:
        logger.info("Saving final NanoSaur model")
        _save_model_checkpoint(args, accelerator, model, save_dtype, num_train_epochs, global_step)

    accelerator.end_training()
    logger.info("NanoSaur training complete!")


def _save_model_checkpoint(
    args: argparse.Namespace,
    accelerator,
    model,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
) -> None:
    """Save only the diffusion model weights as safetensors."""
    ckpt_name = f"{args.output_name}_{global_step:08d}.safetensors"
    ckpt_path = os.path.join(args.output_dir, ckpt_name)
    os.makedirs(args.output_dir, exist_ok=True)

    sai_metadata = train_util.get_sai_model_spec(None, args, False, True, False)
    nanosaur_train_util.save_nanosaur_model(
        ckpt_path,
        accelerator.unwrap_model(model),
        sai_metadata,
        save_dtype=save_dtype,
        use_mem_eff_save=getattr(args, "mem_eff_save", False),
    )
    logger.info(f"Saved checkpoint: {ckpt_path}")


# Argument parser


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NanoSaur full fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    train_util.add_masked_loss_arguments(parser)
    add_logging_arguments(parser)
    nanosaur_train_util.add_nanosaur_train_arguments(parser)

    parser.add_argument(
        "--attr_lr",
        type=str,
        default=None,
        help="Per-attribute learning rate overrides. Format: 'attr_pattern=lr,attr_pattern2=lr2'. "
        "Range notation: 'blocks.0-5=1e-5,dec_net=2e-5'. Unmatched params use --learning_rate. "
        "/ アトリビュートごとの学習率。書式: 'attr_pattern=lr,attr2=lr2'",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    train(args)
