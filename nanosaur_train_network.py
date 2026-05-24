"""
NanoSaur LoRA / network training script.

Usage:
    accelerate launch --mixed_precision bf16 nanosaur_train_network.py \\
      --pretrained_model_name_or_path nanosaur_diffusion_model.safetensors \\
      --text_encoder nanosaur_text_encoder.safetensors \\
      --vae nanosaur_vae_decoder.safetensors \\
      --dataset_config dataset.toml \\
      --output_dir output/ \\
      --output_name my_lora \\
      --network_module networks.lora_nanosaur \\
      --network_dim 16 \\
      --network_alpha 16 \\
      --max_train_steps 5000 \\
      --learning_rate 1e-4 \\
      --mixed_precision bf16 \\
      --blocks_to_swap 10

Full argument reference: docs/nanosaur_train_network.md
"""

import argparse
from typing import Tuple

import torch
from accelerate import Accelerator

import train_network
from library import (
    nanosaur_models,
    nanosaur_train_util,
    nanosaur_utils,
    strategy_base,
    strategy_nanosaur,
    train_util,
)
from library.device_utils import clean_memory_on_device, init_ipex
from library.utils import setup_logging

init_ipex()
setup_logging()
import logging

logger = logging.getLogger(__name__)


class NanoSaurNetworkTrainer(train_network.NetworkTrainer):
    """
    NetworkTrainer subclass for NanoSaur.

    Inherits the full sd-scripts training loop from NetworkTrainer and overrides
    model-specific hooks: loading, strategy creation, loss computation, sampling.
    """

    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_swapping_blocks: bool = False

    # Extra arg validation

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "Enabling cache_text_encoder_outputs because cache_text_encoder_outputs_to_disk is set."
            )
            args.cache_text_encoder_outputs = True

        # NanoSaur VAE stride = 16, bucket steps must be multiple of 16
        train_dataset_group.verify_bucket_reso_steps(16)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

        # Whether to train text encoder LoRA
        self.train_gemma3 = not args.network_train_unet_only

    # Model loading

    def load_target_model(self, args, weight_dtype, accelerator):
        loading_dtype = None if args.fp8_base else weight_dtype

        # Load diffusion model
        model = nanosaur_utils.load_nanosaur_model(
            args.pretrained_model_name_or_path,
            loading_dtype,
            torch.device("cpu"),
            disable_mmap=args.disable_mmap_load_safetensors,
            use_flash_attn=getattr(args, "use_flash_attn", False),
            use_sage_attn=getattr(args, "use_sage_attn", False),
        )

        if args.fp8_base:
            if model.dtype not in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
                torch.float8_e5m2,
                torch.float8_e5m2fnuz,
            ):
                logger.info("Casting NanoSaur model to fp8 (e4m3fn)")
                model.to(torch.float8_e4m3fn)

        if args.blocks_to_swap:
            logger.info(f"NanoSaur: enabling block swap for {args.blocks_to_swap} blocks")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)
            self.is_swapping_blocks = True

        # Load text encoder (Gemma3 + tokenizer)
        gemma3_tokenizer, gemma3 = nanosaur_utils.load_nanosaur_text_encoder(
            args.text_encoder, weight_dtype, "cpu"
        )
        gemma3.eval()
        # Store tokenizer for strategy use
        self._gemma3_tokenizer = gemma3_tokenizer

        # Load VAE
        vae = nanosaur_utils.load_nanosaur_vae(args.vae, weight_dtype, "cpu")

        return nanosaur_utils.MODEL_VERSION_NANOSAUR, [gemma3], vae, model

    # Strategy factories

    def get_tokenize_strategy(self, args):
        return strategy_nanosaur.NanoSaurTokenizeStrategy(
            tokenizer=self._gemma3_tokenizer,
            max_length=nanosaur_models.TEXT_MAX_LENGTH,
        )

    def get_tokenizers(self, tokenize_strategy: strategy_nanosaur.NanoSaurTokenizeStrategy):
        # Return the underlying SentencePiece processor wrapped; no HF tokenizer here
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_nanosaur.NanoSaurLatentsCachingStrategy(
            args.cache_latents_to_disk,
            args.vae_batch_size,
            False,
        )

    def get_text_encoding_strategy(self, args):
        return strategy_nanosaur.NanoSaurTextEncodingStrategy()

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [self.train_gemma3]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_nanosaur.NanoSaurTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                is_partial=self.train_gemma3,
            )
        return None

    # Text encoder caching

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                logger.info("Moving VAE and UNet to CPU to save memory during text encoder caching")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("Moving text encoder to GPU for caching")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # Cache sample prompt TE outputs
            if args.sample_prompts is not None:
                logger.info("Caching text encoder outputs for sample prompts")
                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

                sample_prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}

                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in sample_prompts:
                        for key in ("prompt", "negative_prompt"):
                            prompt = prompt_dict.get(key, "")
                            if prompt in sample_prompts_te_outputs:
                                continue
                            tokens = tokenize_strategy.tokenize(prompt)
                            sample_prompts_te_outputs[prompt] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy, text_encoders, tokens
                            )

                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            if not self.is_train_text_encoder(args):
                logger.info("Moving Gemma3 back to CPU")
                text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("Restoring VAE and UNet to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)

    # Noise scheduler

    def get_noise_scheduler(self, args, device):
        # We use a simple identity scheduler; timestep sampling is done internally
        # We still need a scheduler-like object for compatibility
        return _NanoSaurFlowScheduler()

    # VAE encoding

    def encode_images_to_latents(self, args, vae, images):
        return vae.encode(images)

    def shift_scale_latents(self, args, latents):
        # Scale/shift already applied by NanoSaurVAEWrapper.encode()
        return latents

    # Forward pass and loss

    def get_noise_pred_and_target(
        self,
        args,
        accelerator: Accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds: Tuple,
        dit: nanosaur_models.NanoSaurTransformer2DModel,
        network,
        weight_dtype,
        train_unet: bool,
        is_train: bool = True,
    ):
        # Unpack text encoder conditioning
        # NanoSaurTextEncodingStrategy returns [hidden_states, input_ids, attention_mask]
        hidden_states, input_ids, attention_mask = text_encoder_conds

        # Ensure grads for gradient checkpointing
        if args.gradient_checkpointing:
            latents.requires_grad_(True)
            if hidden_states is not None and hidden_states.dtype.is_floating_point:
                hidden_states.requires_grad_(True)

        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        noisy_input, t = nanosaur_train_util.get_noisy_model_input_and_timesteps(
            args, latents, noise, accelerator.device, weight_dtype
        )

        def call_dit(img, ctx, t_val):
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                # _forward returns x0 prediction
                return dit._forward(img, t_val, ctx)

        x0_pred = call_dit(noisy_input, hidden_states.to(weight_dtype), t)

        # Convert to velocity predictions for the loss
        velocity_pred, velocity_target = nanosaur_train_util.get_flow_matching_loss(
            x0_pred, latents, noisy_input, t
        )

        # Differential output preservation
        if "custom_attributes" in batch:
            diff_indices = [
                i for i, attrs in enumerate(batch["custom_attributes"])
                if attrs.get("diff_output_preservation", False)
            ]
            if diff_indices:
                network.set_multiplier(0.0)
                with torch.no_grad():
                    x0_prior = call_dit(
                        noisy_input[diff_indices],
                        hidden_states[diff_indices].to(weight_dtype),
                        t[diff_indices],
                    )
                    v_prior, _ = nanosaur_train_util.get_flow_matching_loss(
                        x0_prior, latents[diff_indices], noisy_input[diff_indices], t[diff_indices]
                    )
                network.set_multiplier(1.0)
                velocity_target[diff_indices] = v_prior.to(velocity_target.dtype)

        # timesteps in [0, 1] scaled to [0, 1000] for compatibility with base class
        timesteps = (t * 1000).long()
        return velocity_pred, velocity_target, timesteps, None  # weighting=None

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    # Sample generation

    def sample_images(
        self,
        accelerator,
        args,
        epoch,
        global_step,
        device,
        vae,
        tokenizer,
        text_encoder,
        unet,
    ):
        nanosaur_train_util.sample_images(
            accelerator=accelerator,
            args=args,
            epoch=epoch,
            global_step=global_step,
            model=unet,
            vae=vae,
            text_encoders=self.get_models_for_text_encoding(args, accelerator, text_encoder),
            sample_prompts_te_outputs=self.sample_prompts_te_outputs,
        )

    # Model metadata

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, False, True, False)

    def update_metadata(self, metadata, args):
        metadata["ss_model_type"] = "nanosaur"
        metadata["ss_time_sampling_alpha"] = str(getattr(args, "time_sampling_alpha", 2.0))

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Gemma3 embedding table needs grad for gradient checkpointing
        text_encoder.model.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        logger.info(f"Preparing Gemma3 for fp8: set to {te_weight_dtype}")
        text_encoder.to(te_weight_dtype)
        text_encoder.model.embed_tokens.to(dtype=weight_dtype)

    def prepare_unet_with_accelerator(
        self, args, accelerator: Accelerator, unet
    ) -> torch.nn.Module:
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        unet = accelerator.prepare(unet, device_placement=[False])
        accelerator.unwrap_model(unet).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(unet).prepare_block_swap_before_forward()
        return unet

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


# Minimal flow scheduler stub


class _NanoSaurFlowScheduler:
    """
    Minimal scheduler stub compatible with NetworkTrainer expectations.
    NanoSaur uses its own rectified-flow sampling; this exists only to satisfy
    the base class interface.
    """

    def __init__(self):
        self.num_train_timesteps = 1000
        self.config = type("cfg", (), {"num_train_timesteps": 1000})()


# Parser


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    nanosaur_train_util.add_nanosaur_train_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NanoSaurNetworkTrainer()
    trainer.train(args)
