# Cache Anima (WanVAE) latents to disk
# Usage:
#   python anima_cache_latents.py --qwen3_path /path/to/qwen3.safetensors --vae_path /path/to/wan_vae.safetensors \
#       --dataset_config config.toml --vae_batch_size 4 --no_half_vae
#
# Multi-GPU:
#   accelerate launch --num_processes 4 anima_cache_latents.py \
#       --vae_path /path/to/wan_vae.safetensors --dataset_config config.toml ...

import argparse
import os

from accelerate.utils import set_seed
import torch

from library import anima_utils, anima_train_utils, strategy_base, strategy_anima
from library import config_util, train_util
from library.config_util import ConfigSanitizer, BlueprintGenerator
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def cache_to_disk(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    train_util.prepare_dataset_args(args, True)
    train_util.enable_high_vram(args)

    # Force cache to disk
    args.cache_latents = True
    args.cache_latents_to_disk = True

    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)

    # Set tokenize strategy (required by dataset __init__, but tokenizer
    # is NOT actually used during VAE latent caching — only loaded as a
    # lightweight dependency so the dataset can initialize properly)
    logger.info("Loading tokenizers (lightweight, for dataset init only)...")
    qwen3_tokenizer = anima_utils.load_qwen3_tokenizer(args.qwen3_path)

    t5_tokenizer = anima_utils.load_t5_tokenizer(getattr(args, 't5_tokenizer_path', None))

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
        qwen3_max_length=getattr(args, 'qwen3_max_token_length', 512),
        t5_max_length=getattr(args, 't5_max_token_length', 512),
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

    # Set latents caching strategy
    latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(
        True, args.vae_batch_size, args.skip_cache_check
    )
    strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # Prepare dataset
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Loading dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "reg_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignoring the following options because config file is found: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    # Prepare accelerator (handles multi-GPU)
    logger.info("Preparing accelerator...")
    args.deepspeed = False
    accelerator = train_util.prepare_accelerator(args)

    # Prepare dtype
    weight_dtype, _ = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # Load VAE
    logger.info("Loading Anima VAE...")
    vae, vae_mean, vae_std, vae_scale = anima_utils.load_anima_vae(
        args.vae_path, dtype=vae_dtype, device=accelerator.device
    )

    # Cache latents (multi-GPU distribution handled inside new_cache_latents)
    logger.info(f"Caching latents to disk (process {accelerator.process_index}/{accelerator.num_processes})...")
    train_dataset_group.new_cache_latents(vae, accelerator)

    # Cleanup
    vae = None
    train_util.clean_memory_on_device(accelerator.device)

    accelerator.wait_for_everyone()
    accelerator.print("Finished caching latents to disk.")

    # Clean shutdown to avoid "destroy_process_group() was not called" warning
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    config_util.add_config_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)

    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="Use float32 VAE instead of mixed precision dtype",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="[Deprecated] Existing .npz files are always checked. Use --skip_cache_check instead.",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    cache_to_disk(args)
