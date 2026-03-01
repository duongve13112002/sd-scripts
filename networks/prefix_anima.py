# Prefix tuning network module for Anima LLM Adapter
#
# Learns N continuous prefix vectors prepended to the adapter's cross-attention
# source (Qwen3 side). These discover quality signals in embedding space that
# improve generation across all artist tags.

import os
from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn as nn

from library.utils import setup_logging

import logging

setup_logging()
logger = logging.getLogger(__name__)

# Default Qwen3 hidden dimension
DEFAULT_EMBED_DIM = 1024


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    num_prefix_tokens = network_dim if network_dim is not None else 8

    # Allow override via network_kwargs
    embed_dim = int(kwargs.get("embed_dim", DEFAULT_EMBED_DIM))

    network = PrefixNetwork(
        num_prefix_tokens=num_prefix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
    )
    return network


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    prefix_weight = weights_sd.get("prefix")
    if prefix_weight is None:
        raise ValueError("No 'prefix' key found in weights file")

    num_prefix_tokens, embed_dim = prefix_weight.shape
    network = PrefixNetwork(
        num_prefix_tokens=num_prefix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
    )
    return network, weights_sd


class PrefixNetwork(nn.Module):
    def __init__(self, num_prefix_tokens: int, embed_dim: int, multiplier: float = 1.0):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.embed_dim = embed_dim
        self.multiplier = multiplier

        self.prefix = nn.Parameter(torch.randn(num_prefix_tokens, embed_dim) * 0.02)

        logger.info(f"PrefixNetwork: {num_prefix_tokens} tokens, dim {embed_dim}, {self.prefix.numel()} params")

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if not hasattr(unet, "llm_adapter") or unet.llm_adapter is None:
            raise ValueError("unet does not have an llm_adapter — prefix tuning requires the LLM adapter")
        unet.llm_adapter.prefix_embeds = self.prefix
        logger.info(f"Attached {self.num_prefix_tokens} prefix embeddings to LLM adapter")

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def is_mergeable(self):
        return False

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return [self.prefix]

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        lr = unet_lr or default_lr
        params = [{"params": [self.prefix], "lr": lr}]
        descriptions = ["prefix"]
        return params, descriptions

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None):
        lr = unet_lr or default_lr
        return [{"params": [self.prefix], "lr": lr}]

    def save_weights(self, file, dtype, metadata):
        state_dict = {"prefix": self.prefix.detach().clone().cpu()}
        if dtype is not None:
            state_dict["prefix"] = state_dict["prefix"].to(dtype)

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            metadata["ss_network_module"] = "networks.prefix_anima"
            metadata["ss_num_prefix_tokens"] = str(self.num_prefix_tokens)
            metadata["ss_embed_dim"] = str(self.embed_dim)

            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        if "prefix" in weights_sd:
            self.prefix.data.copy_(weights_sd["prefix"])
            logger.info(f"Loaded prefix weights: {self.prefix.shape}")
        else:
            raise ValueError("No 'prefix' key found in weights file")
