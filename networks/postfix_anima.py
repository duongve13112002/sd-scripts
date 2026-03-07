# Postfix tuning network module for Anima LLM Adapter
#
# Learns N continuous postfix vectors appended to the adapter's cross-attention
# source (Qwen3 side). These discover quality signals in embedding space that
# improve generation across all artist tags.
#
# Modes:
#   "hidden"    — postfix injected into adapter's source (Qwen3 hidden-state space)
#   "embedding" — postfix injected into Qwen3 input embeddings (goes through all layers)
#   "cfg"       — two postfix sets (pos + neg) trained with CFG-aware loss

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
    num_postfix_tokens = network_dim if network_dim is not None else 8

    # Allow override via network_kwargs
    embed_dim = int(kwargs.get("embed_dim", DEFAULT_EMBED_DIM))
    mode = kwargs.get("mode", "hidden")
    cfg_scale = float(kwargs.get("cfg_scale", 4.0))

    network = PostfixNetwork(
        num_postfix_tokens=num_postfix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
        mode=mode,
        cfg_scale=cfg_scale,
    )
    return network


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # Detect mode from keys
    if "postfix_pos" in weights_sd:
        mode = "cfg"
        postfix_weight = weights_sd["postfix_pos"]
    else:
        mode = "hidden"
        postfix_weight = weights_sd.get("postfix", weights_sd.get("prefix"))
    if postfix_weight is None:
        raise ValueError("No 'postfix'/'postfix_pos' (or legacy 'prefix') key found in weights file")

    num_postfix_tokens, embed_dim = postfix_weight.shape
    network = PostfixNetwork(
        num_postfix_tokens=num_postfix_tokens,
        embed_dim=embed_dim,
        multiplier=multiplier,
        mode=mode,
    )
    return network, weights_sd


class PostfixNetwork(nn.Module):
    def __init__(self, num_postfix_tokens: int, embed_dim: int, multiplier: float = 1.0, mode: str = "hidden", cfg_scale: float = 4.0):
        super().__init__()
        self.num_postfix_tokens = num_postfix_tokens
        self.embed_dim = embed_dim
        self.multiplier = multiplier
        self.mode = mode
        self.cfg_scale = cfg_scale

        if mode == "cfg":
            self.postfix_pos = nn.Parameter(torch.randn(num_postfix_tokens, embed_dim) * 0.02)
            self.postfix_neg = nn.Parameter(torch.randn(num_postfix_tokens, embed_dim) * 0.02)
            total_params = self.postfix_pos.numel() + self.postfix_neg.numel()
            logger.info(
                f"PostfixNetwork: {num_postfix_tokens} tokens × 2 (pos+neg), dim {embed_dim}, "
                f"cfg_scale={cfg_scale}, {total_params} params"
            )
        else:
            self.postfix = nn.Parameter(torch.randn(num_postfix_tokens, embed_dim) * 0.02)
            logger.info(f"PostfixNetwork: {num_postfix_tokens} tokens, dim {embed_dim}, mode={mode}, {self.postfix.numel()} params")

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if self.mode == "cfg":
            if not hasattr(unet, "llm_adapter") or unet.llm_adapter is None:
                raise ValueError("unet does not have an llm_adapter — cfg postfix requires the LLM adapter")
            # Set positive postfix as default (training loop swaps for neg pass)
            unet.llm_adapter.postfix_embeds = self.postfix_pos
            logger.info(
                f"Attached {self.num_postfix_tokens} postfix embeddings (pos+neg) to LLM adapter "
                f"(cfg mode, scale={self.cfg_scale})"
            )
        elif self.mode == "embedding":
            if text_encoders is None:
                raise ValueError("embedding mode requires live text encoder (cannot use --cache_text_encoder_outputs)")
            te = text_encoders[0] if isinstance(text_encoders, (list, tuple)) else text_encoders
            self._wrap_text_encoder(te)
            logger.info(f"Attached {self.num_postfix_tokens} postfix embeddings to text encoder (embedding mode)")
        else:
            if not hasattr(unet, "llm_adapter") or unet.llm_adapter is None:
                raise ValueError("unet does not have an llm_adapter — postfix tuning requires the LLM adapter")
            unet.llm_adapter.postfix_embeds = self.postfix
            logger.info(f"Attached {self.num_postfix_tokens} postfix embeddings to LLM adapter")

    def _wrap_text_encoder(self, text_encoder):
        original_forward = text_encoder.forward
        postfix = self.postfix

        def wrapped_forward(*args, **kwargs):
            input_ids = kwargs.pop("input_ids", None)
            if input_ids is None and args:
                input_ids = args[0]
                args = args[1:]

            if input_ids is not None:
                inputs_embeds = text_encoder.embed_tokens(input_ids)
                B = inputs_embeds.shape[0]
                extra = postfix.unsqueeze(0).expand(B, -1, -1).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                inputs_embeds = torch.cat([inputs_embeds, extra], dim=1)
                kwargs["inputs_embeds"] = inputs_embeds

                attention_mask = kwargs.get("attention_mask")
                if attention_mask is not None:
                    extra_mask = torch.ones(B, postfix.shape[0], device=attention_mask.device, dtype=attention_mask.dtype)
                    kwargs["attention_mask"] = torch.cat([attention_mask, extra_mask], dim=1)

            return original_forward(*args, **kwargs)

        text_encoder.forward = wrapped_forward
        self._original_forward = original_forward

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
        if self.mode == "cfg":
            return [self.postfix_pos, self.postfix_neg]
        return [self.postfix]

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        lr = unet_lr or default_lr
        if self.mode == "cfg":
            params = [{"params": [self.postfix_pos, self.postfix_neg], "lr": lr}]
            descriptions = ["postfix_pos+neg"]
        else:
            params = [{"params": [self.postfix], "lr": lr}]
            descriptions = ["postfix"]
        return params, descriptions

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None):
        lr = unet_lr or default_lr
        if self.mode == "cfg":
            return [{"params": [self.postfix_pos, self.postfix_neg], "lr": lr}]
        return [{"params": [self.postfix], "lr": lr}]

    def save_weights(self, file, dtype, metadata):
        if self.mode == "cfg":
            state_dict = {
                "postfix_pos": self.postfix_pos.detach().clone().cpu(),
                "postfix_neg": self.postfix_neg.detach().clone().cpu(),
            }
            if dtype is not None:
                state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
        else:
            state_dict = {"postfix": self.postfix.detach().clone().cpu()}
            if dtype is not None:
                state_dict["postfix"] = state_dict["postfix"].to(dtype)

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            metadata["ss_network_module"] = "networks.postfix_anima"
            metadata["ss_num_postfix_tokens"] = str(self.num_postfix_tokens)
            metadata["ss_embed_dim"] = str(self.embed_dim)
            metadata["ss_mode"] = self.mode
            if self.mode == "cfg":
                metadata["ss_cfg_scale"] = str(self.cfg_scale)

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

        if self.mode == "cfg":
            if "postfix_pos" in weights_sd:
                self.postfix_pos.data.copy_(weights_sd["postfix_pos"])
                self.postfix_neg.data.copy_(weights_sd["postfix_neg"])
                logger.info(f"Loaded cfg postfix weights: pos={self.postfix_pos.shape}, neg={self.postfix_neg.shape}")
            else:
                raise ValueError("No 'postfix_pos' key found in weights file for cfg mode")
        else:
            weight = weights_sd.get("postfix", weights_sd.get("prefix"))
            if weight is not None:
                self.postfix.data.copy_(weight)
                logger.info(f"Loaded postfix weights: {self.postfix.shape}")
            else:
                raise ValueError("No 'postfix' (or legacy 'prefix') key found in weights file")
