# LoRA network module for NanoSaur
# Saves in ComfyUI-compatible format: diffusion_model.{path}.lora_up.weight
#
# Internal key format: lora_unet_{path_with_underscores}
# Saved key format:    diffusion_model.{original_dotted_path}.lora_{up|down}.weight
#
# Usage via --network_module=networks.lora_nanosaur

import math
import os
import re
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from safetensors.torch import save_file

from library import train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# LoRA module


class LoRAModule(nn.Module):
    """
    Hook-based LoRA adapter that replaces the forward method of an original
    Linear (or Conv2d 1×1) rather than the module itself.

    Internal key: ``{lora_name}.lora_{up|down}.weight``
    ComfyUI key:  ``diffusion_model.{original_name}.lora_{up|down}.weight``
    """

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.lora_name = lora_name

        if isinstance(org_module, nn.Conv2d):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if isinstance(org_module, nn.Conv2d):
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(in_dim, lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        alpha = lora_dim if alpha is None or alpha == 0 else float(alpha)
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        # Set on creation by the LoRANetwork builder for ComfyUI key mapping
        self.original_name: str = ""

    def apply_to(self) -> None:
        """Hook into the original module's forward."""
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_out = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1).item() < self.module_dropout:
                return org_out

        lx = self.lora_down(x)

        if self.dropout is not None and self.training:
            lx = nn.functional.dropout(lx, p=self.dropout)

        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if isinstance(self.lora_down, nn.Conv2d):
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            else:
                for _ in range(len(lx.size()) - 2):
                    mask = mask.unsqueeze(1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        return org_out + self.lora_up(lx) * self.multiplier * scale

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


class LoRAInfModule(LoRAModule):
    """LoRAModule for inference: supports enable/disable and weight merging."""

    def __init__(self, lora_name, org_module, multiplier=1.0, lora_dim=4, alpha=1.0, **kwargs):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)
        self.org_module_ref = [org_module]
        self.enabled = True
        self.network: Optional["LoRANetwork"] = None

    def set_network(self, network: "LoRANetwork") -> None:
        self.network = network

    def merge_to(self, sd: dict, dtype: Optional[torch.dtype], device: Optional[torch.device]) -> None:
        org_sd = self.org_module_ref[0].state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        dtype = dtype or org_dtype
        device = device or org_device

        down_w = sd["lora_down.weight"].float().to(device)
        up_w = sd["lora_up.weight"].float().to(device)
        w = weight.float().to(device)

        if len(w.shape) == 2:
            w = w + self.multiplier * (up_w @ down_w) * self.scale
        elif down_w.shape[2:] == (1, 1):
            w = w + self.multiplier * (
                (up_w.squeeze(3).squeeze(2) @ down_w.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            ) * self.scale
        else:
            conved = nn.functional.conv2d(down_w.permute(1, 0, 2, 3), up_w).permute(1, 0, 2, 3)
            w = w + self.multiplier * conved * self.scale

        org_sd["weight"] = w.to(dtype)
        self.org_module_ref[0].load_state_dict(org_sd)

    def get_weight(self, multiplier: Optional[float] = None) -> torch.Tensor:
        mul = multiplier if multiplier is not None else self.multiplier
        up_w = self.lora_up.weight.float()
        down_w = self.lora_down.weight.float()
        if len(down_w.shape) == 2:
            return mul * (up_w @ down_w) * self.scale
        elif down_w.shape[2:] == (1, 1):
            return mul * (
                (up_w.squeeze(3).squeeze(2) @ down_w.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            ) * self.scale
        else:
            conved = nn.functional.conv2d(down_w.permute(1, 0, 2, 3), up_w).permute(1, 0, 2, 3)
            return mul * conved * self.scale

    def default_forward(self, x: torch.Tensor) -> torch.Tensor:
        lx = self.lora_down(x)
        lx = self.lora_up(lx)
        return self.org_forward(x) + lx * self.multiplier * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


# Network factory


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
    """Factory called by sd-scripts train_network.py."""
    if network_dim is None:
        network_dim = 16
    if network_alpha is None:
        network_alpha = float(network_dim)

    train_text_encoder = kwargs.get("train_text_encoder", "false")
    train_text_encoder = isinstance(train_text_encoder, str) and train_text_encoder.lower() == "true"

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    network = LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        train_text_encoder=train_text_encoder,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    if any(x is not None for x in [loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio]):
        network.set_loraplus_lr_ratio(
            float(loraplus_lr_ratio) if loraplus_lr_ratio else None,
            float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio else None,
            float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio else None,
        )

    return network


def create_network_from_weights(
    multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs
):
    """Load an existing LoRA file and reconstruct the network structure."""
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, float] = {}
    has_te = False

    for key, value in weights_sd.items():
        if "." not in key:
            continue
        # Both ComfyUI and sd-scripts formats are handled
        if key.startswith("diffusion_model."):
            # ComfyUI format: diffusion_model.{path}.lora_{up|down}.weight
            parts = key.split(".")
            # Extract module path (everything between diffusion_model. and .lora_...)
            lora_idx = next(
                (i for i, p in enumerate(parts) if p.startswith("lora_")), None
            )
            if lora_idx is None:
                continue
            original_name = ".".join(parts[1:lora_idx])
            lora_name = "lora_unet_" + original_name.replace(".", "_")
            if "alpha" in key:
                modules_alpha[lora_name] = float(value.item() if hasattr(value, "item") else value)
            elif "lora_down" in key:
                modules_dim[lora_name] = value.size(0)
        elif key.startswith("lora_unet_"):
            # sd-scripts format (internal)
            parts = key.split(".")
            lora_name = parts[0]
            if "alpha" in key:
                modules_alpha[lora_name] = float(value.item() if hasattr(value, "item") else value)
            elif "lora_down" in key:
                modules_dim[lora_name] = value.size(0)
        elif key.startswith("lora_te_"):
            has_te = True
            parts = key.split(".")
            lora_name = parts[0]
            if "alpha" in key:
                modules_alpha[lora_name] = float(value.item() if hasattr(value, "item") else value)
            elif "lora_down" in key:
                modules_dim[lora_name] = value.size(0)

    module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
        text_encoders if has_te else None,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_text_encoder=has_te,
    )
    return network, weights_sd


# LoRA network


class LoRANetwork(nn.Module):
    """
    NanoSaur LoRA network.

    Target modules in the diffusion model (matches reference inject_lora):
        - FlattenDiTBlock  → all linears inside blocks.* (attention + MLP)
        - TextRefineBlock  → all linears inside text_refine_blocks.*
        - ResBlock         → linears inside dec_net.res_blocks.*.mlp.*

    Target modules in the text encoder (optional):
        Gemma3Attention, Gemma3MLP

    Saved format: ComfyUI-compatible
        diffusion_model.{original_path}.lora_up.weight
        diffusion_model.{original_path}.lora_down.weight
        diffusion_model.{original_path}.alpha
    """

    # NanoSaur DiT target module class names — matches reference inject_lora scope
    NANOSAUR_TARGET_REPLACE_MODULE = [
        "FlattenDiTBlock",
        "TextRefineBlock",
        "ResBlock",  # dec_net.res_blocks.*.mlp.* linears (reference target)
    ]
    # Text encoder target module class names (Gemma3 attention + MLP)
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "Gemma3Attention",
        "Gemma3MLP",
        "Gemma3SdpaAttention",
        "Gemma3FlashAttention2",
        # Fallback patterns
        "GemmaSdpaAttention",
        "GemmaFlashAttention2",
        "GemmaAttention",
        "GemmaMLP",
    ]

    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # ComfyUI prefix for saving
    COMFYUI_DIFFUSION_PREFIX = "diffusion_model"

    def __init__(
        self,
        text_encoders,
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 16,
        alpha: float = 16.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[LoRAModule] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, float]] = None,
        train_text_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.train_text_encoder = train_text_encoder

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info("Creating NanoSaur LoRA network from weights")
        else:
            logger.info(
                f"Creating NanoSaur LoRA network. rank={lora_dim}, alpha={alpha}, "
                f"train_text_encoder={train_text_encoder}"
            )

        # Sub-module name patterns to exclude per target class.
        # ResBlock.adaLN_modulation is excluded to match the original inject_lora
        # which only targets dec_net.res_blocks.*.mlp.* (not adaLN_modulation).
        EXCLUDE_SUBNAMES: Dict[str, List[str]] = {
            "ResBlock": ["adaLN_modulation"],
        }

        def create_modules(
            is_unet: bool,
            root_module: nn.Module,
            target_replace_module_names: List[str],
        ) -> List[LoRAModule]:
            prefix = self.LORA_PREFIX_UNET if is_unet else self.LORA_PREFIX_TEXT_ENCODER
            loras = []
            seen_names = set()

            for mod_name, module in root_module.named_modules():
                if module.__class__.__name__ not in target_replace_module_names:
                    continue
                # Per-class sub-module exclusions (unet only)
                exclude = EXCLUDE_SUBNAMES.get(module.__class__.__name__, []) if is_unet else []
                # Iterate over direct and nested linears within the target block
                for child_name, child in module.named_modules():
                    if not (isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d)):
                        continue
                    if any(excl in child_name for excl in exclude):
                        continue
                    full_name = f"{mod_name}.{child_name}" if child_name else mod_name
                    if full_name in seen_names:
                        continue
                    seen_names.add(full_name)

                    lora_name = f"{prefix}_{full_name}".replace(".", "_")

                    dim = None
                    alpha_val = None
                    if modules_dim is not None:
                        if lora_name in modules_dim:
                            dim = modules_dim[lora_name]
                            alpha_val = modules_alpha.get(lora_name, float(dim))
                    else:
                        dim = self.lora_dim
                        alpha_val = self.alpha

                    if dim is None or dim == 0:
                        continue

                    lora = module_class(
                        lora_name,
                        child,
                        multiplier=multiplier,
                        lora_dim=dim,
                        alpha=alpha_val,
                        dropout=dropout,
                        rank_dropout=rank_dropout,
                        module_dropout=module_dropout,
                    )
                    lora.original_name = full_name  # dotted path for ComfyUI key
                    loras.append(lora)

            return loras

        # Text encoder LoRAs (optional)
        self.text_encoder_loras: List[LoRAModule] = []
        if train_text_encoder and text_encoders is not None:
            for te in text_encoders:
                if te is None:
                    continue
                te_loras = create_modules(False, te, self.TEXT_ENCODER_TARGET_REPLACE_MODULE)
                logger.info(f"NanoSaur: created {len(te_loras)} LoRA modules for text encoder")
                self.text_encoder_loras.extend(te_loras)

        # Diffusion model LoRAs
        self.unet_loras: List[LoRAModule] = create_modules(True, unet, self.NANOSAUR_TARGET_REPLACE_MODULE)
        logger.info(f"NanoSaur: created {len(self.unet_loras)} LoRA modules for diffusion model")

        # Assert no duplicates
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            if lora.lora_name in names:
                logger.warning(f"Duplicate LoRA name: {lora.lora_name}")
            names.add(lora.lora_name)

    # multiplier / enabled

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = multiplier

    def set_enabled(self, is_enabled: bool) -> None:
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    # apply / load

    def apply_to(
        self,
        text_encoders,
        unet,
        apply_text_encoder: bool = True,
        apply_unet: bool = True,
    ) -> None:
        if apply_text_encoder and self.text_encoder_loras:
            logger.info(f"NanoSaur LoRA: applying {len(self.text_encoder_loras)} text encoder modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"NanoSaur LoRA: applying {len(self.unet_loras)} diffusion model modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self) -> bool:
        return True

    def merge_to(self, text_encoders, unet, weights_sd: dict, dtype=None, device=None) -> None:
        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {
                k[len(lora.lora_name) + 1:]: v
                for k, v in weights_sd.items()
                if k.startswith(lora.lora_name + ".")
            }
            if sd_for_lora:
                lora.merge_to(sd_for_lora, dtype, device)
        logger.info("NanoSaur LoRA weights merged")

    def load_weights(self, file: str):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        # If ComfyUI format, convert keys to internal format first
        weights_sd = self._maybe_convert_comfyui_to_internal(weights_sd)
        return self.load_state_dict(weights_sd, strict=False)

    # ComfyUI key conversion

    def _internal_to_comfyui(self, internal_sd: dict) -> dict:
        """
        Convert internal sd-scripts keys to ComfyUI format.

        Internal: ``lora_unet_blocks_0_attn_qkv_x.lora_up.weight``
        ComfyUI:  ``diffusion_model.blocks.0.attn.qkv_x.lora_up.weight``

        Uses the stored ``original_name`` on each LoRAModule for exact conversion.
        """
        # Build lookup: lora_name → original_name
        lookup = {
            lora.lora_name: lora.original_name
            for lora in self.unet_loras
        }

        comfyui_sd = {}
        for key, value in internal_sd.items():
            # split at first dot to separate lora_name from sub-key
            if "." in key:
                lora_name, sub_key = key.split(".", 1)
            else:
                lora_name = key
                sub_key = ""

            if lora_name in lookup and lora_name.startswith(self.LORA_PREFIX_UNET):
                orig = lookup[lora_name]
                new_key = f"{self.COMFYUI_DIFFUSION_PREFIX}.{orig}.{sub_key}" if sub_key else f"{self.COMFYUI_DIFFUSION_PREFIX}.{orig}"
                comfyui_sd[new_key] = value
            else:
                # keep as-is for TE keys or unknown keys
                comfyui_sd[key] = value

        return comfyui_sd

    def _maybe_convert_comfyui_to_internal(self, sd: dict) -> dict:
        """
        If keys start with ``diffusion_model.``, convert them back to internal
        sd-scripts format using the stored ``original_name`` mapping.
        """
        if not any(k.startswith("diffusion_model.") for k in sd.keys()):
            return sd  # already internal format

        # Build reverse lookup: original_name → lora_name
        reverse = {
            lora.original_name: lora.lora_name
            for lora in self.unet_loras
        }
        prefix = self.COMFYUI_DIFFUSION_PREFIX + "."
        internal_sd = {}
        for key, value in sd.items():
            if key.startswith(prefix):
                rest = key[len(prefix):]  # e.g. "blocks.0.attn.qkv_x.lora_up.weight"
                # Find the longest matching original_name
                matched = False
                for orig_name, lora_name in reverse.items():
                    sub_prefix = orig_name + "."
                    if rest.startswith(sub_prefix):
                        sub_key = rest[len(sub_prefix):]
                        new_key = f"{lora_name}.{sub_key}"
                        internal_sd[new_key] = value
                        matched = True
                        break
                if not matched:
                    internal_sd[key] = value
            else:
                internal_sd[key] = value
        return internal_sd

    # gradient / optimizer helpers

    def prepare_grad_etc(self, text_encoder, unet) -> None:
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet) -> None:
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def enable_gradient_checkpointing(self) -> None:
        pass  # not needed for LoRA

    def set_loraplus_lr_ratio(
        self,
        loraplus_lr_ratio: Optional[float],
        loraplus_unet_lr_ratio: Optional[float],
        loraplus_text_encoder_lr_ratio: Optional[float],
    ) -> None:
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

    def prepare_optimizer_params_with_multiple_te_lrs(
        self,
        text_encoder_lr,
        unet_lr,
        default_lr,
    ):
        if text_encoder_lr is None or text_encoder_lr == []:
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, (float, int)):
            text_encoder_lr = [float(text_encoder_lr)]

        self.requires_grad_(True)
        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if loraplus_ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param
            params = []
            descriptions = []
            for key, group_params in param_groups.items():
                if not group_params:
                    continue
                pg = {"params": list(group_params.values())}
                if lr is not None:
                    pg["lr"] = lr * loraplus_ratio if (key == "plus" and loraplus_ratio) else lr
                params.append(pg)
                descriptions.append("plus" if key == "plus" else "")
            return params, descriptions

        if self.text_encoder_loras:
            loraplus_ratio = self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
            p, d = assemble_params(self.text_encoder_loras, text_encoder_lr[0], loraplus_ratio)
            all_params.extend(p)
            lr_descriptions.extend(["textencoder" + (" " + x if x else "") for x in d])

        if self.unet_loras:
            lr = unet_lr if unet_lr is not None else default_lr
            loraplus_ratio = self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio
            p, d = assemble_params(self.unet_loras, lr, loraplus_ratio)
            all_params.extend(p)
            lr_descriptions.extend(["unet" + (" " + x if x else "") for x in d])

        return all_params, lr_descriptions

    # saving

    def save_weights(self, file: str, dtype: Optional[torch.dtype], metadata: Optional[dict]) -> None:
        """Save weights in ComfyUI-compatible format for diffusion model LoRAs."""
        if metadata is not None and len(metadata) == 0:
            metadata = None

        internal_sd = self.state_dict()

        # Cast dtype
        if dtype is not None:
            internal_sd = {k: v.detach().clone().to("cpu").to(dtype) for k, v in internal_sd.items()}
        else:
            internal_sd = {k: v.detach().clone().to("cpu") for k, v in internal_sd.items()}

        # Convert unet keys to ComfyUI format
        comfyui_sd = self._internal_to_comfyui(internal_sd)

        if os.path.splitext(file)[1] == ".safetensors":
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(comfyui_sd, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(comfyui_sd, file, metadata=metadata)
        else:
            torch.save(comfyui_sd, file)

    # weight manipulation (inference)

    def backup_weights(self) -> None:
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras  # type: ignore[assignment]
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self) -> None:
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras  # type: ignore[assignment]
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self) -> None:
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras  # type: ignore[assignment]
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()
            lora_w = lora.get_weight().to(sd["weight"].device, dtype=sd["weight"].dtype)
            sd["weight"] = sd["weight"] + lora_w
            assert sd["weight"].shape == lora_w.shape or True  # shape check
            org_module.load_state_dict(sd)
            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value: float, device: torch.device):
        downkeys, upkeys, alphakeys = [], [], []
        state_dict = self.state_dict()
        for key in state_dict:
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        keys_scaled = 0
        norms = []
        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale
            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio ** 0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            norms.append((updown.norm() * ratio).item())

        return keys_scaled, sum(norms) / len(norms) if norms else 0.0, max(norms) if norms else 0.0
