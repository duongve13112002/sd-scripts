# Copyright 2026 Linus (Anima DiT Adapter Contributors)
# Licensed under the Apache License, Version 2.0
# coding=utf-8

import os
import argparse
import logging
import gc
from collections import deque, defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, random_split
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from einops import rearrange
from safetensors.torch import save_file, load_file
from prodigyopt import Prodigy
from PIL import Image
import torchvision.transforms as T
import numpy as np

from library.device_utils import init_ipex, clean_memory_on_device

from library import anima_utils, strategy_anima, strategy_base, flux_train_utils
from library.qwen_image_autoencoder_kl import AutoencoderKLQwenImage
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler
from library.anima_train_utils import compute_loss_weighting_for_anima
from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

VAE_DOWNSAMPLE_FACTOR = 8

DIT_SPATIAL_PATCH_SIZE = 2

IMAGE_SIZE_MULTIPLE = VAE_DOWNSAMPLE_FACTOR * DIT_SPATIAL_PATCH_SIZE

tokenize_strategy: Optional[strategy_anima.AnimaTokenizeStrategy] = None
text_encoding_strategy: Optional[strategy_anima.AnimaTextEncodingStrategy] = None


class ZeroLinear(nn.Module):
    """零初始化线性层，训练初期输出严格为0"""

    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.linear = nn.Linear(in_channels, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ModTextAdapter(nn.Module):
    """调制式文本增强器（ICLR 2026 Mod-Adapter）"""

    def __init__(self, text_dim: int = 2048, dit_dim: int = 1152, enable: bool = True):
        super().__init__()
        self.enable = enable
        if not enable:
            return
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, dit_dim * 2),
            nn.SiLU(),
            ZeroLinear(dit_dim * 2)
        )

    def forward(self, text_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enable:
            return None, None
        global_text = text_embeds.mean(dim=1)
        gamma, beta = self.text_proj(global_text).chunk(2, dim=-1)
        return gamma, beta


DIT_HIDDEN_SIZE = 2048  # 图像隐层
TEXT_EMBED_DIM = 1024  # 文本条件


class SemanticScaleAdapter(nn.Module):
    """语义缩放适配器（解决DiT文本嵌入语义瓶颈）"""

    def __init__(self, text_dim=1024, enable=True):
        super().__init__()
        self.enable = enable
        self.scale = nn.Parameter(torch.ones(text_dim))
        self.shift = nn.Parameter(torch.zeros(text_dim))

    def forward(self, text_embeds):
        if not self.enable:
            return text_embeds
        return text_embeds * self.scale + self.shift


class LocalConvAdapter(nn.Module):
    """局部卷积适配器，兼容 [B,N,C] 和 [B,T,H,W,C]，自动修复偶数 kernel 导致的尺寸错误"""

    def __init__(self, dim: int = 2048, kernel_size: int = 3, enable: bool = True):
        super().__init__()
        self.enable = enable
        if not enable:
            return

        kernel_size = int(kernel_size)

        if kernel_size < 1:
            kernel_size = 3

        if kernel_size % 2 == 0:
            fixed_kernel = kernel_size - 1
            if fixed_kernel < 1:
                fixed_kernel = 3
            logger.warning(
                f"⚠️ LocalConvAdapter 收到偶数 kernel_size={kernel_size}，"
                f"会导致输出尺寸不一致，已自动修正为 {fixed_kernel}"
            )
            kernel_size = fixed_kernel

        self.kernel_size = kernel_size

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=dim,
        )
        self.pwconv = ZeroLinear(dim)

    def forward(self, x: torch.Tensor, h: int = None, w: int = None) -> torch.Tensor:
        if not self.enable:
            return x

        init_ndim = x.ndim

        if init_ndim == 3:
            if h is None or w is None:
                return x

            b, n, c = x.shape
            hw = h * w
            if hw <= 0 or n % hw != 0:
                return x

            t = n // hw
            x = rearrange(x, "b (t h w) c -> b t h w c", t=t, h=h, w=w)

        if x.ndim != 5:
            return x

        b, t, hh, ww, c = x.shape

        x_2d = rearrange(x, "b t h w c -> (b t) c h w")
        x_2d = self.dwconv(x_2d)

        if x_2d.shape[-2] != hh or x_2d.shape[-1] != ww:
            x_2d = x_2d[..., :hh, :ww]
            pad_h = hh - x_2d.shape[-2]
            pad_w = ww - x_2d.shape[-1]
            if pad_h > 0 or pad_w > 0:
                x_2d = F.pad(x_2d, (0, pad_w, 0, pad_h), mode="constant", value=0)

        x_conv = rearrange(x_2d, "(b t) c h w -> b t h w c", b=b, t=t)

        out = x + self.pwconv(x_conv)

        if init_ndim == 3:
            out = rearrange(out, "b t h w c -> b (t h w) c")

        return out


class StyleCrossAttention(nn.Module):
    """画风交叉注意力（增强画风绑定能力）"""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = ZeroLinear(query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        init_ndim = x.ndim
        if init_ndim == 5:
            b, t, h, w, c = x.shape
            x = rearrange(x, 'b t h w c -> b (t h w) c')

        b, n, _ = x.shape
        _, m, _ = context.shape
        q = self.to_q(x).view(b, n, self.num_heads, -1).transpose(1, 2)
        k = self.to_k(context).view(b, m, self.num_heads, -1).transpose(1, 2)
        v = self.to_v(context).view(b, m, self.num_heads, -1).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, -1)
        out = self.to_out(out)

        if init_ndim == 5:
            out = rearrange(out, 'b (t h w) c -> b t h w c', t=t, h=h, w=w)
        return out


class EdgeDetailConv(nn.Module):
    """边缘细节增强，兼容 [B,N,C] 和 [B,T,H,W,C]"""

    def __init__(self, dim: int = 2048, kernel_size: int = 3, enable: bool = True):
        super().__init__()
        self.enable = enable
        if not enable:
            return

        kernel = torch.tensor(
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ],
            dtype=torch.float32,
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
        self.register_buffer("laplacian_kernel", kernel)

        self.out = ZeroLinear(dim)

    def forward(self, x: torch.Tensor, h: int = None, w: int = None) -> torch.Tensor:
        if not self.enable:
            return x

        init_ndim = x.ndim

        if init_ndim == 3:
            if h is None or w is None:
                return x

            b, n, c = x.shape
            hw = h * w
            if hw <= 0 or n % hw != 0:
                return x

            t = n // hw
            x = rearrange(x, "b (t h w) c -> b t h w c", t=t, h=h, w=w)

        if x.ndim != 5:
            return x

        b, t, hh, ww, c = x.shape

        kernel = self.laplacian_kernel
        if kernel.device != x.device or kernel.dtype != x.dtype:
            kernel = kernel.to(device=x.device, dtype=x.dtype)

        x_2d = rearrange(x, "b t h w c -> (b t) c h w")
        edge = F.conv2d(x_2d, kernel, padding=1, groups=c)
        edge = rearrange(edge, "(b t) c h w -> b t h w c", b=b, t=t)

        out = x + self.out(edge)

        if init_ndim == 3:
            out = rearrange(out, "b t h w c -> b (t h w) c")

        return out


class SubjectCrossAttention(nn.Module):
    """主体交叉注意力（ICML 2026）
    文本绑定主体，防止画风侵蚀人物/物体，解决"画风吞主体"通病
    推荐挂载：Block6~11（中层）
    """

    def __init__(self, query_dim: int, text_dim: int = 2048, num_heads: int = 16, enable: bool = True):
        super().__init__()
        self.enable = enable
        if not enable:
            return
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(text_dim, query_dim, bias=False)
        self.to_v = nn.Linear(text_dim, query_dim, bias=False)
        self.to_out = ZeroLinear(query_dim)

    def forward(self, x: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return x

        init_ndim = x.ndim

        if init_ndim == 5:
            b, t, h, w, c = x.shape
            x = rearrange(x, 'b t h w c -> b (t h w) c')

        b, n, _ = x.shape
        _, m, _ = text_embeds.shape
        q = self.to_q(x).view(b, n, self.num_heads, -1).transpose(1, 2)
        k = self.to_k(text_embeds).view(b, m, self.num_heads, -1).transpose(1, 2)
        v = self.to_v(text_embeds).view(b, m, self.num_heads, -1).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, -1)
        out = x + self.to_out(out)

        if init_ndim == 5:
            out = rearrange(out, 'b (t h w) c -> b t h w c', t=t, h=h, w=w)
        return out


class ContrastModAdapter(nn.Module):
    """对比度调制适配器（全局）
    文本引导特征对比度缩放，解决Anima画面发灰、过曝问题
    全层生效，不占用Block
    """

    def __init__(self, text_dim: int = 2048, enable: bool = True):
        super().__init__()
        self.enable = enable
        if not enable:
            return
        self.gamma = nn.Parameter(torch.ones(text_dim))
        self.beta = nn.Parameter(torch.zeros(text_dim))

    def forward(self, text_embeds: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return text_embeds
        return text_embeds * self.gamma + self.beta


class ColorTuneAdapter(nn.Module):
    """色彩调制适配器（全局）
    Qwen文本控制特征冷暖色偏移，精准控制画面色调
    全层生效，不占用Block
    """

    def __init__(self, text_dim: int = 2048, enable: bool = True):
        super().__init__()
        self.enable = enable
        if not enable:
            return
        self.shift = nn.Parameter(torch.zeros(text_dim))

    def forward(self, text_embeds: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return text_embeds
        return text_embeds + self.shift


class AnimaDiTAdapter(nn.Module):
    """
    Anima DiT Adapter。

    修复点：
    1. 文本全局模块真正作用到 prompt_emb。
    2. ModTextAdapter 真正作用到 DiT hidden。
    3. EdgeDetail / LocalConv 支持 3D hidden。
    4. 支持 gradient checkpointing。
    """

    def __init__(
            self,
            dit_hidden_size: int = 2048,
            num_blocks: int = 28,
            text_embed_dim: int = 1024,
            style_dim: int = 768,
            patch_size: int = 2,
            gradient_checkpointing: bool = False,

            enable_mod_text: bool = True,
            enable_semantic_scale: bool = True,
            enable_local_conv: bool = True,
            local_conv_layers: int = 6,
            local_conv_kernel: int = 3,
            local_conv_start_layer: int = None,

            enable_style_attn: bool = False,
            style_attn_layers: int = 8,
            style_num_heads: int = 16,
            style_attn_start_layer: int = None,

            enable_edge_detail: bool = False,
            edge_detail_layers: int = 5,
            edge_detail_kernel: int = 3,
            edge_detail_start_layer: int = 0,

            enable_subject_attn: bool = False,
            subject_attn_layers: int = 6,
            subject_attn_heads: int = 16,
            subject_attn_start_layer: int = 6,

            enable_contrast_mod: bool = False,
            enable_color_tune: bool = False,
    ):
        super().__init__()

        self.dit_hidden_size = dit_hidden_size
        self.total_blocks = num_blocks
        self.text_embed_dim = text_embed_dim
        self.style_dim = style_dim
        self.patch_size = max(1, patch_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        def _check_and_clip(start_layer: Optional[int], layers: int, default_start: int):
            if start_layer is None:
                start_layer = default_start
            start_layer = max(0, min(int(start_layer), self.total_blocks - 1))
            layers = max(0, min(int(layers), self.total_blocks - start_layer))
            return start_layer, layers

        self.enable_mod_text = enable_mod_text
        self.enable_semantic_scale = enable_semantic_scale
        self.enable_local_conv = enable_local_conv
        self.enable_style_attn = enable_style_attn
        self.enable_edge_detail = enable_edge_detail
        self.enable_subject_attn = enable_subject_attn
        self.enable_contrast_mod = enable_contrast_mod
        self.enable_color_tune = enable_color_tune

        self.local_conv_start_layer, self.local_conv_layers = _check_and_clip(
            local_conv_start_layer,
            local_conv_layers,
            self.total_blocks - local_conv_layers,
        )

        self.style_attn_start_layer, self.style_attn_layers = _check_and_clip(
            style_attn_start_layer,
            style_attn_layers,
            self.total_blocks - style_attn_layers,
        )

        self.edge_detail_start_layer, self.edge_detail_layers = _check_and_clip(
            edge_detail_start_layer,
            edge_detail_layers,
            0,
        )

        self.subject_attn_start_layer, self.subject_attn_layers = _check_and_clip(
            subject_attn_start_layer,
            subject_attn_layers,
            6,
        )

        self.semantic_scale = SemanticScaleAdapter(
            text_dim=self.text_embed_dim,
            enable=self.enable_semantic_scale,
        )

        self.mod_text = ModTextAdapter(
            text_dim=self.text_embed_dim,
            dit_dim=self.dit_hidden_size,
            enable=self.enable_mod_text,
        )

        self.contrast_mod = ContrastModAdapter(
            text_dim=self.text_embed_dim,
            enable=self.enable_contrast_mod,
        )

        self.color_tune = ColorTuneAdapter(
            text_dim=self.text_embed_dim,
            enable=self.enable_color_tune,
        )

        self.local_convs = nn.ModuleList()
        if self.enable_local_conv and self.local_conv_layers > 0:
            for _ in range(self.local_conv_layers):
                self.local_convs.append(
                    LocalConvAdapter(
                        dim=self.dit_hidden_size,
                        kernel_size=local_conv_kernel,
                    )
                )

        self.edge_details = nn.ModuleList()
        if self.enable_edge_detail and self.edge_detail_layers > 0:
            for _ in range(self.edge_detail_layers):
                self.edge_details.append(
                    EdgeDetailConv(
                        dim=self.dit_hidden_size,
                        kernel_size=edge_detail_kernel,
                    )
                )

        self.subject_blocks = nn.ModuleList()
        if self.enable_subject_attn and self.subject_attn_layers > 0:
            for _ in range(self.subject_attn_layers):
                self.subject_blocks.append(
                    SubjectCrossAttention(
                        query_dim=self.dit_hidden_size,
                        text_dim=self.text_embed_dim,
                        num_heads=subject_attn_heads,
                    )
                )

        self.style_blocks = nn.ModuleList()
        self.style_scale = nn.Parameter(torch.zeros(max(1, self.style_attn_layers)))
        if self.enable_style_attn and self.style_attn_layers > 0:
            for _ in range(self.style_attn_layers):
                self.style_blocks.append(
                    StyleCrossAttention(
                        query_dim=self.dit_hidden_size,
                        context_dim=self.style_dim,
                        num_heads=style_num_heads,
                    )
                )

        self._patched = False
        self.style_embeds = None
        self._text_embeds_cache = None
        self._mod_gamma = None
        self._mod_beta = None
        self._current_h = 64
        self._current_w = 64

        self._print_layer_distribution()

    def _print_layer_distribution(self):
        logger.info("📊 Adapter模块层位分布:")

        if self.enable_local_conv and self.local_conv_layers > 0:
            blocks = list(range(self.local_conv_start_layer, self.local_conv_start_layer + self.local_conv_layers))
            logger.info(f"  LocalConv: Block{blocks[0]}~{blocks[-1]}")

        if self.enable_edge_detail and self.edge_detail_layers > 0:
            blocks = list(range(self.edge_detail_start_layer, self.edge_detail_start_layer + self.edge_detail_layers))
            logger.info(f"  EdgeDetail: Block{blocks[0]}~{blocks[-1]}")

        if self.enable_subject_attn and self.subject_attn_layers > 0:
            blocks = list(
                range(self.subject_attn_start_layer, self.subject_attn_start_layer + self.subject_attn_layers))
            logger.info(f"  SubjectAttn: Block{blocks[0]}~{blocks[-1]}")

        if self.enable_style_attn and self.style_attn_layers > 0:
            blocks = list(range(self.style_attn_start_layer, self.style_attn_start_layer + self.style_attn_layers))
            logger.info(f"  StyleAttn: Block{blocks[0]}~{blocks[-1]}")

        logger.info(
            f"  全局模块: "
            f"SemanticScale {'✅' if self.enable_semantic_scale else '❌'} | "
            f"ModText {'✅' if self.enable_mod_text else '❌'} | "
            f"ContrastMod {'✅' if self.enable_contrast_mod else '❌'} | "
            f"ColorTune {'✅' if self.enable_color_tune else '❌'}"
        )

        logger.info(f"  GradientCheckpointing: {'✅' if self.gradient_checkpointing else '❌'}")

    def set_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = bool(enabled)
        logger.info(f"✅ Adapter block gradient checkpointing = {self.gradient_checkpointing}")

    def set_current_resolution(self, h: int, w: int):
        h = int(h)
        w = int(w)

        if h <= 0 or w <= 0:
            raise RuntimeError(f"❌ set_current_resolution 收到非法 latent 尺寸: h={h}, w={w}")

        self._current_h = max(1, h // self.patch_size)
        self._current_w = max(1, w // self.patch_size)

    def set_style_embeds(self, style_embeds: torch.Tensor):
        self.style_embeds = style_embeds

    def set_text_embeds_cache(self, text_embeds: torch.Tensor):
        self._text_embeds_cache = text_embeds

    def prepare_text_condition(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        训练前必须调用。
        这里让 SemanticScale / ContrastMod / ColorTune 真正接入训练图。
        同时计算 ModText 的 gamma/beta，供 block hidden 调制使用。
        """
        if text_embeds is None:
            self._text_embeds_cache = None
            self._mod_gamma = None
            self._mod_beta = None
            return text_embeds

        x = text_embeds

        if self.enable_semantic_scale:
            x = self.semantic_scale(x)

        if self.enable_contrast_mod:
            x = self.contrast_mod(x)

        if self.enable_color_tune:
            x = self.color_tune(x)

        if self.enable_mod_text:
            gamma, beta = self.mod_text(x)
            self._mod_gamma = gamma
            self._mod_beta = beta
        else:
            self._mod_gamma = None
            self._mod_beta = None

        self._text_embeds_cache = x
        return x

    def _apply_mod_text_to_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.enable_mod_text:
            return hidden

        if self._mod_gamma is None or self._mod_beta is None:
            return hidden

        gamma = self._mod_gamma.to(device=hidden.device, dtype=hidden.dtype)
        beta = self._mod_beta.to(device=hidden.device, dtype=hidden.dtype)

        if hidden.ndim == 3:
            gamma = gamma[:, None, :]
            beta = beta[:, None, :]
        elif hidden.ndim == 5:
            gamma = gamma[:, None, None, None, :]
            beta = beta[:, None, None, None, :]
        else:
            return hidden

        return hidden * (1.0 + gamma) + beta

    def _run_original_block(self, original_forward, args, kwargs):
        if not self.gradient_checkpointing or not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)

        tensor_args = [x for x in args if torch.is_tensor(x)]
        has_grad_input = any(x.requires_grad for x in tensor_args)

        if not has_grad_input:
            return original_forward(*args, **kwargs)

        from torch.utils.checkpoint import checkpoint

        def custom_forward(*inputs):
            return original_forward(*inputs, **kwargs)

        return checkpoint(
            custom_forward,
            *args,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def patch_dit_blocks(self, dit_model):
        if self._patched:
            return

        adapter_self = self

        if not hasattr(dit_model, "blocks"):
            raise RuntimeError("❌ dit_model 没有 blocks 属性，无法 patch Adapter")

        for idx, block in enumerate(dit_model.blocks):
            original_forward = block.forward

            has_conv = (
                    self.enable_local_conv
                    and self.local_conv_layers > 0
                    and self.local_conv_start_layer <= idx < self.local_conv_start_layer + self.local_conv_layers
            )
            conv_idx = idx - self.local_conv_start_layer if has_conv else -1

            has_edge = (
                    self.enable_edge_detail
                    and self.edge_detail_layers > 0
                    and self.edge_detail_start_layer <= idx < self.edge_detail_start_layer + self.edge_detail_layers
            )
            edge_idx = idx - self.edge_detail_start_layer if has_edge else -1

            has_subj = (
                    self.enable_subject_attn
                    and self.subject_attn_layers > 0
                    and self.subject_attn_start_layer <= idx < self.subject_attn_start_layer + self.subject_attn_layers
            )
            subj_idx = idx - self.subject_attn_start_layer if has_subj else -1

            has_style = (
                    self.enable_style_attn
                    and self.style_attn_layers > 0
                    and self.style_attn_start_layer <= idx < self.style_attn_start_layer + self.style_attn_layers
            )
            style_idx = idx - self.style_attn_start_layer if has_style else -1

            def new_forward(
                    block_self,
                    *args,
                    _orig=original_forward,
                    _has_conv=has_conv,
                    _conv_idx=conv_idx,
                    _has_edge=has_edge,
                    _edge_idx=edge_idx,
                    _has_subj=has_subj,
                    _subj_idx=subj_idx,
                    _has_style=has_style,
                    _style_idx=style_idx,
                    **kwargs,
            ):
                out = adapter_self._run_original_block(_orig, args, kwargs)

                is_tuple = isinstance(out, tuple)
                hidden = out[0] if is_tuple else out

                h = adapter_self._current_h
                w = adapter_self._current_w

                hidden = adapter_self._apply_mod_text_to_hidden(hidden)

                if _has_edge:
                    hidden = adapter_self.edge_details[_edge_idx](hidden, h, w)

                if _has_subj and adapter_self._text_embeds_cache is not None:
                    hidden = adapter_self.subject_blocks[_subj_idx](
                        hidden,
                        adapter_self._text_embeds_cache,
                    )

                if _has_conv:
                    hidden = adapter_self.local_convs[_conv_idx](hidden, h, w)

                if _has_style and adapter_self.style_embeds is not None:
                    style_out = adapter_self.style_blocks[_style_idx](
                        hidden,
                        adapter_self.style_embeds,
                    )
                    hidden = hidden + adapter_self.style_scale[_style_idx] * style_out

                if is_tuple:
                    return (hidden,) + tuple(out[1:])

                return hidden

            block.forward = new_forward.__get__(block, type(block))

        self._patched = True
        logger.info("✅ DiT blocks 已 patch Adapter")

    def get_param_groups(self):
        param_groups = []
        base_lr = 2.0

        def add_group(module_or_params, name, d_coeff, weight_decay):
            if isinstance(module_or_params, nn.Module):
                params = list(module_or_params.parameters())
            else:
                params = list(module_or_params)

            params = [p for p in params if p.requires_grad]
            if len(params) == 0:
                return

            param_groups.append(
                {
                    "params": params,
                    "lr": base_lr,
                    "d_coeff": d_coeff,
                    "weight_decay": weight_decay,
                    "name": name,
                }
            )

        if self.enable_semantic_scale:
            add_group(self.semantic_scale, "semantic_scale", 0.7, 0.005)

        if self.enable_mod_text:
            add_group(self.mod_text, "mod_text", 0.6, 0.008)

        if self.enable_contrast_mod:
            add_group(self.contrast_mod, "contrast_mod", 0.65, 0.006)

        if self.enable_color_tune:
            add_group(self.color_tune, "color_tune", 0.65, 0.006)

        if self.enable_local_conv and self.local_conv_layers > 0:
            add_group(self.local_convs, "local_conv", 0.5, 0.01)

        if self.enable_edge_detail and self.edge_detail_layers > 0:
            add_group(self.edge_details, "edge_detail", 0.45, 0.01)

        if self.enable_subject_attn and self.subject_attn_layers > 0:
            add_group(self.subject_blocks, "subject_attention", 0.4, 0.012)

        if self.enable_style_attn and self.style_attn_layers > 0:
            style_params = list(self.style_blocks.parameters()) + [self.style_scale]
            add_group(style_params, "style_attention", 0.3, 0.015)

        logger.info(f"✅ 创建 {len(param_groups)} 个参数组，全部 lr={base_lr}，已适配 Prodigy")
        return param_groups


class ResolutionGroupedSampler(Sampler):
    """按 latent 分辨率分组采样器，兼容 Dataset 和 random_split 后的 Subset"""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resolution_groups = defaultdict(list)

        logger.info("🔍 预计算数据集分辨率分组...")

        for idx in range(len(dataset)):
            item_dataset, real_idx = self._resolve_dataset_index(dataset, idx)
            imgp, _ = item_dataset.sample_list[real_idx]
            h, w = item_dataset._get_target_latent_hw(imgp)
            self.resolution_groups[(h, w)].append(idx)

        logger.info("📊 数据集分辨率统计:")
        self.total_batches = 0

        total_samples = 0
        for (h, w), idxs in self.resolution_groups.items():
            cnt = len(idxs)
            total_samples += cnt
            batches = (cnt + batch_size - 1) // batch_size
            logger.info(
                f"  pixel={w * VAE_DOWNSAMPLE_FACTOR}x{h * VAE_DOWNSAMPLE_FACTOR} | latent={w}x{h}: {cnt}张 → {batches}批")
            self.total_batches += batches

        logger.info(f"  总计: {total_samples}张 → {self.total_batches}批")

    @staticmethod
    def _resolve_dataset_index(dataset, idx):
        """
        兼容 torch.utils.data.Subset。
        返回真正的 AnimaDataset 和真实样本 index。
        """
        if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            return dataset.dataset, dataset.indices[idx]
        return dataset, idx

    def __iter__(self):
        batches = []

        for idxs in self.resolution_groups.values():
            idxs = idxs.copy()
            if self.shuffle:
                np.random.shuffle(idxs)

            for i in range(0, len(idxs), self.batch_size):
                batches.append(idxs[i:i + self.batch_size])

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return self.total_batches


def collate_fn(batch):
    """
    支持两种模式：
    1. 预计算模式：batch 内有 latents，stack 成 [B,C,H,W]
    2. 在线 VAE 模式：batch 内无 latents，stack pixel_values 成 [B,3,H,W]
    """
    if not batch:
        return {}

    if tokenize_strategy is None:
        raise RuntimeError("❌ tokenize_strategy 尚未初始化，请确认 main() 中已先创建文本分词策略")

    has_latents = batch[0].get("latents") is not None
    has_pixels = batch[0].get("pixel_values") is not None

    latents = None
    pixel_values = None

    if has_latents:
        def _to_chw(lat: torch.Tensor) -> torch.Tensor:
            if lat.ndim == 5:
                lat = lat[0]
                if lat.ndim == 4 and lat.shape[1] == 1:
                    lat = lat[:, 0]
                elif lat.ndim == 4:
                    lat = lat.reshape(lat.shape[0], lat.shape[-2], lat.shape[-1])

            elif lat.ndim == 4:
                lat = lat[0]

            if lat.ndim != 3:
                raise RuntimeError(f"❌ latent 维度异常，期望 [C,H,W]，实际 shape={tuple(lat.shape)}")

            return lat.contiguous()

        latents_list = [_to_chw(item["latents"]) for item in batch]

        shapes = [lat.shape for lat in latents_list]
        max_h = max(s[-2] for s in shapes)
        max_w = max(s[-1] for s in shapes)

        if max_h % DIT_SPATIAL_PATCH_SIZE != 0:
            max_h = ((max_h + DIT_SPATIAL_PATCH_SIZE - 1) // DIT_SPATIAL_PATCH_SIZE) * DIT_SPATIAL_PATCH_SIZE

        if max_w % DIT_SPATIAL_PATCH_SIZE != 0:
            max_w = ((max_w + DIT_SPATIAL_PATCH_SIZE - 1) // DIT_SPATIAL_PATCH_SIZE) * DIT_SPATIAL_PATCH_SIZE

        padded = []
        for lat in latents_list:
            c, h, w = lat.shape
            pad_h = max_h - h
            pad_w = max_w - w

            if pad_h < 0 or pad_w < 0:
                raise RuntimeError(f"❌ latent padding 计算异常: current={h}x{w}, target={max_h}x{max_w}")

            if pad_h > 0 or pad_w > 0:
                lat = F.pad(lat, (0, pad_w, 0, pad_h), mode="constant", value=0)

            padded.append(lat)

        latents = torch.stack(padded, dim=0)
        out_h = max_h
        out_w = max_w


    elif has_pixels:
        pixels_list = [item["pixel_values"].contiguous() for item in batch]

        shapes = [p.shape for p in pixels_list]
        max_ph = max(s[-2] for s in shapes)
        max_pw = max(s[-1] for s in shapes)

        if max_ph % IMAGE_SIZE_MULTIPLE != 0:
            max_ph = ((max_ph + IMAGE_SIZE_MULTIPLE - 1) // IMAGE_SIZE_MULTIPLE) * IMAGE_SIZE_MULTIPLE

        if max_pw % IMAGE_SIZE_MULTIPLE != 0:
            max_pw = ((max_pw + IMAGE_SIZE_MULTIPLE - 1) // IMAGE_SIZE_MULTIPLE) * IMAGE_SIZE_MULTIPLE

        padded_pixels = []
        for pix in pixels_list:
            c, h, w = pix.shape
            pad_h = max_ph - h
            pad_w = max_pw - w

            if pad_h < 0 or pad_w < 0:
                raise RuntimeError(f"❌ pixel padding 计算异常: current={h}x{w}, target={max_ph}x{max_pw}")

            if pad_h > 0 or pad_w > 0:
                pix = F.pad(pix, (0, pad_w, 0, pad_h), mode="constant", value=0)

            padded_pixels.append(pix)

        pixel_values = torch.stack(padded_pixels, dim=0)

        out_h = max_ph // VAE_DOWNSAMPLE_FACTOR
        out_w = max_pw // VAE_DOWNSAMPLE_FACTOR

        if out_h % DIT_SPATIAL_PATCH_SIZE != 0 or out_w % DIT_SPATIAL_PATCH_SIZE != 0:
            raise RuntimeError(
                f"❌ 在线 VAE 模式得到非法 latent 尺寸: latent={out_w}x{out_h}"
            )

    else:
        raise RuntimeError("❌ batch 中既没有 latents，也没有 pixel_values")

    raw_caps = [i.get("raw_caption", "masterpiece") for i in batch]

    q_list, qm_list, t5_list, t5m_list = [], [], [], []
    safe_prompt = "masterpiece, best quality"

    for item in batch:
        cap = item.get("raw_caption", safe_prompt)
        if not str(cap).strip():
            cap = safe_prompt

        tokens = tokenize_strategy.tokenize(cap)

        if isinstance(tokens, dict):
            q_in = tokens["qwen3_input_ids"]
            qm = tokens["qwen3_attention_mask"]
            t5_in = tokens["t5_input_ids"]
            t5m = tokens["t5_attention_mask"]
        else:
            q_in, qm, t5_in, t5m = tokens

        q_in = q_in.squeeze(0) if q_in.ndim > 1 else q_in
        qm = qm.squeeze(0) if qm.ndim > 1 else qm
        t5_in = t5_in.squeeze(0) if t5_in.ndim > 1 else t5_in
        t5m = t5m.squeeze(0) if t5m.ndim > 1 else t5m

        q_list.append(q_in)
        qm_list.append(qm)
        t5_list.append(t5_in)
        t5m_list.append(t5m)

    qwen_ids = torch.stack(q_list, dim=0)
    qwen_mask = torch.stack(qm_list, dim=0)
    t5_ids = torch.stack(t5_list, dim=0)
    t5_mask = torch.stack(t5m_list, dim=0)

    if batch[0].get("style_embeds") is not None:
        style_embeds = torch.cat([i["style_embeds"] for i in batch], dim=0)
    else:
        style_embeds = torch.zeros(len(batch), 1, 768)

    return {
        "latents": latents,
        "pixel_values": pixel_values,

        "qwen_input_ids": qwen_ids,
        "qwen_attention_mask": qwen_mask,
        "t5_input_ids": t5_ids,
        "t5_attention_mask": t5_mask,

        "style_embeds": style_embeds,
        "timestep": torch.cat([i["timestep"] for i in batch], dim=0),

        "h": torch.tensor([out_h for _ in batch], dtype=torch.long),
        "w": torch.tensor([out_w for _ in batch], dtype=torch.long),

        "raw_caption": raw_caps,
        "caption_dropout_rate": torch.tensor(
            [i.get("caption_dropout_rate", 0.0) for i in batch],
            dtype=torch.float32,
        ),
    }


def normalize_vae_latents(lat: torch.Tensor) -> torch.Tensor:
    """
    统一 VAE 输出为 [B,C,H,W]
    常见输入：
    [B,C,H,W]
    [B,C,1,H,W]
    """
    if lat.ndim == 5:
        if lat.shape[2] == 1:
            lat = lat.squeeze(2)
        else:
            lat = lat[:, :, 0]

    if lat.ndim != 4:
        raise RuntimeError(f"❌ VAE 输出 latent 维度异常，期望 [B,C,H,W]，实际 shape={tuple(lat.shape)}")

    return lat.contiguous()


def encode_pixels_to_latents_online(
        vae_model,
        pixel_values: torch.Tensor,
        device,
        weight_dtype,
        vae_encode_batch_size: int = 1,
) -> torch.Tensor:
    """
    在线 GPU VAE 编码。
    注意：
    - pixel_values 来自 DataLoader，形状 [B,3,H,W]，CPU tensor
    - 这里在主进程里分小批搬到 GPU 编码，避免 DataLoader worker 碰 CUDA
    """
    vae_encode_batch_size = max(1, int(vae_encode_batch_size))

    latents_list = []

    vae_model.eval()

    with torch.no_grad():
        for i in range(0, pixel_values.shape[0], vae_encode_batch_size):
            pix = pixel_values[i:i + vae_encode_batch_size]
            pix = pix.to(device=device, dtype=torch.float32, non_blocking=True)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=False):
                    lat = vae_model.encode_pixels_to_latents(pix.float())
            else:
                lat = vae_model.encode_pixels_to_latents(pix.float())

            lat = normalize_vae_latents(lat)
            lat = lat.to(device=device, dtype=weight_dtype)

            latents_list.append(lat)

            del pix, lat

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    latents = torch.cat(latents_list, dim=0).contiguous()

    return latents


class AnimaDataset(Dataset):
    """兼容 Kohya 目录格式 | 安全 VAE 预计算 | 自动 resize 到 DiT 合法尺寸 | 支持 CPU 预计算防止 CUDA VAE 爆显存"""

    def __init__(
            self,
            data_dir: str,
            max_width: int = 1024,
            max_height: int = 1024,
            vae_model=None,
            clip_model=None,
            clip_processor=None,
            device="cuda",
            precompute_embeddings=True,
            caption_dropout_rate=0.1,
            shuffle_caption=True,
            timestep_sampling="sigmoid",
            qwen3_max_token_length=77,
            t5_max_token_length=77,
            vae_precompute_device: str = "auto",
            skip_failed_precompute: bool = False,
            style_dim: int = 768,
            vae_precompute_batch_size: int = 1,
            clip_precompute_batch_size: int = 8,
    ):
        super().__init__()

        self.data_dir = os.path.abspath(data_dir)
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.vae_model = vae_model
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device
        self.precompute_embeddings = bool(precompute_embeddings)
        self.caption_dropout_rate = caption_dropout_rate
        self.shuffle_caption = shuffle_caption
        self.timestep_sampling = timestep_sampling
        self.qwen3_max_token_length = qwen3_max_token_length
        self.t5_max_token_length = t5_max_token_length
        self.vae_precompute_device = vae_precompute_device
        self.skip_failed_precompute = bool(skip_failed_precompute)
        self.style_dim = int(style_dim)
        self.vae_precompute_batch_size = max(1, int(vae_precompute_batch_size))
        self.clip_precompute_batch_size = max(1, int(clip_precompute_batch_size))

        self.sample_list = []
        self._image_size_cache = {}
        self._target_pixel_cache = {}
        self._target_latent_cache = {}

        if self.vae_model is not None:
            self.vae_model.float().eval().requires_grad_(False)

        if self.clip_model is not None:
            self.clip_model.eval().requires_grad_(False)

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"❌ 数据集目录不存在: {self.data_dir}")

        logger.info(f"📂 正在扫描数据集目录: {self.data_dir}")
        logger.info(f"📋 目录下所有文件/文件夹: {os.listdir(self.data_dir)}")

        for subdir in os.listdir(self.data_dir):
            if subdir.startswith("."):
                logger.info(f"🔒 跳过隐藏项: '{subdir}'")
                continue

            subpath = os.path.join(self.data_dir, subdir)
            logger.info(f"🔍 检查子项: '{subdir}' → 完整路径: {subpath}")

            if not os.path.isdir(subpath):
                logger.info("  ❌ 不是有效文件夹，跳过")
                continue

            repeat = 1
            if "_" in subdir and subdir.split("_")[0].isdigit():
                repeat = int(subdir.split("_")[0])
                logger.info(f"  📊 图片重复次数: {repeat}")

            try:
                files = os.listdir(subpath)
            except Exception as e:
                raise RuntimeError(f"❌ 无法读取文件夹内容: {subpath}, 错误: {str(e)}") from e

            img_count = 0

            for fn in files:
                if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) or fn.startswith("style_"):
                    continue

                imgp = os.path.join(subpath, fn)
                txtp = os.path.join(subpath, os.path.splitext(fn)[0] + ".txt")

                if not os.path.exists(imgp):
                    logger.warning(f"  ⚠️ 图片文件不存在: {imgp}，跳过")
                    continue

                cap = "masterpiece, best quality"

                if os.path.exists(txtp):
                    try:
                        with open(txtp, "r", encoding="utf-8") as f:
                            cap = f.read().strip()
                    except Exception as e:
                        logger.warning(f"  ⚠️ 无法读取 caption: {txtp}, 错误: {str(e)}，使用默认文案")
                else:
                    logger.warning(f"  ⚠️ 未找到 caption 文件: {txtp}，使用默认文案")

                for _ in range(repeat):
                    self.sample_list.append((imgp, cap))

                img_count += 1

            logger.info(f"  ✅ 找到 {img_count} 张有效图片")

        if len(self.sample_list) == 0:
            raise RuntimeError("❌ 数据集中没有找到任何有效图片，请检查目录结构和文件格式")

        logger.info(f"✅ 数据集最终加载完成: 共 {len(self.sample_list)} 个训练样本")

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

        if self.precompute_embeddings:
            if self.vae_model is None:
                raise RuntimeError("❌ 已启用预计算，但 vae_model 为 None")
            self._precompute_cache()
        else:
            logger.warning(
                "⚠️ 已关闭 --enable_precompute_embeddings。"
                "当前训练循环仍依赖 latent 缓存，因此只有在所有 *_latent_宽x高.npy 已经存在时才能继续训练。"
            )

    def _safe_load_image(self, img_path: str) -> Image.Image:
        """安全加载图片，支持透明通道和 EXIF 修正"""
        try:
            img = Image.open(img_path)

            try:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass

            if img.mode in ["RGBA", "LA"] or (img.mode == "P" and "transparency" in img.info):
                background = Image.new("RGB", img.size, (255, 255, 255))
                alpha = img.split()[-1]
                background.paste(img, mask=alpha)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            return img

        except Exception as e1:
            logger.warning(f"⚠️ 标准加载失败 {img_path}: {e1}，尝试二进制读取")

            try:
                from io import BytesIO

                with open(img_path, "rb") as f:
                    data = f.read()

                img = Image.open(BytesIO(data))

                try:
                    from PIL import ImageOps
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass

                if img.mode in ["RGBA", "LA"] or (img.mode == "P" and "transparency" in img.info):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    alpha = img.split()[-1]
                    background.paste(img, mask=alpha)
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                return img

            except Exception as e2:
                raise RuntimeError(f"❌ 图片彻底损坏，无法加载: {img_path}, 错误: {str(e2)}") from e2

    def _sample_timestep(self, bs=1):
        if self.timestep_sampling == "uniform":
            return torch.randint(0, 1000, (bs,)).long()

        u = torch.rand((bs,))
        return (1000 * torch.sigmoid(10 * (u - 0.5))).long()

    def _read_image_size_fast(self, img_path: str) -> tuple[int, int]:
        if img_path in self._image_size_cache:
            return self._image_size_cache[img_path]

        try:
            img = Image.open(img_path)
            ow, oh = img.size

            try:
                exif = img.getexif()
                orientation = exif.get(274, None)
                if orientation in [5, 6, 7, 8]:
                    ow, oh = oh, ow
            except Exception:
                pass

            img.close()

        except Exception:
            img = self._safe_load_image(img_path)
            ow, oh = img.size

        if ow <= 0 or oh <= 0:
            raise RuntimeError(f"❌ 图片尺寸异常: {img_path}, size={ow}x{oh}")

        self._image_size_cache[img_path] = (ow, oh)
        return ow, oh

    def _get_target_pixel_size(self, img_path: str) -> tuple[int, int]:
        if img_path in self._target_pixel_cache:
            return self._target_pixel_cache[img_path]

        ow, oh = self._read_image_size_fast(img_path)

        scale = min(self.max_width / ow, self.max_height / oh)
        scale = min(scale, 1.0)

        nw = int(round(ow * scale))
        nh = int(round(oh * scale))

        nw = max((nw // IMAGE_SIZE_MULTIPLE) * IMAGE_SIZE_MULTIPLE, IMAGE_SIZE_MULTIPLE)
        nh = max((nh // IMAGE_SIZE_MULTIPLE) * IMAGE_SIZE_MULTIPLE, IMAGE_SIZE_MULTIPLE)

        nw = min(nw, max((self.max_width // IMAGE_SIZE_MULTIPLE) * IMAGE_SIZE_MULTIPLE, IMAGE_SIZE_MULTIPLE))
        nh = min(nh, max((self.max_height // IMAGE_SIZE_MULTIPLE) * IMAGE_SIZE_MULTIPLE, IMAGE_SIZE_MULTIPLE))

        nw = max(nw, IMAGE_SIZE_MULTIPLE)
        nh = max(nh, IMAGE_SIZE_MULTIPLE)

        self._target_pixel_cache[img_path] = (nw, nh)
        return nw, nh

    def _get_target_latent_hw(self, img_path: str) -> tuple[int, int]:
        if img_path in self._target_latent_cache:
            return self._target_latent_cache[img_path]

        nw, nh = self._get_target_pixel_size(img_path)

        h = nh // VAE_DOWNSAMPLE_FACTOR
        w = nw // VAE_DOWNSAMPLE_FACTOR

        if h % DIT_SPATIAL_PATCH_SIZE != 0 or w % DIT_SPATIAL_PATCH_SIZE != 0:
            raise RuntimeError(
                f"❌ 内部尺寸计算错误: pixel={nw}x{nh}, latent={w}x{h}, "
                f"latent H/W 必须能被 {DIT_SPATIAL_PATCH_SIZE} 整除"
            )

        self._target_latent_cache[img_path] = (h, w)
        return h, w

    def _load_resized_image(self, img_path: str) -> Image.Image:
        img = self._safe_load_image(img_path)
        nw, nh = self._get_target_pixel_size(img_path)

        if img.size != (nw, nh):
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)

        return img

    def _cache_paths(self, img_path: str):
        """缓存文件名带 resize 后尺寸，避免继续读取旧尺寸 latent"""
        bn = os.path.splitext(img_path)[0]
        nw, nh = self._get_target_pixel_size(img_path)

        latent_path = f"{bn}_latent_{nw}x{nh}.npy"
        style_path = f"{bn}_style_{nw}x{nh}.npy"

        return latent_path, style_path

    def _select_vae_precompute_device(self) -> torch.device:
        """
        预计算设备选择。
        A800 80GB 场景下，不建议高分辨率 auto 自动回 CPU，因为 CPU 内存反而更容易爆。
        """
        if self.vae_precompute_device == "cpu":
            logger.warning(
                "⚠️ 当前指定 --vae_precompute_device=cpu。"
                "如果你的 CPU 内存小于显存，建议改为 --vae_precompute_device=cuda。"
            )
            return torch.device("cpu")

        if self.vae_precompute_device in ["cuda", "auto"]:
            if torch.cuda.is_available():
                return torch.device(self.device)

            logger.warning("⚠️ CUDA 不可用，VAE 预计算只能回退 CPU")
            return torch.device("cpu")

        return torch.device(self.device if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _safe_cuda_cleanup(force: bool = False):
        gc.collect()

        if not torch.cuda.is_available():
            return

        if force:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @staticmethod
    def _atomic_save_npy(path: str, array: np.ndarray):
        tmp_path = path + ".tmp.npy"
        np.save(tmp_path, array)
        os.replace(tmp_path, path)

    def _encode_vae_latent_batch(
            self,
            imt_batch: torch.Tensor,
            img_paths: list[str],
            encode_device: torch.device,
    ) -> torch.Tensor:
        """
        批量 VAE encode。
        输入:
            imt_batch: [B,3,H,W]，CPU float32，范围 [-1,1]
        输出:
            latents: [B,C,H,W]，CPU float32

        注意:
        - 不自动回 CPU，避免 CPU 内存爆。
        - 如果 OOM，请降低 --vae_precompute_batch_size。
        """
        if self.vae_model is None:
            raise RuntimeError("❌ vae_model 为 None，无法预计算 latent")

        try:
            self.vae_model.to(encode_device)
            self.vae_model.float().eval().requires_grad_(False)

            imt_batch = imt_batch.to(
                encode_device,
                dtype=torch.float32,
                non_blocking=True,
            )

            with torch.inference_mode():
                if encode_device.type == "cuda":
                    with torch.autocast(device_type="cuda", enabled=False):
                        lat = self.vae_model.encode_pixels_to_latents(imt_batch.float())
                else:
                    lat = self.vae_model.encode_pixels_to_latents(imt_batch.float())

            lat = normalize_vae_latents(lat)

            if not torch.isfinite(lat).all():
                raise RuntimeError("VAE 输出 latent 出现 NaN 或 Inf")

            lat = lat.detach().cpu().float()

            del imt_batch

            return lat

        except RuntimeError as err:
            err_text = str(err)
            joined_paths = "\n".join(img_paths[:8])

            if "out of memory" in err_text.lower() or "cuda" in err_text.lower():
                self._safe_cuda_cleanup(force=True)
                raise RuntimeError(
                    f"❌ 批量 VAE 预计算失败，可能是显存不足。\n"
                    f"当前 batch_size={self.vae_precompute_batch_size}，device={encode_device}。\n"
                    f"建议降低 --vae_precompute_batch_size，例如 1 或 2。\n"
                    f"本批部分图片:\n{joined_paths}\n"
                    f"原始错误: {err_text}"
                ) from err

            raise

    def _encode_clip_style_batch(
            self,
            imgs: list[Image.Image],
            img_paths: list[str],
    ) -> torch.Tensor:
        """
        批量 CLIP image feature encode。
        输出 [B,1,style_dim]，CPU float32。
        """
        batch_size = len(imgs)

        if self.clip_model is None or self.clip_processor is None:
            return torch.zeros(batch_size, 1, self.style_dim, dtype=torch.float32)

        clip_device = next(self.clip_model.parameters()).device

        try:
            with torch.inference_mode():
                clip_in = self.clip_processor(images=imgs, return_tensors="pt")
                clip_in = clip_in.to(clip_device)

                if clip_device.type == "cuda":
                    with torch.autocast(device_type="cuda", enabled=False):
                        sty = self.clip_model.get_image_features(**clip_in).unsqueeze(1)
                else:
                    sty = self.clip_model.get_image_features(**clip_in).unsqueeze(1)

            if not torch.isfinite(sty).all():
                raise RuntimeError("CLIP style embedding 出现 NaN 或 Inf")

            sty = sty.detach().cpu().float()

            if sty.shape[-1] != self.style_dim:
                raise RuntimeError(
                    f"❌ CLIP style_dim 不匹配: 实际={sty.shape[-1]}, 配置={self.style_dim}。"
                    f"ViT-L/14 通常应为 768。"
                )

            del clip_in

            return sty

        except RuntimeError as err:
            err_text = str(err)
            joined_paths = "\n".join(img_paths[:8])
            self._safe_cuda_cleanup(force=True)

            raise RuntimeError(
                f"❌ 批量 CLIP 风格预计算失败。\n"
                f"本批部分图片:\n{joined_paths}\n"
                f"原始错误: {err_text}"
            ) from err

    def _encode_vae_latent(self, imt: torch.Tensor, img_path: str, preferred_device: torch.device) -> torch.Tensor:
        """
        优先按指定设备编码。
        CUDA 出现 GET/cudnn/OOM 时自动回退 CPU。
        CUDA illegal memory access 不尝试恢复，直接抛出，要求重启进程。
        """
        devices_to_try = [preferred_device]

        if preferred_device.type == "cuda":
            devices_to_try.append(torch.device("cpu"))

        last_error = None

        for encode_device in devices_to_try:
            try:
                self._safe_cuda_cleanup()

                logger.debug(f"VAE encode device={encode_device}: {img_path}")

                self.vae_model.to(encode_device)
                self.vae_model.float().eval()

                imt_on_device = imt.to(encode_device, dtype=torch.float32, non_blocking=False)

                with torch.inference_mode():
                    if encode_device.type == "cuda":
                        with torch.autocast(device_type="cuda", enabled=False):
                            lat = self.vae_model.encode_pixels_to_latents(imt_on_device.float())
                    else:
                        lat = self.vae_model.encode_pixels_to_latents(imt_on_device.float())

                if not torch.isfinite(lat).all():
                    raise RuntimeError("VAE 输出 latent 出现 NaN 或 Inf")

                lat = lat.detach().cpu().float()

                del imt_on_device
                self._safe_cuda_cleanup()

                return lat

            except RuntimeError as err:
                err_text = str(err)
                last_error = err

                logger.error(f"{img_path}: VAE encode failed on {encode_device}: {err_text}")

                if "illegal memory access" in err_text.lower():
                    raise

                if encode_device.type == "cuda":
                    logger.warning(
                        f"⚠️ CUDA VAE 编码失败，自动回退 CPU 重新编码该图: {img_path}"
                    )
                    self._safe_cuda_cleanup()
                    continue

                raise

        raise last_error

    def _encode_clip_style(self, img: Image.Image, img_path: str) -> torch.Tensor:
        if self.clip_model is None or self.clip_processor is None:
            return torch.zeros(1, 1, 1024, dtype=torch.float32)

        clip_device = next(self.clip_model.parameters()).device

        try:
            with torch.inference_mode():
                clip_in = self.clip_processor(images=img, return_tensors="pt")
                clip_in = clip_in.to(clip_device)

                if clip_device.type == "cuda":
                    with torch.autocast(device_type="cuda", enabled=False):
                        sty = self.clip_model.get_image_features(**clip_in).unsqueeze(1)
                else:
                    sty = self.clip_model.get_image_features(**clip_in).unsqueeze(1)

            if not torch.isfinite(sty).all():
                raise RuntimeError("CLIP style embedding 出现 NaN 或 Inf")

            sty = sty.detach().cpu().float()

            del clip_in
            self._safe_cuda_cleanup()

            return sty

        except RuntimeError as err:
            err_text = str(err)
            logger.error(f"{img_path}: CLIP encode failed: {err_text}")

            if "illegal memory access" in err_text.lower():
                raise

            raise

    def _precompute_cache(self):
        preferred_vae_device = self._select_vae_precompute_device()

        logger.info(
            f"🔄 VAE/CLIP 批量预计算缓存 | "
            f"vae_precompute_device={preferred_vae_device} | "
            f"vae_batch={self.vae_precompute_batch_size} | "
            f"clip_batch={self.clip_precompute_batch_size}"
        )

        unique_imgs = list({p for p, _ in self.sample_list})

        need_calc = []
        for p in unique_imgs:
            latp, styp = self._cache_paths(p)

            need_lat = not os.path.exists(latp)
            need_style = self.clip_model is not None and not os.path.exists(styp)

            if need_lat or need_style:
                need_calc.append(p)

        logger.info(
            f"📊 待计算图片: {len(need_calc)} | "
            f"已有合法尺寸缓存跳过: {len(unique_imgs) - len(need_calc)}"
        )

        if len(need_calc) == 0:
            logger.info("✅ 全部合法尺寸缓存已存在，无需预计算")
            return

        if self.vae_model is not None:
            self.vae_model.eval().float().requires_grad_(False)

        if self.clip_model is not None:
            self.clip_model.eval().requires_grad_(False)

        groups = defaultdict(list)
        for img_path in need_calc:
            nw, nh = self._get_target_pixel_size(img_path)
            groups[(nw, nh)].append(img_path)

        logger.info("📊 预计算分辨率分组:")
        for (nw, nh), paths in sorted(groups.items(), key=lambda x: x[0][0] * x[0][1]):
            logger.info(f"  pixel={nw}x{nh}: {len(paths)} 张")

        failed_imgs = []

        try:
            if preferred_vae_device.type == "cuda":
                torch.cuda.empty_cache()

            for (nw, nh), paths in tqdm(
                    list(groups.items()),
                    desc="按分辨率批量预计算",
            ):
                logger.info(f"🔄 处理分辨率组 pixel={nw}x{nh} | {len(paths)} 张")

                vae_need = []
                for img_path in paths:
                    latp, _ = self._cache_paths(img_path)
                    if not os.path.exists(latp):
                        vae_need.append(img_path)

                for start in tqdm(
                        range(0, len(vae_need), self.vae_precompute_batch_size),
                        desc=f"VAE {nw}x{nh}",
                        leave=False,
                ):
                    batch_paths = vae_need[start:start + self.vae_precompute_batch_size]

                    try:
                        imgs = [self._load_resized_image(p) for p in batch_paths]

                        for img in imgs:
                            if img.size != (nw, nh):
                                raise RuntimeError(
                                    f"resize 后尺寸不一致: 期望 {nw}x{nh}, 实际 {img.size}"
                                )

                        imt_batch = torch.stack(
                            [self.transform(img).contiguous() for img in imgs],
                            dim=0,
                        )

                        lat_batch = self._encode_vae_latent_batch(
                            imt_batch=imt_batch,
                            img_paths=batch_paths,
                            encode_device=preferred_vae_device,
                        )

                        if lat_batch.shape[0] != len(batch_paths):
                            raise RuntimeError(
                                f"VAE batch 输出数量异常: out={lat_batch.shape[0]}, in={len(batch_paths)}"
                            )

                        for img_path, lat in zip(batch_paths, lat_batch):
                            latp, _ = self._cache_paths(img_path)
                            self._atomic_save_npy(latp, lat.numpy())

                        del imgs, imt_batch, lat_batch

                    except Exception as err:
                        err_text = str(err)
                        logger.error(f"❌ VAE 批量预计算失败: {err_text}")

                        for img_path in batch_paths:
                            failed_imgs.append((img_path, err_text))

                        self._safe_cuda_cleanup(force=True)

                        if not self.skip_failed_precompute:
                            raise

                if self.clip_model is not None:
                    style_need = []
                    for img_path in paths:
                        _, styp = self._cache_paths(img_path)
                        if not os.path.exists(styp):
                            style_need.append(img_path)

                    for start in tqdm(
                            range(0, len(style_need), self.clip_precompute_batch_size),
                            desc=f"CLIP {nw}x{nh}",
                            leave=False,
                    ):
                        batch_paths = style_need[start:start + self.clip_precompute_batch_size]

                        try:
                            imgs = [self._load_resized_image(p) for p in batch_paths]

                            sty_batch = self._encode_clip_style_batch(
                                imgs=imgs,
                                img_paths=batch_paths,
                            )

                            if sty_batch.shape[0] != len(batch_paths):
                                raise RuntimeError(
                                    f"CLIP batch 输出数量异常: out={sty_batch.shape[0]}, in={len(batch_paths)}"
                                )

                            for img_path, sty in zip(batch_paths, sty_batch):
                                _, styp = self._cache_paths(img_path)
                                self._atomic_save_npy(styp, sty.numpy())

                            del imgs, sty_batch

                        except Exception as err:
                            err_text = str(err)
                            logger.error(f"❌ CLIP 批量预计算失败: {err_text}")

                            for img_path in batch_paths:
                                failed_imgs.append((img_path, err_text))

                            self._safe_cuda_cleanup(force=True)

                            if not self.skip_failed_precompute:
                                raise

                self._safe_cuda_cleanup(force=False)

        finally:
            self._safe_cuda_cleanup(force=True)

        if failed_imgs:
            logger.error(f"❌ 预计算失败 {len(failed_imgs)} 张:")
            for p, e in failed_imgs[:50]:
                logger.error(f"{p}: {e}")

            raise RuntimeError(
                "存在图片预处理失败。"
                "如果是 CUDA OOM，请降低 --vae_precompute_batch_size 或 --clip_precompute_batch_size。"
            )

        logger.info("✅ 批量缓存预计算全部完成")

    def _resize_align_16(self, imgp):
        """保留旧接口，供 sampler 调用。实际返回合法 latent H/W。"""
        return self._get_target_latent_hw(imgp)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        idx = idx % len(self.sample_list)

        imgp, cap = self.sample_list[idx]

        h, w = self._get_target_latent_hw(imgp)
        latp, styp = self._cache_paths(imgp)

        is_drop = np.random.rand() < self.caption_dropout_rate

        if self.precompute_embeddings:
            if not os.path.exists(latp):
                raise RuntimeError(
                    f"❌ 合法尺寸 latent 缓存不存在: {latp}\n"
                    f"当前启用了 --enable_precompute_embeddings，但缓存不存在。"
                )

            lat = torch.from_numpy(np.load(latp)).float()

            while lat.ndim > 3:
                lat = lat[0]

            if lat.ndim == 4:
                if lat.shape[1] == 1:
                    lat = lat[:, 0]
                else:
                    lat = lat.reshape(lat.shape[0], lat.shape[-2], lat.shape[-1])

            if lat.ndim != 3:
                raise RuntimeError(f"❌ latent 缓存维度异常: {latp}, shape={tuple(lat.shape)}")

            real_h, real_w = lat.shape[-2], lat.shape[-1]

            if real_h % DIT_SPATIAL_PATCH_SIZE != 0 or real_w % DIT_SPATIAL_PATCH_SIZE != 0:
                raise RuntimeError(
                    f"❌ latent 缓存尺寸非法: {latp}, latent={real_w}x{real_h}，"
                    f"请删除旧 latent 缓存并重新预计算。"
                )

            h, w = real_h, real_w

            if self.clip_model is not None:
                if not os.path.exists(styp):
                    raise RuntimeError(f"❌ 风格嵌入缓存不存在: {styp}")
                sty = torch.from_numpy(np.load(styp)).float()
            else:
                sty = torch.zeros(1, 1, self.style_dim, dtype=torch.float32)

            return {
                "latents": lat,
                "pixel_values": None,
                "style_embeds": sty,
                "timestep": self._sample_timestep(1),
                "h": h,
                "w": w,
                "raw_caption": cap,
                "caption_dropout_rate": float(is_drop),
            }

        img = self._load_resized_image(imgp)
        pixel_values = self.transform(img).contiguous()

        sty = torch.zeros(1, 1, self.style_dim, dtype=torch.float32)

        return {
            "latents": None,
            "pixel_values": pixel_values,
            "style_embeds": sty,
            "timestep": self._sample_timestep(1),
            "h": h,
            "w": w,
            "raw_caption": cap,
            "caption_dropout_rate": float(is_drop),
        }


def main(args):
    global tokenize_strategy, text_encoding_strategy

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    set_seed(args.seed)
    device = accelerator.device

    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    logger.info("🔄 加载模型...")

    logger.info("🔄 初始化官方文本处理策略...")
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.qwen_model_path,
        t5_tokenizer_path=args.t5_tokenizer_path,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length
    )
    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)
    logger.info("✅ 官方文本策略初始化完成（T5分词器已从DiT内部提取）")

    logger.info(f"🔄 加载Qwen3: {args.qwen_model_path}")
    qwen_model, _ = anima_utils.load_qwen3_text_encoder(
        args.qwen_model_path, dtype=weight_dtype, device=device
    )
    qwen_model.requires_grad_(False).eval()
    logger.info("✅ Qwen3加载完成（权重已冻结）")

    clip_model, clip_processor = None, None
    if args.enable_style_attn:
        from transformers import CLIPModel, CLIPImageProcessor
        logger.info(f"🔄 加载CLIP: {args.clip_model_path}")

        try:
            # 先尝试本地加载
            clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model_path, local_files_only=True)
            clip_model = CLIPModel.from_pretrained(
                args.clip_model_path, torch_dtype=weight_dtype, local_files_only=True
            )
        except (OSError, ValueError):
            # 本地没有，自动从清华源下载完整模型
            logger.info("本地未找到CLIP，正在从清华镜像源自动下载...")
            clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model_path)
            clip_model = CLIPModel.from_pretrained(
                args.clip_model_path, torch_dtype=weight_dtype
            )

        clip_model = clip_model.to(device).requires_grad_(False).eval()
        logger.info("✅ CLIP加载完成（已自动使用清华镜像源，权重已冻结）")

    logger.info(f"🔄 加载VAE: {args.vae_model_path}")
    vae_model = AutoencoderKLQwenImage(
        spatial_chunk_size=args.vae_chunk_size
    ).to(device).float()
    vae_sd = load_file(args.vae_model_path)
    vae_model.load_state_dict(vae_sd, strict=False)
    vae_model.requires_grad_(False).eval()
    logger.info("✅ VAE加载完成（权重已冻结）")

    logger.info("🔄 加载Anima DiT...")
    attn_mode = "xformers" if args.xformers else "torch"
    dit_model = anima_utils.load_anima_model(
        device, args.anima_model_path, attn_mode=attn_mode,
        split_attn=args.split_attn, loading_device=device,
        dit_weight_dtype=weight_dtype
    ).requires_grad_(False).eval()
    logger.info("✅ DiT加载完成（权重已冻结，和LoRA训练完全一致）")

    logger.info("🔧 初始化Adapter...")
    adapter = AnimaDiTAdapter(
        dit_hidden_size=args.dit_hidden_size,
        num_blocks=args.num_blocks,
        text_embed_dim=args.text_embed_dim,
        style_dim=args.style_dim,
        patch_size=DIT_SPATIAL_PATCH_SIZE,
        gradient_checkpointing=args.gradient_checkpointing,

        enable_mod_text=args.enable_mod_text,
        enable_semantic_scale=args.enable_semantic_scale,
        enable_local_conv=args.enable_local_conv,
        local_conv_layers=args.local_conv_layers,
        local_conv_kernel=args.local_conv_kernel,
        local_conv_start_layer=args.local_conv_start_layer,

        enable_style_attn=args.enable_style_attn,
        style_attn_layers=args.style_attn_layers,
        style_num_heads=args.style_num_heads,
        style_attn_start_layer=args.style_attn_start_layer,

        enable_edge_detail=args.enable_edge_detail,
        edge_detail_layers=args.edge_detail_layers,
        edge_detail_kernel=args.edge_detail_kernel,
        edge_detail_start_layer=args.edge_detail_start_layer,

        enable_subject_attn=args.enable_subject_attn,
        subject_attn_layers=args.subject_attn_layers,
        subject_attn_heads=args.subject_attn_heads,
        subject_attn_start_layer=args.subject_attn_start_layer,

        enable_contrast_mod=args.enable_contrast_mod,
        enable_color_tune=args.enable_color_tune,
    )

    if args.gradient_checkpointing:
        logger.info("✅ 已开启 gradient checkpointing：用运行速度换显存")

        if hasattr(dit_model, "gradient_checkpointing_enable"):
            try:
                dit_model.gradient_checkpointing_enable()
                logger.info("✅ dit_model.gradient_checkpointing_enable() 调用成功")
            except Exception as e:
                logger.warning(
                    f"⚠️ dit_model.gradient_checkpointing_enable() 调用失败，将使用 Adapter block checkpoint: {e}")

        adapter.set_gradient_checkpointing(True)

    if args.resume:
        logger.info(f"🔄 从检查点恢复: {args.resume}")
        adapter.load_state_dict(load_file(args.resume), strict=False)
    adapter.patch_dit_blocks(dit_model)
    logger.info("✅ Adapter初始化完成")

    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=args.discrete_flow_shift
    )

    param_groups = adapter.get_param_groups()
    logger.info(f"✅ 创建{len(param_groups)}个参数组")
    optimizer = Prodigy(
        param_groups, betas=(0.9, 0.999), use_bias_correction=True,
        safeguard_warmup=True, decouple=True
    )

    logger.info("📂 加载数据集...")
    full_ds = AnimaDataset(
        data_dir=args.train_data_dir,
        max_width=args.max_width,
        max_height=args.max_height,
        vae_model=vae_model if args.enable_precompute_embeddings else None,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
        precompute_embeddings=args.enable_precompute_embeddings,
        caption_dropout_rate=args.caption_dropout_rate,
        shuffle_caption=args.shuffle_caption,
        timestep_sampling=args.timestep_sampling,
        qwen3_max_token_length=args.qwen3_max_token_length,
        t5_max_token_length=args.t5_max_token_length,
        vae_precompute_device=args.vae_precompute_device,
        skip_failed_precompute=args.skip_failed_precompute,
        style_dim=args.style_dim,
        vae_precompute_batch_size=args.vae_precompute_batch_size,
        clip_precompute_batch_size=args.clip_precompute_batch_size,
    )
    train_ds = full_ds
    val_ds = None
    if args.val_split > 0 and len(full_ds) > 10:
        val_size = int(len(full_ds) * args.val_split)
        train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])
        logger.info(f"✅ 数据集分割: 训练{len(train_ds)} | 验证{val_size}")

    train_sampler = ResolutionGroupedSampler(train_ds, args.train_batch_size)
    train_dl = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=args.max_data_loader_n_workers,
        pin_memory=True, collate_fn=collate_fn
    )
    val_dl = None
    if val_ds:
        val_sampler = ResolutionGroupedSampler(val_ds, args.train_batch_size, shuffle=False)
        val_dl = DataLoader(
            val_ds, batch_sampler=val_sampler,
            num_workers=args.max_data_loader_n_workers,
            pin_memory=True, collate_fn=collate_fn
        )

    steps_per_epoch = max(1, (len(train_dl) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    total_steps = max(1, args.max_train_epochs * steps_per_epoch)
    logger.info(f"📊 每轮优化步数: {steps_per_epoch} | 总训练步数: {total_steps}")

    warmup_steps = max(0, min(args.lr_warmup_steps, total_steps - 1))

    if args.use_cosine_scheduler:
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

        if warmup_steps > 0:
            warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0)
            scheduler = SequentialLR(optimizer, [warmup, cosine], [warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=0)
    else:
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max(step / warmup_steps, 1e-6)

            decay_start = int(0.9 * total_steps)
            if step < decay_start:
                return 1.0

            decay_steps = max(1, total_steps - decay_start)
            return max(0.0, 1.0 - (step - decay_start) / decay_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    dit_model, adapter, optimizer, train_dl, scheduler = accelerator.prepare(
        dit_model, adapter, optimizer, train_dl, scheduler
    )
    qwen_model = accelerator.prepare(qwen_model)
    if val_dl: val_dl = accelerator.prepare(val_dl)

    loss_window = deque(maxlen=100)
    global_step = 0
    best_val_loss = float('inf')
    pb = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.max_train_epochs):
        adapter.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(adapter):
                if batch["latents"] is not None:
                    latents = batch["latents"].to(device, dtype=weight_dtype, non_blocking=True)
                else:
                    pixel_values = batch["pixel_values"]
                    latents = encode_pixels_to_latents_online(
                        vae_model=vae_model,
                        pixel_values=pixel_values,
                        device=device,
                        weight_dtype=weight_dtype,
                        vae_encode_batch_size=args.vae_encode_batch_size,
                    )
                    del pixel_values
                qwen_ids = batch["qwen_input_ids"].to(device)
                qwen_mask = batch["qwen_attention_mask"].to(device)
                t5_ids = batch["t5_input_ids"].to(device)
                t5_mask = batch["t5_attention_mask"].to(device)
                style_emb = batch["style_embeds"].to(device, weight_dtype)
                timesteps = batch["timestep"].to(device)
                h, w = batch["h"][0].item(), batch["w"][0].item()

                with torch.no_grad(), accelerator.autocast():
                    tokens = [qwen_ids, qwen_mask, t5_ids, t5_mask]
                    te_out = text_encoding_strategy.encode_tokens(tokenize_strategy, [qwen_model], tokens)
                    prompt_emb, attn_mask = te_out[0], te_out[1]

                drop_rates = batch["caption_dropout_rate"].to(device)
                prompt_emb, attn_mask, _, _ = text_encoding_strategy.drop_cached_text_encoder_outputs(
                    prompt_emb, attn_mask, None, None, caption_dropout_rates=drop_rates
                )
                unwrapped_adapter = accelerator.unwrap_model(adapter)

                prompt_emb = unwrapped_adapter.prepare_text_condition(prompt_emb)

                unwrapped_adapter.set_current_resolution(h, w)
                unwrapped_adapter.set_text_embeds_cache(prompt_emb)
                unwrapped_adapter.set_style_embeds(style_emb)

                if latents.ndim == 5:
                    latents = latents.squeeze(2)

                noise = torch.randn_like(latents)

                noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
                    args, noise_scheduler, latents, noise, device, weight_dtype
                )
                timesteps = timesteps / 1000.0

                bs = latents.shape[0]
                h_latent = latents.shape[-2]
                w_latent = latents.shape[-1]
                pad_mask = torch.zeros(
                    bs, 1, h_latent, w_latent,
                    dtype=weight_dtype,
                    device=device
                )

                noisy_model_input = noisy_model_input.unsqueeze(2)

                with accelerator.autocast():
                    pred = dit_model(
                        noisy_model_input, timesteps, prompt_emb, padding_mask=pad_mask,
                        target_input_ids=t5_ids, target_attention_mask=t5_mask,
                        source_attention_mask=attn_mask
                    )
                    pred = pred.squeeze(2)
                    target = noise - latents
                    weight = compute_loss_weighting_for_anima(
                        args.weighting_scheme, sigmas
                    ).view(-1, 1, 1, 1).to(pred.dtype)
                    loss = (F.mse_loss(pred, target, reduction="none") * weight).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(adapter.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_item = loss.detach().float().item()
                loss_window.append(loss_item)
                epoch_loss += loss_item

                if accelerator.sync_gradients:
                    avg_loss = sum(loss_window) / len(loss_window)
                    pb.update(1)
                    global_step += 1

                    pb.set_postfix({
                        "Epoch": f"{epoch + 1}/{args.max_train_epochs}",
                        "Loss": f"{loss_item:.4f}",
                        "Avg": f"{avg_loss:.4f}",
                        "LR": f"{optimizer.param_groups[0]['lr']:.6f}",
                    })

        epoch_avg = epoch_loss / len(train_dl)
        logger.info(f"📝 Epoch {epoch + 1} 完成 | 训练损失: {epoch_avg:.4f}")

        if val_dl:
            adapter.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    if batch["latents"] is not None:
                        latents = batch["latents"].to(device, dtype=weight_dtype, non_blocking=True)
                    else:
                        pixel_values = batch["pixel_values"]
                        latents = encode_pixels_to_latents_online(
                            vae_model=vae_model,
                            pixel_values=pixel_values,
                            device=device,
                            weight_dtype=weight_dtype,
                            vae_encode_batch_size=args.vae_encode_batch_size,
                        )
                        del pixel_values
                    qwen_ids = batch["qwen_input_ids"].to(device)
                    qwen_mask = batch["qwen_attention_mask"].to(device)
                    t5_ids = batch["t5_input_ids"].to(device)
                    t5_mask = batch["t5_attention_mask"].to(device)
                    style_emb = batch["style_embeds"].to(device, weight_dtype)
                    timesteps = batch["timestep"].to(device)
                    h, w = batch["h"][0].item(), batch["w"][0].item()

                    tokens = [qwen_ids, qwen_mask, t5_ids, t5_mask]
                    te_out = text_encoding_strategy.encode_tokens(tokenize_strategy, [qwen_model], tokens)
                    prompt_emb, attn_mask = te_out[0], te_out[1]

                    unwrapped_adapter = accelerator.unwrap_model(adapter)
                    unwrapped_adapter.set_current_resolution(h, w)
                    unwrapped_adapter.set_text_embeds_cache(prompt_emb)
                    unwrapped_adapter.set_style_embeds(style_emb)

                    if latents.ndim == 5:
                        latents = latents.squeeze(2)

                    noise = torch.randn_like(latents)

                    noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
                        args, noise_scheduler, latents, noise, device, weight_dtype
                    )
                    timesteps = timesteps / 1000.0

                    bs = latents.shape[0]
                    h_latent = latents.shape[-2]
                    w_latent = latents.shape[-1]
                    pad_mask = torch.zeros(
                        bs, 1, h_latent, w_latent,
                        dtype=weight_dtype,
                        device=device
                    )

                    noisy_model_input = noisy_model_input.unsqueeze(2)

                    pred = dit_model(
                        noisy_model_input, timesteps, prompt_emb, padding_mask=pad_mask,
                        target_input_ids=t5_ids, target_attention_mask=t5_mask,
                        source_attention_mask=attn_mask
                    )
                    pred = pred.squeeze(2)

                    target = noise - latents
                    weight = compute_loss_weighting_for_anima(
                        args.weighting_scheme, sigmas
                    ).view(-1, 1, 1, 1).to(pred.dtype)
                    loss = (F.mse_loss(pred, target, reduction="none") * weight).mean()
                    val_loss += loss.item()

            val_avg = val_loss / len(val_dl)
            logger.info(f"📊 验证损失: {val_avg:.4f}")
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.makedirs(args.output_dir, exist_ok=True)
                    save_file(
                        accelerator.unwrap_model(adapter).state_dict(),
                        os.path.join(args.output_dir, f"{args.output_name}_best.safetensors")
                    )
                    logger.info(f"🏆 保存最佳检查点 (val_loss={val_avg:.4f})")

        if (epoch + 1) % args.save_every_n_epochs == 0 or epoch == args.max_train_epochs - 1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.makedirs(args.output_dir, exist_ok=True)
                save_file(
                    accelerator.unwrap_model(adapter).state_dict(),
                    os.path.join(args.output_dir, f"{args.output_name}_epoch_{epoch + 1}.safetensors")
                )
                logger.info(f"💾 保存检查点: epoch_{epoch + 1}")

    logger.info("🎉 训练完成！")
    if val_dl:
        logger.info(f"🏆 最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anima DiT Adapter 训练脚本（8模块完整版）")

    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--output_name", default="anima_adapter")
    parser.add_argument("--resume", default=None)

    parser.add_argument("--anima_model_path", required=True, help="Anima DiT底模路径")
    parser.add_argument("--qwen_model_path", required=True, help="Qwen3路径")
    parser.add_argument("--vae_model_path", required=True, help="VAE路径")
    parser.add_argument("--t5_tokenizer_path", type=str, default=None, help="T5分词器路径（默认None，自动从DiT提取）")
    parser.add_argument("--clip_model_path", default="openai/clip-vit-large-patch14", help="CLIP模型名/本地路径")

    # VAE配置
    parser.add_argument("--vae_chunk_size", type=int, default=0)
    parser.add_argument("--vae_disable_cache", action="store_true")

    parser.add_argument(
        "--vae_precompute_batch_size",
        type=int,
        default=2,
        help="启用预计算时，VAE 批量编码大小。A800 80GB 可尝试 2/4/8，OOM 就降低。",
    )

    parser.add_argument(
        "--clip_precompute_batch_size",
        type=int,
        default=8,
        help="启用 style_attn 时，CLIP 风格特征批量编码大小。",
    )

    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=1,
        help="关闭预计算时，在线 VAE 编码的小批大小。2048 分辨率建议 1。",
    )

    parser.add_argument(
        "--vae_precompute_device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="VAE 预计算设备。高分辨率建议 cpu，避免 CUDA OOM / illegal memory access。",
    )

    parser.add_argument(
        "--skip_failed_precompute",
        action="store_true",
        help="预计算遇到坏图时跳过。注意：跳过后该图如果仍在 sample_list 中，训练读取时仍可能报错，默认不建议开启。",
    )

    # 文本配置
    parser.add_argument("--qwen3_max_token_length", type=int, default=77)
    parser.add_argument("--t5_max_token_length", type=int, default=256)

    # 分辨率
    parser.add_argument("--max_width", type=int, default=1024)
    parser.add_argument("--max_height", type=int, default=1024)

    # 训练配置
    parser.add_argument("--val_split", type=float, default=0.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--caption_dropout_rate", type=float, default=0.1)
    parser.add_argument("--shuffle_caption", action="store_true", default=True)
    parser.add_argument("--timestep_sampling", default="sigmoid", choices=["uniform", "sigmoid"])
    parser.add_argument("--weighting_scheme", default="logit_normal")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--discrete_flow_shift", type=float, default=3.0)

    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Scale factor for sigmoid timestep sampling")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale factor for mode timestep sampling")

    parser.add_argument("--ip_noise_gamma", type=float, default=None,
                        help="Input perturbation gamma for noise schedule")

    # 显存控制
    parser.add_argument(
        "--gradient_checkpointing",
        "--enable_gradient_checkpointing",
        action="store_true",
        dest="gradient_checkpointing",
        help="开启 gradient checkpointing，降低显存占用但会降低训练速度",
    )
    parser.add_argument(
        "--enable_precompute_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否预计算 latent/style 缓存。可用 --no-enable_precompute_embeddings 关闭。",
    )
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--split_attn", action="store_true")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=12)

    # 8模块完整配置参数
    parser.add_argument("--dit_hidden_size", type=int, default=2048)
    parser.add_argument("--num_blocks", type=int, default=28)

    parser.add_argument("--text_embed_dim", type=int, default=1024)
    parser.add_argument("--style_dim", type=int, default=768)  # CLIP ViT-L 的真实维度

    # 原有4个基础模块
    parser.add_argument("--enable_mod_text", action="store_true", default=True)
    parser.add_argument("--enable_semantic_scale", action="store_true", default=True)
    parser.add_argument("--enable_local_conv", action="store_true", default=True)
    parser.add_argument("--local_conv_layers", type=int, default=6)
    parser.add_argument("--local_conv_kernel", type=int, default=3)
    parser.add_argument("--local_conv_start_layer", type=int, default=None, help="None=默认尾添")
    parser.add_argument("--enable_style_attn", action="store_true", default=False,
                        help="开启画风交叉注意力（自动清华源下载CLIP）")
    parser.add_argument("--style_attn_layers", type=int, default=8)
    parser.add_argument("--style_num_heads", type=int, default=16)
    parser.add_argument("--style_attn_start_layer", type=int, default=None, help="None=默认尾添")

    # 新增4个前沿模块
    parser.add_argument("--enable_edge_detail", action="store_true", default=False,
                        help="开启边缘细节增强（解决线条模糊）")
    parser.add_argument("--edge_detail_layers", type=int, default=5)
    parser.add_argument("--edge_detail_kernel", type=int, default=3)
    parser.add_argument("--edge_detail_start_layer", type=int, default=0, help="默认从第0层开始")
    parser.add_argument("--enable_subject_attn", action="store_true", default=False,
                        help="开启主体交叉注意力（防止画风吞主体）")
    parser.add_argument("--subject_attn_layers", type=int, default=6)
    parser.add_argument("--subject_attn_heads", type=int, default=16)
    parser.add_argument("--subject_attn_start_layer", type=int, default=6)
    parser.add_argument("--enable_contrast_mod", action="store_true", default=False)
    parser.add_argument("--enable_color_tune", action="store_true", default=False)

    # 训练轮次、batch、优化器相关
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--max_train_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cosine_scheduler", action="store_true")
    parser.add_argument("--lr_warmup_steps", type=int, default=200)

    args = parser.parse_args()
    main(args)
