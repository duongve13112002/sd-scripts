# NanoSaur model architecture

from functools import lru_cache
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention as attention

from library import custom_offloading_utils
import logging

logger = logging.getLogger(__name__)

# Model constants

MODEL_CHANNELS: int = 96
MODEL_HEADS: int = 16
MODEL_DIM: int = 1536
MODEL_DECODER_HIDDEN: int = 2048
MODEL_ENCODER_LAYERS: int = 26
MODEL_DECODER_LAYERS: int = 3
MODEL_TEXT_BLOCKS: int = 2
MODEL_PATCH: int = 1

TEXT_EMBED_DIM: int = 640
TEXT_VOCAB_SIZE: int = 262144
TEXT_INTERMEDIATE_SIZE: int = 2048
TEXT_LAYERS: int = 18
TEXT_ATTENTION_HEADS: int = 4
TEXT_KEY_VALUE_HEADS: int = 1
TEXT_HEAD_DIM: int = 256
TEXT_MAX_POSITION_EMBEDDINGS: int = 32768
TEXT_SLIDING_WINDOW: int = 512
TEXT_MAX_LENGTH: int = 128

LATENT_SCALE: float = 2.3623
LATENT_SHIFT: float = 0.0179

# Disable torch.compile for specific methods that use dynamic shapes / caching
_compile_disable = torch.compiler.disable if hasattr(torch, "compiler") else torch._dynamo.disable


# Diffusion model building blocks


class Norm(nn.Module):
    """RMSNorm used throughout the model."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Embed(nn.Module):
    """Linear projection with optional normalization."""

    def __init__(self, in_chans: int, embed_dim: int, norm_layer=None, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into a dense vector representation."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


def precompute_freqs_cis_2d(
    dim: int, height: int, width: int, theta: float = 10000.0, scale=1.0
) -> torch.Tensor:
    if isinstance(scale, (float, int)):
        scale = (float(scale), float(scale))
    scale_y, scale_x = float(scale[0]), float(scale[1])

    rotary_dim = (dim // 4) * 4
    if rotary_dim == 0:
        return torch.empty(height * width, 0, 2, dtype=torch.float32)

    axis_dim = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
    y_pos = (torch.arange(height, dtype=torch.float32) + 0.5) / height * scale_y
    x_pos = (torch.arange(width, dtype=torch.float32) + 0.5) / width * scale_x
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)

    x_freqs = torch.outer(x_pos, inv_freq)
    y_freqs = torch.outer(y_pos, inv_freq)
    cos = torch.cat([torch.cos(x_freqs), torch.cos(y_freqs)], dim=-1)
    sin = torch.cat([torch.sin(x_freqs), torch.sin(y_freqs)], dim=-1)
    return torch.stack((cos, sin), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs_cis.unbind(dim=-1)
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim == 0:
        return xq, xk

    cos = cos[None, None, :, :].to(dtype=xq.dtype, device=xq.device)
    sin = sin[None, None, :, :].to(dtype=xq.dtype, device=xq.device)

    xq_rot, xq_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    xk_rot, xk_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]
    xq1, xq2 = xq_rot.chunk(2, dim=-1)
    xk1, xk2 = xk_rot.chunk(2, dim=-1)
    xq_rot = torch.cat([xq1 * cos - xq2 * sin, xq1 * sin + xq2 * cos], dim=-1)
    xk_rot = torch.cat([xk1 * cos - xk2 * sin, xk1 * sin + xk2 * cos], dim=-1)

    if xq_pass.shape[-1] == 0:
        return xq_rot, xk_rot
    return torch.cat([xq_rot, xq_pass], dim=-1), torch.cat([xk_rot, xk_pass], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class LocalContext2D(nn.Module):
    """Lightweight local depthwise convolution for spatial context."""

    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) for _ in range(num_layers)]
        )
        self.lambdas = nn.Parameter(0.1 * torch.ones(num_layers))

    def forward(self, x: torch.Tensor, layer_idx: int, h: int, w: int) -> torch.Tensor:
        b, n, d = x.shape
        x_2d = x.view(b, h, w, d).permute(0, 3, 1, 2)
        local = self.convs[layer_idx](x_2d).permute(0, 2, 3, 1).view(b, n, d)
        return x + self.lambdas[layer_idx] * local


class Attention(nn.Module):
    """Combined self-attention + cross-attention over text tokens."""

    def __init__(
        self, dim: int, num_heads: int = 8, qkv_bias: bool = False, use_cross_attention: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.use_cross_attention = use_cross_attention
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if use_cross_attention:
            self.kv_y = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = Norm(dim // num_heads)
        self.k_norm = Norm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pos: torch.Tensor,
        y_token_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        qkv_x = self.qkv_x(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q)
        kx = self.k_norm(kx)
        q, kx = apply_rotary_emb(q, kx, freqs_cis=pos)
        if self.use_cross_attention:
            kv_y = self.kv_y(y).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
            ky, vy = kv_y[0], kv_y[1]
            ky = self.k_norm(ky)
            k = torch.cat([kx, ky], dim=2)
            v = torch.cat([vx, vy], dim=2)
        else:
            k = kx
            v = vx
        q = q.view(b, self.num_heads, -1, c // self.num_heads)
        k = k.view(b, self.num_heads, -1, c // self.num_heads)
        v = v.view(b, self.num_heads, -1, c // self.num_heads)
        if self.use_cross_attention and y_token_weights is not None:
            y_token_weights = y_token_weights.to(device=q.device, dtype=q.dtype)
            y_token_bias = torch.log(torch.clamp(y_token_weights, min=1e-4))
            x_token_bias = torch.zeros(b, n, device=q.device, dtype=q.dtype)
            attn_bias = torch.cat([x_token_bias, y_token_bias], dim=1)[:, None, None, :]
            x = attention(q, k, v, attn_mask=attn_bias)
        else:
            x = attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class FlattenDiTBlock(nn.Module):
    """Main DiT block used in the encoder stack (shared adaLN or per-block)."""

    def __init__(
        self,
        hidden_size: int,
        groups: int,
        mlp_ratio: float = 4,
        is_encoder_block: bool = False,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False, use_cross_attention=use_cross_attention)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, int(hidden_size * mlp_ratio))
        self.is_encoder_block = is_encoder_block
        if not is_encoder_block:
            self.adaLN_modulation = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
        pos: torch.Tensor,
        shared_ada_ln: Optional[nn.Module] = None,
        local_context: Optional["LocalContext2D"] = None,
        layer_idx: Optional[int] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        y_token_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ada_ln_output = shared_ada_ln(c) if self.is_encoder_block else self.adaLN_modulation(c)
        if local_context is not None and h is not None and w is not None:
            x = local_context(x, layer_idx, h, w)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_ln_output.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), y, pos, y_token_weights=y_token_weights
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def precompute_freqs_cis_ex2d(
    dim: int, height: int, width: int, theta: float = 10000.0, scale=1.0
) -> torch.Tensor:
    if isinstance(scale, float):
        scale = (scale, scale)
    x_pos = torch.linspace(0, height * scale[0], width)
    y_pos = torch.linspace(0, width * scale[1], height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    return torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1).reshape(height * width, -1)


class NerfEmbedder(nn.Module):
    """Position-aware patch embedder for the decoder path."""

    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(nn.Linear(in_channels + max_freqs**2, hidden_size_input, bias=True))

    @lru_cache
    def fetch_pos(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = precompute_freqs_cis_ex2d(self.max_freqs**2 * 2, patch_size, patch_size)
        return pos[None, :, :].to(device=device, dtype=dtype)

    @torch.compiler.disable
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, patch_tokens, _ = inputs.shape
        patch_size = int(patch_tokens**0.5)
        dct = self.fetch_pos(patch_size, inputs.device, inputs.dtype).repeat(batch, 1, 1)
        return self.embedder(torch.cat([inputs, dct], dim=-1))


class TextRefineAttention(nn.Module):
    """Self-attention used inside TextRefineBlock."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = Norm(dim // num_heads)
        self.k_norm = Norm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv_x = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        x = attention(
            q.view(b, self.num_heads, -1, c // self.num_heads),
            k.view(b, self.num_heads, -1, c // self.num_heads),
            v.view(b, self.num_heads, -1, c // self.num_heads),
        )
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class TextRefineBlock(nn.Module):
    """Refines text embeddings conditioned on timestep."""

    def __init__(self, hidden_size: int, groups: int, mlp_ratio: float = 4):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = TextRefineAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ResBlock(nn.Module):
    """Residual block used in the lightweight MLP decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        return x + gate_mlp * self.mlp(modulate(self.in_ln(x), shift_mlp, scale_mlp))


class FinalLayer(nn.Module):
    """Final projection layer for the decoder."""

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class SimpleMLPAdaLN(nn.Module):
    """Lightweight MLP decoder that reconstructs per-patch latents."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        patch_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        y = self.cond_embed(c).reshape(c.shape[0], self.patch_size**2, -1)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x)


# Main NanoSaur Diffusion Transformer


class NanoSaurTransformer2DModel(nn.Module):
    """
    NanoSaur diffusion transformer (DiT-based, rectified flow).

    Architecture overview:
    - Input: patchified latent tokens (patch_size=1 by default)
    - 2D RoPE positional encodings
    - 26 FlattenDiTBlocks with shared adaLN
      - SPRINT mechanism: fast(2) + global(22) + head(2) blocks
    - 2 TextRefineBlocks for conditioning
    - 1 LocalContext2D module for spatial locality
    - Lightweight MLP decoder (SimpleMLPAdaLN)
    """

    def __init__(
        self,
        in_channels: int = MODEL_CHANNELS,
        num_groups: int = MODEL_HEADS,
        hidden_size: int = MODEL_DIM,
        decoder_hidden_size: int = MODEL_DECODER_HIDDEN,
        num_encoder_blocks: int = MODEL_ENCODER_LAYERS,
        num_decoder_blocks: int = MODEL_DECODER_LAYERS,
        num_text_blocks: int = MODEL_TEXT_BLOCKS,
        patch_size: int = MODEL_PATCH,
        txt_embed_dim: int = TEXT_EMBED_DIM,
        sprint_num_f: int = 2,
        sprint_num_h: int = 2,
        rope_scale: float = 2 * math.pi,
        **kwargs,
    ):
        super().__init__()
        assert (hidden_size // num_groups) % 4 == 0, "hidden_size // num_groups must be divisible by 4"

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.patch_size = patch_size
        self.sprint_num_f = sprint_num_f
        self.sprint_num_h = sprint_num_h
        self.sprint_num_g = num_encoder_blocks - sprint_num_f - sprint_num_h
        self.num_blocks = num_encoder_blocks
        self.rope_scale = rope_scale

        # Block swap state (None = disabled)
        self.blocks_to_swap: Optional[int] = None
        self.offloader: Optional[custom_offloading_utils.ModelOffloader] = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.mask_token2 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.fusion_proj = nn.Linear(2 * hidden_size, hidden_size, bias=True)

        self.s_embedder = Embed(in_channels * patch_size**2, hidden_size, bias=True)
        self.x_embedder = NerfEmbedder(in_channels, decoder_hidden_size, max_freqs=8)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = Embed(txt_embed_dim, hidden_size, bias=True, norm_layer=Norm)

        self.shared_encoder_adaLN = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.blocks = nn.ModuleList(
            [
                FlattenDiTBlock(
                    hidden_size,
                    num_groups,
                    is_encoder_block=True,
                    use_cross_attention=(i % 2 == 0),
                )
                for i in range(num_encoder_blocks)
            ]
        )
        self.text_refine_blocks = nn.ModuleList(
            [TextRefineBlock(hidden_size, num_groups) for _ in range(num_text_blocks)]
        )
        self.local_context = LocalContext2D(hidden_size, num_encoder_blocks)
        self.dec_net = SimpleMLPAdaLN(
            in_channels=decoder_hidden_size,
            model_channels=decoder_hidden_size,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_decoder_blocks,
            patch_size=patch_size,
        )
        self.precompute_pos: dict = {}

    @property
    def dtype(self) -> torch.dtype:
        return self.s_embedder.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.s_embedder.proj.weight.device

    # Block swap API

    def enable_block_swap(self, num_blocks: int, device: torch.device) -> None:
        """
        Enable CPU↔GPU block swapping to reduce VRAM usage.

        Args:
            num_blocks: Number of encoder blocks to swap (max: num_blocks-2).
            device: The GPU device for training.
        """
        self.blocks_to_swap = num_blocks
        assert self.blocks_to_swap <= self.num_blocks - 2, (
            f"NanoSaur: Cannot swap more than {self.num_blocks - 2} blocks. "
            f"Requested: {self.blocks_to_swap} blocks."
        )
        self.offloader = custom_offloading_utils.ModelOffloader(self.blocks, self.blocks_to_swap, device)
        logger.info(
            f"NanoSaur: Block swap enabled. Swapping {num_blocks} blocks, "
            f"total blocks: {self.num_blocks}, device: {device}."
        )

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        """Move all parameters to device except the swapped blocks."""
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = nn.ModuleList()  # temporarily empty to skip .to()

        self.to(device)

        if self.blocks_to_swap:
            self.blocks = save_blocks

    def prepare_block_swap_before_forward(self) -> None:
        """Must be called before each forward pass when block swap is active."""
        if not self.blocks_to_swap:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def switch_block_swap_for_inference(self) -> None:
        if not self.blocks_to_swap:
            return
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if not self.blocks_to_swap:
            return
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()

    # Gradient checkpointing

    def enable_gradient_checkpointing(self) -> None:
        for block in self.blocks:
            block.gradient_checkpointing = True
        logger.info("NanoSaur: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self) -> None:
        for block in self.blocks:
            block.gradient_checkpointing = False
        logger.info("NanoSaur: Gradient checkpointing disabled.")

    def get_block_swap_module_list(self) -> list:
        return list(self.blocks)

    def get_checkpointing_wrap_module_list(self) -> list:
        return list(self.blocks)

    # Position encoding

    @_compile_disable
    def fetch_pos(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (int(height), int(width))
        pos = self.precompute_pos.get(key)
        if pos is None:
            pos = precompute_freqs_cis_2d(
                self.hidden_size // self.num_groups, key[0], key[1], scale=self.rope_scale
            )
            self.precompute_pos[key] = pos
        return pos.to(device=device)

    # SPRINT helpers

    def _sprint_fuse(self, s_enc: torch.Tensor, g_pad: torch.Tensor) -> torch.Tensor:
        return self.fusion_proj(torch.cat([s_enc, g_pad], dim=-1))

    # Core forward

    def _forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        uncond: bool = False,
        token_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict clean x0 from noisy input xt.

        Args:
            x: Noisy latents (B, C, H, W).
            timesteps: Diffusion timesteps (B,).
            context: Text embeddings (B, L, D).
            uncond: If True uses sparse SPRINT path (skip global blocks).
            token_weights: Per-token attention weights for emphasis (B, L).

        Returns:
            Predicted x0 (B, C, H, W).
        """
        device = self.s_embedder.proj.weight.device
        embed_dtype = self.s_embedder.proj.weight.dtype
        if context.device != device:
            context = context.to(device)

        y_emb = self.y_embedder(context).view(context.size(0), -1, self.hidden_size).to(embed_dtype)
        batch, _, height, width = x.shape
        x_tokens = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size)
        xpos = self.fetch_pos(height // self.patch_size, width // self.patch_size, x.device)

        t_emb = self.t_embedder(timesteps.view(-1)).view(batch, -1, self.hidden_size)
        condition = torch.nn.functional.silu(t_emb)
        y_latent = y_emb.to(dtype=t_emb.dtype)
        for block in self.text_refine_blocks:
            y_latent = block(y_latent, condition)

        s = self.s_embedder(x_tokens)
        h_patches = height // self.patch_size
        w_patches = width // self.patch_size

        # Phase 1: Fast blocks (always run)
        for i in range(self.sprint_num_f):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(i)
            s = self.blocks[i](
                s, y_latent, condition, xpos,
                shared_ada_ln=self.shared_encoder_adaLN,
                local_context=self.local_context,
                layer_idx=i, h=h_patches, w=w_patches,
                y_token_weights=token_weights,
            )
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.blocks, i)
        s_enc = s

        # Phase 2: Global blocks (SPRINT — skipped for uncond path)
        # wait/submit always run so the block-swap pipeline stays live even when
        # computation is skipped (uncond=True path drops these 22 blocks).
        s_sparse = s
        for i in range(self.sprint_num_f, self.sprint_num_f + self.sprint_num_g):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(i)
            if not uncond:
                s_sparse = self.blocks[i](
                    s_sparse, y_latent, condition, xpos,
                    shared_ada_ln=self.shared_encoder_adaLN,
                    y_token_weights=token_weights,
                )
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.blocks, i)

        g_pad = self.mask_token2.expand_as(s_sparse) if uncond else s_sparse
        s = self._sprint_fuse(s_enc, g_pad)

        # Phase 3: Head blocks (always run)
        for i in range(self.sprint_num_f + self.sprint_num_g, len(self.blocks)):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(i)
            s = self.blocks[i](
                s, y_latent, condition, xpos,
                shared_ada_ln=self.shared_encoder_adaLN,
                local_context=self.local_context,
                layer_idx=i, h=h_patches, w=w_patches,
                y_token_weights=token_weights,
            )
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.blocks, i)

        s = torch.nn.functional.silu(t_emb + s)

        # Decode each patch independently
        batch_size, length, _ = s.shape
        x_dec = x_tokens.reshape(batch_size * length, self.in_channels, self.patch_size**2).transpose(1, 2)
        s_dec = s.view(batch_size * length, self.hidden_size)
        x_dec = self.x_embedder(x_dec)
        x_dec = self.dec_net(x_dec, s_dec).transpose(1, 2).reshape(batch_size, length, -1)

        return rearrange(
            x_dec,
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=height // self.patch_size,
            w=width // self.patch_size,
            p1=self.patch_size, p2=self.patch_size,
            c=self.in_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Standard sd-scripts-compatible forward.
        Returns velocity: (xt - x0_pred) / t.
        """
        if context is None:
            raise ValueError("NanoSaurTransformer2DModel requires text context.")
        uncond = bool(kwargs.pop("uncond", False))
        token_weights = kwargs.pop("token_weights", None)
        x0 = self._forward(x, timestep, context, uncond=uncond, token_weights=token_weights)
        return (x - x0) / timestep.view(-1, 1, 1, 1)


# VAE


def _nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _Normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


# VAE: DINOv3 Encoder


def _feature_take_indices(num_features: int, indices=None):
    if indices is None:
        return list(range(num_features)), num_features - 1
    if isinstance(indices, int):
        if not 0 < indices <= num_features:
            raise AssertionError(f"last-n ({indices}) out of range (1 to {num_features})")
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            if not 0 <= idx < num_features:
                raise AssertionError(f"feature index {idx} out of range (0 to {num_features - 1})")
            take_indices.append(idx)
    return take_indices, max(take_indices)


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rot_embed_cat(x: torch.Tensor, emb: torch.Tensor, half: bool = False) -> torch.Tensor:
    sin_emb, cos_emb = emb.chunk(2, dim=-1)
    if half:
        return x * cos_emb + _rope_rotate_half(x) * sin_emb
    return x * cos_emb + torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape) * sin_emb


def _make_coords_dinov3(
    height: int,
    width: int,
    normalize_coords: str = "separate",
    grid_indexing: str = "ij",
    grid_offset: float = 0.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    coords_h = torch.arange(0.5, height, device=device, dtype=torch.float32) + grid_offset
    coords_w = torch.arange(0.5, width, device=device, dtype=torch.float32) + grid_offset
    if normalize_coords == "max":
        h_denom = w_denom = float(max(height, width))
    elif normalize_coords == "min":
        h_denom = w_denom = float(min(height, width))
    elif normalize_coords == "separate":
        h_denom, w_denom = float(height), float(width)
    else:
        raise ValueError(f"Unknown normalize_coords: {normalize_coords}")
    coords_h = (coords_h / h_denom).to(dtype)
    coords_w = (coords_w / w_denom).to(dtype)
    if grid_indexing == "xy":
        grid_w, grid_h = torch.meshgrid(coords_w, coords_h, indexing="xy")
        coords = torch.stack([grid_h, grid_w], dim=-1)
    else:
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    return 2.0 * coords.flatten(0, 1) - 1.0


class _RotaryEmbeddingDinoV3(nn.Module):
    def __init__(
        self, dim: int, temperature: float = 100.0, feat_shape=None,
        normalize_coords: str = "separate", grid_offset: float = 0.0,
        grid_indexing: str = "ij", rotate_half: bool = True,
        device=None, dtype=None,
    ):
        super().__init__()
        self.dim = dim
        self.temperature = float(temperature)
        self.feat_shape = feat_shape
        self.normalize_coords = normalize_coords
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing
        self.rotate_half = rotate_half
        self.register_buffer("periods", torch.empty(dim // 4, device=device, dtype=dtype), persistent=False)
        self.register_buffer("pos_embed_cached", None, persistent=False)
        self._init_buffers()

    def _init_buffers(self) -> None:
        exponents = 2.0 * torch.arange(self.dim // 4, device="cpu", dtype=torch.float32) / (self.dim // 2)
        self.periods.copy_(self.temperature ** exponents)
        if self.feat_shape is not None:
            self.pos_embed_cached = self._create_embed(self.feat_shape)

    def _create_embed(self, feat_shape) -> torch.Tensor:
        coords = _make_coords_dinov3(feat_shape[0], feat_shape[1], normalize_coords=self.normalize_coords,
                                     grid_indexing=self.grid_indexing, grid_offset=self.grid_offset)
        coords = coords[:, :, None].to(device=self.periods.device, dtype=self.periods.dtype)
        angles = 2 * math.pi * coords / self.periods[None, None, :]
        angles = angles.flatten(1).tile(2)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def get_embed(self, shape=None) -> torch.Tensor:
        if shape is not None:
            return self._create_embed(shape)
        if self.pos_embed_cached is None:
            raise AssertionError("feature shape must be cached on create")
        return self.pos_embed_cached


class _PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def feat_ratio(self, as_scalar: bool = True):
        return max(self.patch_size) if as_scalar else self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1).contiguous()


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, dim)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class _EvaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_prefix_tokens: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], _apply_rot_embed_cat(q[:, :, npt:, :], rope, half=True)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], _apply_rot_embed_cat(k[:, :, npt:, :], rope, half=True)], dim=2).type_as(v)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(self.norm(x)))


class _EvaBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, num_prefix_tokens: int, init_values: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = _EvaAttention(dim, num_heads=num_heads, num_prefix_tokens=num_prefix_tokens)
        self.gamma_1 = nn.Parameter(torch.full((dim,), init_values))
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        self.gamma_2 = nn.Parameter(torch.full((dim,), init_values))
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope))
        x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class _EvaDinoV3(nn.Module):
    def __init__(self):
        super().__init__()
        img_size, patch_size, embed_dim, depth, num_heads = 256, 16, 768, 12, 12
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 5
        self.patch_embed = _PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.empty(1, 4, embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        self.rope = _RotaryEmbeddingDinoV3(
            dim=embed_dim // num_heads, temperature=100.0,
            feat_shape=self.patch_embed.grid_size, rotate_half=True,
        )
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList(
            [_EvaBlock(embed_dim, num_heads=num_heads, mlp_ratio=4.0, num_prefix_tokens=self.num_prefix_tokens,
                       init_values=1.0e-5) for _ in range(depth)]
        )
        self.norm = nn.Identity()

    def prune_intermediate_layers(self, indices=1, prune_norm: bool = False, prune_head: bool = True):
        take_indices, max_index = _feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[: max_index + 1]
        return take_indices

    def _pos_embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, w, c = x.shape
        x = x.view(b, -1, c)
        x = torch.cat([self.cls_token.expand(b, -1, -1), self.reg_token.expand(b, -1, -1), x], dim=1)
        return self.pos_drop(x), self.rope.get_embed(shape=(h, w))

    def forward_intermediates(self, x: torch.Tensor, indices=None, norm: bool = False,
                              output_fmt: str = "NCHW", intermediates_only: bool = False):
        if output_fmt != "NCHW":
            raise ValueError("Only NCHW output is supported.")
        take_indices, _ = _feature_take_indices(len(self.blocks), indices)
        b, _, height, width = x.shape
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        x = self.norm_pre(x)
        intermediates = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope=rot_pos_embed)
            if i in take_indices:
                intermediates.append(self.norm(x) if norm else x)
        if self.num_prefix_tokens:
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        h, w = self.patch_embed.dynamic_feat_size((height, width))
        intermediates = [y.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
        if intermediates_only:
            return intermediates
        return self.norm(x), intermediates


class _FeatureGetterNet(nn.Module):
    def __init__(self, model: nn.Module, out_indices=(11,)):
        super().__init__()
        self.model = model
        if hasattr(model, "prune_intermediate_layers"):
            out_indices = model.prune_intermediate_layers(out_indices, prune_norm=True)
        self.out_indices = out_indices

    def forward(self, x: torch.Tensor):
        return self.model.forward_intermediates(
            x, indices=self.out_indices, norm=False, output_fmt="NCHW", intermediates_only=True
        )


# VAE: Decoder


class _Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.with_conv else x


class _ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: Optional[int] = None,
                 conv_shortcut: bool = False, dropout: float = 0.0, temb_channels: int = 512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = _Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = _Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.conv1(_nonlinearity(self.norm1(x)))
        if temb is not None and hasattr(self, "temb_proj"):
            h = h + self.temb_proj(_nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(_nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class _AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = _Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.softmax(torch.bmm(q, k) * (int(c) ** -0.5), dim=2)
        v = v.reshape(b, c, h * w)
        h_ = torch.bmm(v, w_.permute(0, 2, 1)).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class _Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, **ignorekwargs):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    _ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=0, dropout=dropout)
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = _Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)
        self.norm_out = _Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, None)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.conv_out(_nonlinearity(self.norm_out(h)))
        return torch.tanh(h) if self.tanh_out else h


# VAE: Semantic encoder


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class _SemanticEncoder(nn.Module):
    def __init__(self, in_dim: int = 768, latent_dim: int = 96, num_blocks: int = 3, num_heads: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.blocks = nn.ModuleList([_TransformerBlock(latent_dim, num_heads=num_heads) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.norm(x))


class _PixelDecoder(nn.Module):
    def __init__(self, latent_dim: int = 96, out_channels: int = 3, out_size: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = _Decoder(
            ch=128, out_ch=out_channels, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
            attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, in_channels=out_channels,
            resolution=out_size, z_channels=latent_dim, give_pre_end=False, tanh_out=False,
        )

    def forward(self, z: torch.Tensor, spatial_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        b, n, c = z.shape
        if spatial_hw is not None:
            h, w = spatial_hw
        else:
            h = w = int(n**0.5)
            assert h * w == n
        return self.decoder(z.permute(0, 2, 1).reshape(b, self.latent_dim, h, w))


class _DINOv3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FeatureGetterNet(_EvaDinoV3(), out_indices=[11])
        self.embed_dim = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)[0]
        b, c, h, w = features.shape
        return features.permute(0, 2, 3, 1).reshape(b, h * w, c)


# Public VAE class


class NanoSaurVAE(nn.Module):
    """
    NanoSaur VAE: DINOv3-based semantic encoder + CNN pixel decoder.

    Encodes images at 16x spatial downscale (1024→64), 96 latent channels.
    VAE parameters: scale=2.3623, shift=0.0179.
    """

    def __init__(self, latent_dim: int = MODEL_CHANNELS):
        super().__init__()
        self.dino_encoder = _DINOv3Encoder()
        self.semantic_encoder = _SemanticEncoder(in_dim=self.dino_encoder.embed_dim, latent_dim=latent_dim)
        self.pixel_decoder = _PixelDecoder(latent_dim=latent_dim, out_size=256)
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

    def _to_imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        img_mean = torch.tensor(self.img_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        img_std = torch.tensor(self.img_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - img_mean) / img_std

    def _from_imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        img_mean = torch.tensor(self.img_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        img_std = torch.tensor(self.img_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = x * img_std + img_mean
        return x * 2.0 - 1.0

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent. Output shape: (B, C, H/16, W/16)."""
        b, _, h, w = x.shape
        x_norm = self._to_imagenet_norm(x)
        dino_features = self.dino_encoder(x_norm)
        z = self.semantic_encoder(dino_features)
        return z.permute(0, 2, 1).reshape(b, -1, h // 16, w // 16)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image. Input shape: (B, C, H, W)."""
        b, c, h, w = z.shape
        z = z.reshape(b, c, h * w).permute(0, 2, 1)
        x_norm = self.pixel_decoder(z, spatial_hw=(h, w))
        return self._from_imagenet_norm(x_norm)
