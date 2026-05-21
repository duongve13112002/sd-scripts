# NanoSaur loading utilities
# Provides model/VAE/text-encoder loading helpers for sd-scripts integration.

import tempfile
from typing import Optional, Tuple, Union

import sentencepiece as spm
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from library.nanosaur_models import (
    MODEL_CHANNELS,
    MODEL_DECODER_HIDDEN,
    MODEL_DECODER_LAYERS,
    MODEL_DIM,
    MODEL_ENCODER_LAYERS,
    MODEL_HEADS,
    MODEL_PATCH,
    MODEL_TEXT_BLOCKS,
    TEXT_ATTENTION_HEADS,
    TEXT_EMBED_DIM,
    TEXT_HEAD_DIM,
    TEXT_INTERMEDIATE_SIZE,
    TEXT_KEY_VALUE_HEADS,
    TEXT_LAYERS,
    TEXT_MAX_LENGTH,
    TEXT_MAX_POSITION_EMBEDDINGS,
    TEXT_SLIDING_WINDOW,
    TEXT_VOCAB_SIZE,
    LATENT_SCALE,
    LATENT_SHIFT,
    NanoSaurTransformer2DModel,
    NanoSaurVAE,
)
from library.safetensors_utils import load_safetensors
import logging

logger = logging.getLogger(__name__)

# Version constant

MODEL_VERSION_NANOSAUR = "nanosaur"


# State dict helpers


def _clean_state_dict(state_dict: dict) -> dict:
    """Remove common prefixes added by DDP / torch.compile wrappers."""
    return {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


# NanoSaur SentencePiece tokenizer wrapper


class NanoSaurSentencePieceTokenizer:
    """
    Thin wrapper around a SentencePieceProcessor loaded from a byte tensor
    (the ``spiece_model`` key inside the text-encoder safetensors file).
    """

    def __init__(self, spiece_model: torch.Tensor, max_length: int = TEXT_MAX_LENGTH) -> None:
        self.max_length = max_length
        model_bytes = bytes(spiece_model.cpu().numpy().tolist())
        self.processor = spm.SentencePieceProcessor()
        with tempfile.NamedTemporaryFile(suffix=".model", delete=True) as handle:
            handle.write(model_bytes)
            handle.flush()
            self.processor.Load(handle.name)
        self.bos_token_id: int = 2
        self.pad_token_id: int = 0

    def __call__(
        self, captions: list, device: Union[str, torch.device]
    ) -> dict:
        rows = []
        for caption in captions:
            ids = [self.bos_token_id] + self.processor.EncodeAsIds(caption)
            ids = ids[: self.max_length]
            ids.extend([self.pad_token_id] * (self.max_length - len(ids)))
            rows.append(ids)
        input_ids = torch.tensor(rows, device=device, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).to(torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_encode(
        self, captions: list, device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, attention_mask) tensors."""
        out = self(captions, device)
        return out["input_ids"], out["attention_mask"]


# Gemma3 config builder


def _build_gemma3_config() -> Gemma3TextConfig:
    """Return the Gemma3TextConfig matching the NanoSaur text encoder."""
    return Gemma3TextConfig(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_EMBED_DIM,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
        num_hidden_layers=TEXT_LAYERS,
        num_attention_heads=TEXT_ATTENTION_HEADS,
        num_key_value_heads=TEXT_KEY_VALUE_HEADS,
        head_dim=TEXT_HEAD_DIM,
        max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=1e-6,
        qkv_bias=False,
        attention_bias=False,
        sliding_window=TEXT_SLIDING_WINDOW,
        use_cache=False,
    )


# Model loaders


def load_nanosaur_model(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
) -> NanoSaurTransformer2DModel:
    """
    Load the NanoSaur diffusion transformer from a safetensors checkpoint.

    Args:
        ckpt_path: Path to the safetensors file.
        dtype: Target dtype (e.g. torch.bfloat16). Pass ``None`` for fp32.
        device: Target device.
        disable_mmap: Disable memory-mapped loading.

    Returns:
        Loaded NanoSaurTransformer2DModel.
    """
    logger.info("Building NanoSaur diffusion model")
    with torch.device("meta"):
        model = NanoSaurTransformer2DModel(
            in_channels=MODEL_CHANNELS,
            num_groups=MODEL_HEADS,
            hidden_size=MODEL_DIM,
            decoder_hidden_size=MODEL_DECODER_HIDDEN,
            num_encoder_blocks=MODEL_ENCODER_LAYERS,
            num_decoder_blocks=MODEL_DECODER_LAYERS,
            num_text_blocks=MODEL_TEXT_BLOCKS,
            patch_size=MODEL_PATCH,
            txt_embed_dim=TEXT_EMBED_DIM,
        ).to(dtype or torch.float32)

    logger.info(f"Loading NanoSaur state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    sd = _clean_state_dict(sd)
    info = model.load_state_dict(sd, strict=False, assign=True)
    if info.missing_keys:
        logger.warning(f"NanoSaur model: missing keys in checkpoint: {info.missing_keys}")
    if info.unexpected_keys:
        logger.info(f"NanoSaur model: ignoring extra checkpoint keys (e.g. projector): {info.unexpected_keys}")
    return model


def load_nanosaur_vae(
    ckpt_path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
) -> "NanoSaurVAEWrapper":
    """
    Load the NanoSaur VAE from a safetensors checkpoint and wrap it so that
    ``encode`` / ``decode`` apply the canonical scale+shift automatically.

    Args:
        ckpt_path: Path to the VAE safetensors file.
        dtype: Target dtype.
        device: Target device.
        disable_mmap: Disable memory-mapped loading.

    Returns:
        NanoSaurVAEWrapper instance (scale/shift applied on encode/decode).
    """
    logger.info("Building NanoSaur VAE")
    with torch.device("meta"):
        vae = NanoSaurVAE(latent_dim=MODEL_CHANNELS).to(dtype)

    logger.info(f"Loading NanoSaur VAE state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    sd = _clean_state_dict(sd)
    info = vae.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded NanoSaur VAE: {info}")
    return NanoSaurVAEWrapper(vae, device=device, dtype=dtype)


def load_nanosaur_text_encoder(
    ckpt_path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
) -> Tuple[NanoSaurSentencePieceTokenizer, Gemma3ForCausalLM]:
    """
    Load the NanoSaur text encoder (Gemma3 270M) and SentencePiece tokenizer
    from a single safetensors file.  The file must contain a ``spiece_model``
    key holding the raw SentencePiece model bytes as a uint8 tensor.

    Args:
        ckpt_path: Path to the text-encoder safetensors file.
        dtype: Target dtype.
        device: Target device.
        disable_mmap: Ignored (load_file always loads eagerly for this model).

    Returns:
        (tokenizer, text_encoder) – both ready for inference.
    """
    logger.info(f"Loading NanoSaur text encoder from {ckpt_path}")
    # load_file loads everything to CPU; we cast to device+dtype after
    checkpoint = load_file(ckpt_path, device="cpu")

    spiece_model_tensor = checkpoint.pop("spiece_model")
    weights = dict(checkpoint)
    # Gemma3ForCausalLM requires lm_head weight (tied embedding)
    if "lm_head.weight" not in weights and "model.embed_tokens.weight" in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

    config = _build_gemma3_config()
    text_encoder = Gemma3ForCausalLM(config)
    info = text_encoder.load_state_dict(weights, strict=True)
    logger.info(f"Loaded NanoSaur text encoder: {info}")
    text_encoder = text_encoder.to(device=device, dtype=dtype).eval()

    tokenizer = NanoSaurSentencePieceTokenizer(spiece_model_tensor, max_length=TEXT_MAX_LENGTH)
    return tokenizer, text_encoder


# VAE wrapper


class NanoSaurVAEWrapper(nn.Module):
    """
    Wraps a ``NanoSaurVAE`` and applies the canonical latent scale / shift on
    ``encode`` and ``decode``, matching the ComfyUI ``NanoSaurLatentFormat``.

    Encoding:   z_scaled = (z_raw + LATENT_SHIFT) / LATENT_SCALE
    Decoding:   z_raw    = z_scaled * LATENT_SCALE - LATENT_SHIFT
    """

    def __init__(
        self,
        vae: NanoSaurVAE,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.vae = vae
        self._device = torch.device(device)
        self._dtype = dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image tensor to scaled latents: (B, 96, H/16, W/16)."""
        x = x.to(self._device, dtype=self._dtype)
        raw = self.vae.encode(x)
        return (raw + LATENT_SHIFT) / LATENT_SCALE

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode scaled latents to image tensor: (B, 3, H, W) in [-1, 1]."""
        z = z.to(self._device, dtype=self._dtype)
        z_raw = z * LATENT_SCALE - LATENT_SHIFT
        return self.vae.decode(z_raw)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # Keep _device/_dtype in sync using the first available parameter
        try:
            p = next(result.vae.parameters())
            result._device = p.device
            result._dtype = p.dtype
        except StopIteration:
            pass
        return result
