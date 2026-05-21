# NanoSaur strategy classes for sd-scripts
# Implements the four strategy interfaces:
#   NanoSaurTokenizeStrategy
#   NanoSaurTextEncodingStrategy
#   NanoSaurTextEncoderOutputsCachingStrategy
#   NanoSaurLatentsCachingStrategy

import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from library import train_util
from library.nanosaur_models import TEXT_MAX_LENGTH
from library.nanosaur_utils import NanoSaurSentencePieceTokenizer
from library.strategy_base import (
    LatentsCachingStrategy,
    TextEncoderOutputsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
)
import logging

logger = logging.getLogger(__name__)


# Tokenize strategy


class NanoSaurTokenizeStrategy(TokenizeStrategy):
    """
    Tokenize strategy for NanoSaur.

    Uses a SentencePiece tokenizer that is bundled inside the text-encoder
    safetensors file (``spiece_model`` key).  Pass the already-loaded
    ``NanoSaurSentencePieceTokenizer`` instance.
    """

    def __init__(
        self,
        tokenizer: NanoSaurSentencePieceTokenizer,
        max_length: int = TEXT_MAX_LENGTH,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        input_ids, attention_mask = self.tokenizer.batch_encode(text, device="cpu")
        return [input_ids, attention_mask]


# Text encoding strategy


class NanoSaurTextEncodingStrategy(TextEncodingStrategy):
    """
    Text encoding strategy for NanoSaur.

    Runs the Gemma3 text encoder and returns the last hidden state.
    """

    def __init__(self) -> None:
        super().__init__()

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Args:
            tokenize_strategy: NanoSaurTokenizeStrategy.
            models: [text_encoder]  (Gemma3ForCausalLM).
            tokens: [input_ids, attention_mask]

        Returns:
            [hidden_states, input_ids, attention_mask]
                hidden_states: (B, L, D) last hidden state from Gemma3.
        """
        text_encoder = models[0]
        input_ids, attention_mask = tokens[0], tokens[1]

        input_ids = input_ids.to(text_encoder.device)
        attention_mask = attention_mask.to(text_encoder.device)

        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]  # last layer, (B, L, D)
        return [hidden_states, input_ids, attention_mask]


# Text encoder outputs caching strategy


class NanoSaurTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """
    Caches Gemma3 hidden states to disk (.npz) for NanoSaur training.

    Cached arrays:
        hidden_state:   float32 (L, D)
        input_ids:      int32   (L,)
        attention_mask: int32   (L,)
    """

    NANOSAUR_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_nanosaur_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return (
            os.path.splitext(image_abs_path)[0]
            + NanoSaurTextEncoderOutputsCachingStrategy.NANOSAUR_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX
        )

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            npz = np.load(npz_path)
            for key in ("hidden_state", "input_ids", "attention_mask"):
                if key not in npz:
                    return False
        except Exception as e:
            logger.error(f"Error loading cached text encoder output: {npz_path}")
            raise e
        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        return [data["hidden_state"], data["input_ids"], data["attention_mask"]]

    @torch.no_grad()
    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ) -> None:
        assert isinstance(text_encoding_strategy, NanoSaurTextEncodingStrategy)
        assert isinstance(tokenize_strategy, NanoSaurTokenizeStrategy)

        captions = [info.caption for info in infos]
        tokens = tokenize_strategy.tokenize(captions)

        hidden_states, input_ids, attention_masks = text_encoding_strategy.encode_tokens(
            tokenize_strategy, models, tokens
        )

        # cast to float32 for safe numpy conversion
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()

        hidden_states_np = hidden_states.cpu().numpy()       # (B, L, D)
        input_ids_np = input_ids.cpu().numpy().astype(np.int32)   # (B, L)
        attn_mask_np = attention_masks.cpu().numpy().astype(np.int32)  # (B, L)

        for i, info in enumerate(infos):
            if self.cache_to_disk:
                assert info.text_encoder_outputs_npz is not None
                np.savez(
                    info.text_encoder_outputs_npz,
                    hidden_state=hidden_states_np[i],
                    input_ids=input_ids_np[i],
                    attention_mask=attn_mask_np[i],
                )
            else:
                info.text_encoder_outputs = [
                    hidden_states_np[i],
                    input_ids_np[i],
                    attn_mask_np[i],
                ]


# Latents caching strategy


class NanoSaurLatentsCachingStrategy(LatentsCachingStrategy):
    """
    Caches VAE latents for NanoSaur.

    NanoSaur VAE has a 16x spatial downscale factor (patch_size=16 ViT-B/16).
    The ``NanoSaurVAEWrapper.encode`` already applies scale + shift.
    """

    NANOSAUR_LATENTS_NPZ_SUFFIX = "_nanosaur.npz"
    # NanoSaur: ViT-B/16 encoder → 16x spatial downscale
    VAE_STRIDE = 16

    def __init__(
        self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return NanoSaurLatentsCachingStrategy.NANOSAUR_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + NanoSaurLatentsCachingStrategy.NANOSAUR_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        alpha_mask: bool,
    ) -> bool:
        return self._default_is_disk_cached_latents_expected(
            self.VAE_STRIDE, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
        )

    def load_latents_from_disk(
        self,
        npz_path: str,
        bucket_reso: Tuple[int, int],
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[List[int]],
        Optional[List[int]],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        return self._default_load_latents_from_disk(self.VAE_STRIDE, npz_path, bucket_reso)

    def cache_batch_latents(
        self,
        model,
        image_infos: List,
        flip_aug: bool,
        alpha_mask: bool,
        random_crop: bool,
    ) -> None:
        """
        Cache a batch of image latents.

        ``model`` is expected to be a ``NanoSaurVAEWrapper`` whose ``encode``
        method returns already-scaled latents.
        """
        encode_by_vae = lambda img_tensor: model.encode(img_tensor).to("cpu")
        vae_device = model.device
        vae_dtype = model.dtype

        self._default_cache_batch_latents(
            encode_by_vae,
            vae_device,
            vae_dtype,
            image_infos,
            flip_aug,
            alpha_mask,
            random_crop,
            multi_resolution=True,
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae_device)
