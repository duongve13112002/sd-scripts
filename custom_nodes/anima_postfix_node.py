"""
ComfyUI custom node: Load Anima Postfix Embedding

Loads postfix tuning weights (trained with networks/postfix_anima.py) and injects
them into the model during inference.

Supports three modes based on how the weights were trained:
- "hidden" (default): appends postfix to Qwen3 output hidden states (adapter side)
- "embedding": appends postfix to Qwen3 input embeddings (goes through all transformer layers)
- "cfg": loads dual pos/neg postfix — pos on model (adapter), neg on separate CLIP output
"""

import torch
import folder_paths
from safetensors.torch import load_file


class PostfixLLMAdapterWrapper(torch.nn.Module):
    """Wraps an existing LLMAdapter to append learned postfix embeddings.

    Appended (not prepended) to preserve tag-order positional encoding (RoPE).
    """

    def __init__(self, original_adapter, postfix_embeds, strength=1.0):
        super().__init__()
        self.original_adapter = original_adapter
        self.postfix_embeds = postfix_embeds  # bf16, moved to device on first forward
        self.strength = strength
        self._cached_extra = None
        self._cached_device = None
        self._cached_dtype = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_adapter, name)

    def _get_extra(self, device, dtype):
        if self._cached_device != device or self._cached_dtype != dtype:
            self._cached_extra = self.postfix_embeds.to(device=device, dtype=dtype)
            if self.strength != 1.0:
                self._cached_extra = self._cached_extra * self.strength
            self._cached_device = device
            self._cached_dtype = dtype
        return self._cached_extra

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if self.strength > 0:
            B = source_hidden_states.shape[0]
            extra = self._get_extra(source_hidden_states.device, source_hidden_states.dtype).unsqueeze(0).expand(B, -1, -1)
            source_hidden_states = torch.cat([source_hidden_states, extra], dim=1)
            if source_attention_mask is not None:
                extra_mask = torch.ones(B, self.postfix_embeds.shape[0], device=source_attention_mask.device, dtype=source_attention_mask.dtype)
                source_attention_mask = torch.cat([source_attention_mask, extra_mask], dim=-1)

        return self.original_adapter(source_hidden_states, target_input_ids, target_attention_mask, source_attention_mask)


def _wrap_clip_for_embedding_postfix(clip_model, postfix_embeds, strength):
    """Wrap the Qwen3 SDClipModel inside CLIP to inject postfix at embedding level."""
    qwen3_clip = clip_model.cond_stage_model.qwen3_06b
    original_process_tokens = qwen3_clip.process_tokens

    def patched_process_tokens(tokens, device):
        embeds, attention_mask, num_tokens, embeds_info = original_process_tokens(tokens, device)
        B = embeds.shape[0]
        extra = postfix_embeds.unsqueeze(0).expand(B, -1, -1).to(device=device)
        if strength != 1.0:
            extra = extra * strength
        embeds = torch.cat([embeds, extra], dim=1)
        extra_mask = torch.ones(B, postfix_embeds.shape[0], device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, extra_mask], dim=1)
        return embeds, attention_mask, num_tokens, embeds_info

    qwen3_clip.process_tokens = patched_process_tokens


class LoadAnimaPostfixEmbedding:
    """Load postfix tuning weights and apply them to the Anima model.

    For "cfg" mode files (containing postfix_pos + postfix_neg):
      - postfix_pos is applied to the model (adapter side, used for positive prompt)
      - A separate neg_clip output is provided for negative prompt encoding
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "postfix_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "mode": (["hidden", "embedding", "cfg"], {"default": "hidden"}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "CLIP")
    RETURN_NAMES = ("model", "clip", "neg_clip")
    FUNCTION = "load_postfix"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Loads Anima postfix tuning weights.\n"
        "hidden: injects into adapter cross-attention.\n"
        "embedding: injects into Qwen3 input embeddings (requires CLIP).\n"
        "cfg: dual pos/neg postfix — pos on model, neg on neg_clip output. "
        "Connect clip→positive encode, neg_clip→negative encode."
    )

    def load_postfix(self, model, postfix_name, strength, mode="hidden", clip=None):
        if strength == 0:
            return (model, clip, clip)

        postfix_path = folder_paths.get_full_path_or_raise("loras", postfix_name)
        weights = load_file(postfix_path)

        if mode == "cfg":
            # CFG mode: load dual pos/neg postfix
            postfix_pos = weights.get("postfix_pos")
            postfix_neg = weights.get("postfix_neg")
            if postfix_pos is None or postfix_neg is None:
                raise ValueError(f"CFG mode requires 'postfix_pos' and 'postfix_neg' keys in {postfix_name}")

            # Positive postfix on model (adapter side)
            model_clone = model.clone()
            adapter = model_clone.get_model_object("diffusion_model.llm_adapter")
            wrapper = PostfixLLMAdapterWrapper(adapter, postfix_pos, strength=strength)
            model_clone.add_object_patch("diffusion_model.llm_adapter", wrapper)

            # Negative postfix on a separate CLIP clone
            neg_clip = None
            if clip is not None:
                neg_clip = clip.clone()
                _wrap_clip_for_embedding_postfix(neg_clip, postfix_neg, strength)

            return (model_clone, clip, neg_clip)

        # Legacy single-postfix modes
        postfix_embeds = weights.get("postfix", weights.get("prefix"))
        if postfix_embeds is None:
            raise ValueError(f"No 'postfix' (or legacy 'prefix') key in {postfix_name}")

        if mode == "embedding":
            if clip is None:
                raise ValueError("Embedding mode requires CLIP input")
            clip = clip.clone()
            _wrap_clip_for_embedding_postfix(clip, postfix_embeds, strength)
            return (model, clip, clip)
        else:
            model_clone = model.clone()
            adapter = model_clone.get_model_object("diffusion_model.llm_adapter")
            wrapper = PostfixLLMAdapterWrapper(adapter, postfix_embeds, strength=strength)
            model_clone.add_object_patch("diffusion_model.llm_adapter", wrapper)
            return (model_clone, clip, clip)


NODE_CLASS_MAPPINGS = {
    "LoadAnimaPostfixEmbedding": LoadAnimaPostfixEmbedding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAnimaPostfixEmbedding": "Load Anima Postfix Embedding",
}
