# Improvements made (Anima LoRA training)

- **Dynamic token padding (Qwen3 + T5):** switched tokenization to pad to the longest sequence in the batch instead of always padding to `max_length`, and **trimmed cached TE outputs** to real mask length to reduce cache size + training compute (`library/strategy_anima.py`).
- **Optional LLM-adapter output caching:** added `--cache_llm_adapter_outputs` to cache adapter outputs (`crossattn_emb`) alongside TE cache and **skip running the adapter during training** (requires `--cache_text_encoder_outputs`; incompatible with `--network_args "train_llm_adapter=True"`) (`library/anima_train_utils.py`, `library/strategy_anima.py`, `library/anima_utils.py`, `anima_train_network.py`, docs in `docs/anima_train_network.md`).
- **Lower per-step overhead:** reuse a cached `padding_mask` tensor (avoid allocating every step) and avoid setting `requires_grad` on cached text conditions when not training the text encoder (`anima_train_network.py`).
- **Padding-mask concat efficiency:** avoid unnecessary resize imports and use `expand` instead of `repeat` when concatenating the padding mask (less memory traffic) (`library/anima_models.py`).
- **Input pipeline knobs:** added `--dataloader_pin_memory` and `--dataloader_prefetch_factor`, and ensured `persistent_workers` only when `num_workers>0` (`library/train_util.py`, `train_network.py`).

Notes:
- After changing TE cache format/behavior, delete old `*_anima_te.npz` files and re-cache.
- Running `anima_train_network.py` requires a training environment with `torchvision` installed (this container was missing it).
