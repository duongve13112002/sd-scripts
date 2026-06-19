# Anima DiT Adapter Training (`anima_train_adapter.py`) / Anima DiT アダプター学習

This document explains `anima_train_adapter.py`, a **standalone** script that trains a structured **adapter** on top of a **frozen** Anima DiT. Unlike LoRA (generic low-rank deltas on Linear layers, run through the shared training pipeline), this script inserts purpose-built conditioning/enhancement modules into the DiT blocks and trains only those modules; the base DiT, VAE, text encoder, and CLIP are all frozen.

It is a self-contained script (its own dataset, sampler, collate, and training loop) and does **not** use the strategy pattern, the caching strategies, EMA, or output distillation used by the main training scripts.

<details>
<summary>日本語</summary>

このドキュメントでは、**凍結した** Anima DiT の上に構造化された **アダプター** を学習する **スタンドアロン** スクリプト `anima_train_adapter.py` を説明します。LoRA（共有学習パイプラインを通る、Linear への汎用的な低ランク差分）とは異なり、本スクリプトは専用の条件付け・強調モジュールを DiT ブロックへ挿入し、それらのモジュールのみを学習します。ベースの DiT、VAE、テキストエンコーダ、CLIP はすべて凍結されます。

これは自己完結型のスクリプト（独自のデータセット・サンプラー・collate・学習ループ）であり、メインの学習スクリプトが使うストラテジパターン・キャッシュ戦略・EMA・出力蒸留は **使用しません**。

</details>

## 1. How It Works / 仕組み

The DiT base is loaded and frozen (`requires_grad_(False).eval()`); the same is done for Qwen3 (text), the VAE, and (optionally) CLIP. An `AnimaDiTAdapter` is built and **patches the DiT blocks** (`patch_dit_blocks`) so that the adapter modules run inside each block's forward. Only the adapter parameters are trained, with the **Prodigy** optimizer.

All adapter output projections use a zero-initialized layer (`ZeroLinear`), so at the start of training the adapter is an **identity** and does not disturb the base; it then learns gradually. Training uses flow matching (`FlowMatchEulerDiscreteScheduler` with `compute_loss_weighting_for_anima`), the same loss family as `anima_train.py`. Pixels are VAE-encoded (optionally pre-computed/cached), and a resolution-grouped sampler batches similar sizes together. Only the adapter `state_dict` is saved (safetensors), so the resulting files are small and loaded back via `--resume`.

<details>
<summary>日本語</summary>

DiT ベースはロードして凍結します（`requires_grad_(False).eval()`）。Qwen3（テキスト）・VAE・（任意で）CLIP も同様に凍結します。`AnimaDiTAdapter` を構築し、**DiT ブロックをパッチ**（`patch_dit_blocks`）して、各ブロックの forward 内でアダプターモジュールが動くようにします。学習されるのはアダプターのパラメータのみで、**Prodigy** オプティマイザを使います。

アダプターの出力投影はすべてゼロ初期化層（`ZeroLinear`）を使うため、学習開始時はアダプターが **恒等写像** となりベースを乱さず、その後徐々に学習します。学習はフローマッチング（`FlowMatchEulerDiscreteScheduler` + `compute_loss_weighting_for_anima`）で、`anima_train.py` と同じ損失系です。ピクセルは VAE でエンコード（任意で事前計算・キャッシュ）し、解像度グループ化サンプラーが近いサイズをまとめてバッチ化します。保存されるのはアダプターの `state_dict`（safetensors）のみで、ファイルは小さく、`--resume` で読み戻せます。

</details>

## 2. Adapter Modules / アダプターモジュール

The adapter is a collection of modules, each enabled by its own flag:

| Module | Flag | Role |
|---|---|---|
| `ModTextAdapter` | `--enable_mod_text` (on) | Modulation-style text enhancement |
| `SemanticScaleAdapter` | `--enable_semantic_scale` (on) | Global semantic scaling |
| `LocalConvAdapter` | `--enable_local_conv` (on) | Local conv detail enhancement |
| `StyleCrossAttention` | `--enable_style_attn` | Inject style from CLIP image-style embeddings (cross-attention) |
| `EdgeDetailConv` | `--enable_edge_detail` | Edge / line-detail enhancement |
| `SubjectCrossAttention` | `--enable_subject_attn` | Inject subject from text embeddings (prevents style overwhelming the subject) |
| `ContrastModAdapter` | `--enable_contrast_mod` | Contrast tuning |
| `ColorTuneAdapter` | `--enable_color_tune` | Color tuning |

Each module has layer-count / kernel / start-layer / head options (e.g. `--local_conv_layers`, `--style_attn_layers`, `--subject_attn_start_layer`).

<details>
<summary>日本語</summary>

アダプターは複数のモジュールの集合で、それぞれ専用フラグで有効化します（上表参照）。各モジュールには層数・カーネル・開始層・ヘッド数などのオプション（例: `--local_conv_layers`, `--style_attn_layers`, `--subject_attn_start_layer`）があります。

</details>

## 3. Command-Line Arguments / コマンドライン引数

This script has its **own** argument set (different names from the main scripts).

### Paths / パス

- `--train_data_dir` (required): image + caption directory.
- `--anima_model_path` (required): Anima DiT base checkpoint.
- `--qwen_model_path` (required): Qwen3 text encoder.
- `--vae_model_path` (required): VAE.
- `--clip_model_path`: CLIP model (default `openai/clip-vit-large-patch14`), used only with `--enable_style_attn`.
- `--t5_tokenizer_path`: T5 tokenizer (default: auto-extracted from the DiT).
- `--output_dir`, `--output_name`, `--resume`.

### Training / 学習

- `--train_batch_size` (default `8`), `--max_train_epochs` (default `10`), `--gradient_accumulation_steps`, `--mixed_precision {no,fp16,bf16}` (default `bf16`), `--seed`.
- `--use_cosine_scheduler`, `--lr_warmup_steps` (the optimizer is Prodigy, which adapts the learning rate).
- `--timestep_sampling {uniform,sigmoid}`, `--weighting_scheme`, `--logit_mean`, `--logit_std`, `--discrete_flow_shift`, `--sigmoid_scale`, `--mode_scale`, `--ip_noise_gamma`.
- `--caption_dropout_rate`, `--shuffle_caption`, `--val_split`, `--save_every_n_epochs`.
- `--max_width`, `--max_height` (default `1024`).

### Memory / VAE / 省メモリ・VAE

- `--gradient_checkpointing`, `--xformers`, `--split_attn`, `--max_data_loader_n_workers`.
- `--enable_precompute_embeddings` (default on; disable with `--no-enable_precompute_embeddings`) pre-computes latent/style caches.
- `--vae_chunk_size`, `--vae_disable_cache`, `--vae_precompute_batch_size`, `--vae_encode_batch_size`, `--vae_precompute_device {auto,cuda,cpu}`, `--clip_precompute_batch_size`, `--skip_failed_precompute`.

### Module enable / sizing / モジュールの有効化・サイズ

- Enable flags as listed in Section 2, plus dimensions: `--dit_hidden_size` (default `2048`), `--num_blocks` (default `28`), `--text_embed_dim`, `--style_dim`, and per-module layer/kernel/head/start-layer options.

<details>
<summary>日本語</summary>

本スクリプトは **独自** の引数セットを持ちます（メインスクリプトとは名前が異なります）。上記の英語セクションを参照してください。要点: 必須はパス系（`--train_data_dir`, `--anima_model_path`, `--qwen_model_path`, `--vae_model_path`）。オプティマイザは学習率を自動調整する Prodigy です。各モジュールはセクション 2 のフラグで有効化し、層数・カーネル・ヘッド・開始層などのサイズ引数を持ちます。

</details>

## 4. Usage Example / 使用例

```bash
accelerate launch --mixed_precision bf16 anima_train_adapter.py \
  --train_data_dir path/to/images_with_captions \
  --anima_model_path path/to/anima_dit.safetensors \
  --qwen_model_path path/to/qwen3 \
  --vae_model_path path/to/vae.safetensors \
  --output_dir path/to/output --output_name anima_adapter \
  --train_batch_size 8 --max_train_epochs 10 \
  --gradient_checkpointing \
  --enable_style_attn --enable_subject_attn --enable_edge_detail
```

On Windows Command Prompt, use `^` instead of `\` for line continuation. Enabling `--enable_style_attn` requires a CLIP model (downloaded if `--clip_model_path` is a name).

<details>
<summary>日本語</summary>

Windows のコマンドプロンプトでは行末の継続文字に `\` ではなく `^` を使用してください。`--enable_style_attn` を有効にすると CLIP モデルが必要です（`--clip_model_path` が名前の場合はダウンロードされます）。

</details>

## 5. Notes and Limitations / 注意事項と制限

- **Standalone script.** It does not share code with the main training pipeline: no strategy pattern, no shared caching strategies, no EMA, and no output distillation. Those features (e.g. `--ema`, `--distillation_*`) are **not** available here.
- **Adapter vs LoRA.** Use this when you want the specific levers above (style from a CLIP image, subject injection, edge/detail/color/contrast tuning). For general concept/style adaptation inside the standard pipeline, use LoRA (`anima_train_network.py`).
- **Output.** Only the adapter weights are saved; you need the adapter plus the base DiT at inference. Resume with `--resume path/to/adapter.safetensors`.
- **Optimizer.** Prodigy is used (adaptive learning rate); there is no `--learning_rate` argument.
- This file is kept verbatim from its original source and retains its original (Chinese) inline comments.

<details>
<summary>日本語</summary>

- **スタンドアロン。** メイン学習パイプラインとコードを共有しません（ストラテジパターン・共有キャッシュ戦略・EMA・出力蒸留なし）。`--ema` や `--distillation_*` などはここでは **使えません**。
- **アダプター vs LoRA。** 上記の特定の制御（CLIP 画像からのスタイル、主体注入、エッジ/細部/色/コントラスト調整）が欲しいときに使います。標準パイプライン内での一般的な概念・スタイル適応には LoRA（`anima_train_network.py`）を使ってください。
- **出力。** アダプターの重みのみが保存されます。推論にはアダプターとベース DiT の両方が必要です。再開は `--resume path/to/adapter.safetensors`。
- **オプティマイザ。** Prodigy（学習率自動調整）を使用し、`--learning_rate` 引数はありません。
- 本ファイルは元のソースをそのまま保持しており、元の（中国語の）インラインコメントが残っています。

</details>

## Additional Resources / 追加リソース

- [Anima LoRA Training Guide](anima_train_network.md)
- [`torch.compile` for Anima Training](anima_torch_compile.md)
- [EMA](ema.md)
- [Output Distillation](distillation.md)
