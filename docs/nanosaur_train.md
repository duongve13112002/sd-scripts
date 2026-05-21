# Full Fine-Tuning Guide for NanoSaur using `nanosaur_train.py` / `nanosaur_train.py` を用いたNanoSaurモデルのフルファインチューニングガイド

This document explains how to perform full fine-tuning of a NanoSaur diffusion model using `nanosaur_train.py` in the `sd-scripts` repository.

## 1. Introduction / はじめに

`nanosaur_train.py` fine-tunes all (or selected) parameters of a NanoSaur diffusion transformer model using rectified flow matching. Unlike LoRA training (`nanosaur_train_network.py`), full fine-tuning updates the base model weights directly. Only the diffusion model weights are saved; the VAE and text encoder are not modified.

NanoSaur uses a DiT (Diffusion Transformer) architecture with:
- Rectified flow matching with logistic-normal timestep sampling
- SPRINT (Selective Path-drop guidance) for efficient unconditional forward passes
- Gemma3 270M text encoder with SentencePiece tokenizer
- DINOv3-based VAE with 16× spatial downscaling and 96 latent channels

**When to use full fine-tuning vs LoRA:**
- **Full fine-tuning** is appropriate for large-scale dataset training where you want to fundamentally alter model knowledge (new concept, style, or domain).
- **LoRA training** (`nanosaur_train_network.py`) is appropriate for smaller datasets or when you want a portable, composable adapter.

**Prerequisites:**

* The `sd-scripts` repository has been cloned and the Python environment is ready.
* A training dataset has been prepared. See the [Dataset Configuration Guide](./config_README-en.md).
* NanoSaur model files are available.

<details>
<summary>日本語</summary>

`nanosaur_train.py`は、NanoSaur拡散トランスフォーマーモデルのパラメータを全て（または選択的に）ファインチューニングするためのスクリプトです。LoRA学習（`nanosaur_train_network.py`）とは異なり、フルファインチューニングはベースモデルの重みを直接更新します。保存されるのは拡散モデルの重みのみです。VAEとテキストエンコーダーは変更されません。

NanoSaurは以下の特徴を持つDiT（Diffusion Transformer）アーキテクチャを使用しています：
- ロジスティック正規分布タイムステップサンプリングによるRectified Flow Matching
- 効率的な非条件前向きパスのためのSPRINT（選択的パスドロップガイダンス）
- SentencePieceトークナイザー付きGemma3 270Mテキストエンコーダー
- 16倍空間ダウンスケールと96潜在チャンネルを持つDINOv3ベースVAE

**フルファインチューニングとLoRAの使い分け:**
- **フルファインチューニング**は、モデルの知識を根本的に変更したい大規模データセット学習に適しています（新しいコンセプト、スタイル、ドメインなど）。
- **LoRA学習** (`nanosaur_train_network.py`) は、小規模データセットや、ポータブルで組み合わせ可能なアダプターが必要な場合に適しています。

**前提条件:**

* `sd-scripts`リポジトリのクローンとPython環境のセットアップが完了していること。
* 学習用データセットの準備が完了していること。（詳細は[データセット設定ガイド](./config_README-en.md)を参照してください）
* NanoSaurモデルファイルが準備できていること。
</details>

## 2. Preparation / 準備

The following files are required before starting training:

1. **Training script:** `nanosaur_train.py`
2. **NanoSaur diffusion model file:** `.safetensors` file for the base diffusion model.
3. **NanoSaur text encoder file:** `.safetensors` file containing Gemma3 weights and the SentencePiece model.
4. **NanoSaur VAE file:** `.safetensors` file for the VAE decoder.
5. **Dataset definition file (.toml):** Dataset settings in TOML format. (See the [Dataset Configuration Guide](./config_README-en.md).)

<details>
<summary>日本語</summary>

学習を開始する前に、以下のファイルが必要です。

1. **学習スクリプト:** `nanosaur_train.py`
2. **NanoSaur拡散モデルファイル:** ベースとなる拡散モデルの`.safetensors`ファイル。
3. **NanoSaurテキストエンコーダーファイル:** Gemma3の重みとSentencePieceモデルを含む`.safetensors`ファイル。
4. **NanoSaur VAEファイル:** VAEデコーダーの`.safetensors`ファイル。
5. **データセット定義ファイル (.toml):** 学習データセットの設定を記述したTOML形式のファイル。（詳細は[データセット設定ガイド](./config_README-en.md)を参照してください）
</details>

## 3. Running the Training / 学習の実行

### 3.1. Basic Example / 基本例

```bash
accelerate launch --mixed_precision bf16 nanosaur_train.py \
  --pretrained_model_name_or_path="nanosaur_diffusion_model.safetensors" \
  --text_encoder="nanosaur_text_encoder.safetensors" \
  --vae="nanosaur_vae_decoder.safetensors" \
  --dataset_config="my_nanosaur_dataset_config.toml" \
  --output_dir="./output" \
  --output_name="my_nanosaur_finetuned" \
  --save_model_as=safetensors \
  --learning_rate=5e-6 \
  --optimizer_type="AdamW" \
  --lr_scheduler="cosine" \
  --max_train_steps=10000 \
  --save_every_n_steps=1000 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents_to_disk \
  --cache_text_encoder_outputs \
  --blocks_to_swap=10
```

### 3.2. Per-Attribute Learning Rates / アトリビュートごとの学習率

`nanosaur_train.py` supports different learning rates for different parts of the model via `--attr_lr`. This allows you to freeze or slow down layers while training others more aggressively.

**Format:** `"attr_pattern=lr,attr_pattern2=lr2"` where `attr_pattern` is a substring of the parameter name.  
**Range notation:** `blocks.0-5=1e-5` is expanded to match `blocks.0`, `blocks.1`, ..., `blocks.5`.  
**Unmatched parameters** use the global `--learning_rate` value.

**Examples:**

```bash
# Fine-tune early blocks aggressively, later blocks gently, decoder separately
--attr_lr "blocks.0-5=1e-5,blocks.6-25=2e-6,dec_net=5e-6"

# Only fine-tune the decoder, freeze everything else (lr=0)
--attr_lr "dec_net=5e-6" --learning_rate=0

# Higher learning rate for cross-attention (text conditioning)
--attr_lr "attn.to_q=3e-5,attn.to_k=3e-5,attn.to_v=3e-5"
```

**NanoSaur parameter name patterns:**
| Pattern | Matches |
|---|---|
| `blocks.N` | Block N of the main DiT stack (26 blocks total, 0-indexed) |
| `blocks.0-5` | Blocks 0 through 5 (fast SPRINT blocks) |
| `blocks.6-27` | Blocks 6 through 27 (global + head SPRINT blocks) |
| `dec_net` | SimpleMLPAdaLN decoder network |
| `text_refine_blocks` | TextRefineBlocks (text conditioning refinement) |
| `t_embedder` | Timestep embedder |
| `y_embedder` | Text/caption embedder |
| `attn` | All attention layers (Q, K, V, output projections) |
| `ff` | All feed-forward layers |

<details>
<summary>日本語</summary>

`nanosaur_train.py`では`--attr_lr`を使用して、モデルの異なる部分に対して異なる学習率を設定できます。これにより、一部のレイヤーを固定または低速に学習させながら、他のレイヤーをより積極的に学習させることができます。

**書式:** `"attr_pattern=lr,attr_pattern2=lr2"` （`attr_pattern`はパラメータ名の部分文字列）  
**範囲表記:** `blocks.0-5=1e-5` は `blocks.0`、`blocks.1`、...、`blocks.5` に展開されます。  
**マッチしないパラメータ**はグローバルな `--learning_rate` の値が使用されます。

**例:**

```bash
# 初期ブロックを積極的に、後のブロックを穏やかに、デコーダーを別途
--attr_lr "blocks.0-5=1e-5,blocks.6-25=2e-6,dec_net=5e-6"

# デコーダーのみファインチューニング、その他は固定 (lr=0)
--attr_lr "dec_net=5e-6" --learning_rate=0
```

**NanoSaurパラメータ名パターン:**
| パターン | マッチする箇所 |
|---|---|
| `blocks.N` | DiTメインスタックのブロックN（合計26ブロック、0インデックス） |
| `blocks.0-5` | ブロック0〜5（SPRINTファーストブロック） |
| `blocks.6-27` | ブロック6〜27（グローバル+ヘッドSPRINTブロック） |
| `dec_net` | SimpleMLPAdaLNデコーダーネットワーク |
| `text_refine_blocks` | TextRefineBlocks（テキスト条件付けリファイン） |
| `t_embedder` | タイムステップエンベッダー |
| `y_embedder` | テキスト/キャプションエンベッダー |
| `attn` | 全アテンション層（Q、K、V、出力射影） |
| `ff` | 全フィードフォワード層 |
</details>

## 4. Key Options / 主要なコマンドライン引数の解説

### Model Options / モデル関連

* `--pretrained_model_name_or_path="<path>"` **required** – Path to the NanoSaur diffusion model `.safetensors` file.
* `--text_encoder="<path>"` **required** – Path to the NanoSaur text encoder `.safetensors` file.
* `--vae="<path>"` **required** – Path to the NanoSaur VAE `.safetensors` file.

### Training Options / 学習関連

* `--learning_rate=<float>` – Global learning rate. Recommended range: `1e-6` to `1e-5` for full fine-tuning.
* `--attr_lr="<string>"` – Per-attribute learning rate overrides. See [Section 3.2](#32-per-attribute-learning-rates--アトリビュートごとの学習率) for format and examples.
* `--max_train_steps=<integer>` – Total number of training steps.
* `--max_train_epochs=<integer>` – Total number of training epochs (alternative to `--max_train_steps`).
* `--save_every_n_steps=<integer>` / `--save_every_n_epochs=<integer>` – Checkpoint saving frequency.
* `--gradient_checkpointing` – Enable gradient checkpointing to reduce VRAM.

### NanoSaur Training Parameters / NanoSaur 学習パラメータ

* `--time_sampling_alpha=<float>` – Alpha for logistic-normal timestep sampling. Default: `2.0`. Higher values concentrate timesteps near `t=0.5`.
* `--sample_shift=<float>` – Timestep schedule shift for sample generation. Default: `4.0`.
* `--sample_cfg=<float>` – CFG guidance scale for sample generation. Default: `7.0`.
* `--sample_steps=<integer>` – Euler denoising steps for sample generation. Default: `40`.
* `--cfg_start=<float>` – CFG start fraction. Default: `0.03`.
* `--cfg_end=<float>` – CFG end fraction. Default: `0.80`.
* `--disable_sprint` – Disable SPRINT optimization during sample generation.

### Memory and Speed / メモリ・速度関連

* `--blocks_to_swap=<integer>` – Number of DiT blocks to offload to CPU. Reduces VRAM at the cost of training speed. Recommended: `10–18` for 24 GB cards.
* `--cache_text_encoder_outputs` – Cache Gemma3 outputs (since the text encoder is not trained in full fine-tuning, this is highly recommended).
* `--cache_text_encoder_outputs_to_disk` – Cache text encoder outputs to disk.
* `--cache_latents` / `--cache_latents_to_disk` – Cache VAE latents in memory / on disk.
* `--fp8_base` – Load model in FP8 precision.
* `--mixed_precision="bf16"` – **Recommended.** Use bf16 mixed precision throughout.

### Output / 出力関連

* `--output_dir="<path>"` – Directory to save checkpoints and the final model.
* `--output_name="<name>"` – Base name for output files (without extension).
* `--save_model_as=safetensors` – Save in safetensors format.

> **Note:** Only the diffusion model weights are saved. The text encoder and VAE are unchanged and should be reused as-is during inference.

<details>
<summary>日本語</summary>

### モデル関連

* `--pretrained_model_name_or_path="<path>"` **[必須]** – NanoSaur拡散モデルの`.safetensors`ファイルのパスを指定します。
* `--text_encoder="<path>"` **[必須]** – NanoSaurテキストエンコーダーの`.safetensors`ファイルのパスを指定します。
* `--vae="<path>"` **[必須]** – NanoSaur VAEの`.safetensors`ファイルのパスを指定します。

### 学習関連

* `--learning_rate=<float>` – グローバル学習率。フルファインチューニングの推奨範囲: `1e-6` 〜 `1e-5`。
* `--attr_lr="<string>"` – アトリビュートごとの学習率オーバーライド。書式と例は[セクション3.2](#32-per-attribute-learning-rates--アトリビュートごとの学習率)を参照してください。
* `--max_train_steps=<integer>` – 総学習ステップ数。
* `--max_train_epochs=<integer>` – 総学習エポック数（`--max_train_steps`の代替）。
* `--save_every_n_steps=<integer>` / `--save_every_n_epochs=<integer>` – チェックポイント保存頻度。
* `--gradient_checkpointing` – 勾配チェックポインティングを有効にしてVRAMを削減します。

### NanoSaur 学習パラメータ

* `--time_sampling_alpha=<float>` – ロジスティック正規分布タイムステップサンプリングのアルファ。デフォルト: `2.0`。値が大きいほどタイムステップが`t=0.5`付近に集中します。
* `--sample_shift=<float>` – サンプル生成のタイムステップスケジュールシフト。デフォルト: `4.0`。
* `--sample_cfg=<float>` – サンプル生成のCFGガイダンススケール。デフォルト: `7.0`。
* `--sample_steps=<integer>` – サンプル生成のオイラーノイズ除去ステップ数。デフォルト: `40`。
* `--cfg_start=<float>` – CFG開始割合。デフォルト: `0.03`。
* `--cfg_end=<float>` – CFG終了割合。デフォルト: `0.80`。
* `--disable_sprint` – サンプル生成中のSPRINT最適化を無効にします。

### メモリ・速度関連

* `--blocks_to_swap=<integer>` – CPUにオフロードするDiTブロック数。VRAMを節約しますが学習速度が低下します。24GB VRAMカードでは`10〜18`を推奨。
* `--cache_text_encoder_outputs` – Gemma3の出力をキャッシュします（フルファインチューニングではテキストエンコーダーは学習されないため、強く推奨します）。
* `--cache_text_encoder_outputs_to_disk` – テキストエンコーダー出力をディスクにキャッシュします。
* `--cache_latents` / `--cache_latents_to_disk` – VAEの潜在変数をメモリ/ディスクにキャッシュします。
* `--fp8_base` – FP8精度でモデルをロードします。
* `--mixed_precision="bf16"` – **推奨。** bf16混合精度を使用します。

### 出力関連

* `--output_dir="<path>"` – チェックポイントと最終モデルを保存するディレクトリ。
* `--output_name="<name>"` – 出力ファイルのベース名（拡張子なし）。
* `--save_model_as=safetensors` – safetensors形式で保存します。

> **注意:** 保存されるのは拡散モデルの重みのみです。テキストエンコーダーとVAEは変更されず、推論時にはそのまま再利用してください。
</details>

## 5. Multi-GPU Training / マルチGPU学習

NanoSaur full fine-tuning supports distributed training via `accelerate`:

```bash
accelerate launch --num_processes=4 --mixed_precision bf16 nanosaur_train.py \
  ...
```

For DeepSpeed support, configure `accelerate config` with DeepSpeed and add:

```bash
--deepspeed
```

<details>
<summary>日本語</summary>

NanoSaurのフルファインチューニングは`accelerate`を使った分散学習をサポートしています：

```bash
accelerate launch --num_processes=4 --mixed_precision bf16 nanosaur_train.py \
  ...
```

DeepSpeedのサポートには、`accelerate config`でDeepSpeedを設定した上で `--deepspeed` を追加してください。
</details>

## 6. Using the Fine-Tuned Model / 学習済みモデルの利用

When training finishes, the fine-tuned diffusion model is saved in `--output_dir` (e.g. `my_nanosaur_finetuned.safetensors`). Use it as a drop-in replacement for the original diffusion model:

```bash
python nanosaur_minimal_inference.py \
  --model my_nanosaur_finetuned.safetensors \
  --text_encoder nanosaur_text_encoder.safetensors \
  --vae nanosaur_vae_decoder.safetensors \
  --prompt "a photo of a cat" \
  --output output.png
```

The fine-tuned model is also compatible with ComfyUI NanoSaur nodes as a replacement for the base diffusion model.

<details>
<summary>日本語</summary>

学習が完了すると、ファインチューニングされた拡散モデルが `--output_dir` に保存されます（例: `my_nanosaur_finetuned.safetensors`）。元の拡散モデルの代替として使用できます：

```bash
python nanosaur_minimal_inference.py \
  --model my_nanosaur_finetuned.safetensors \
  --text_encoder nanosaur_text_encoder.safetensors \
  --vae nanosaur_vae_decoder.safetensors \
  --prompt "a photo of a cat" \
  --output output.png
```

ファインチューニングされたモデルは、ベース拡散モデルの代替としてComfyUI NanoSaurノードとも互換性があります。
</details>

## 7. Recommended Settings / 推奨設定

**Conservative fine-tuning (24 GB VRAM, safe for small datasets):**

```bash
accelerate launch --mixed_precision bf16 nanosaur_train.py \
  --pretrained_model_name_or_path nanosaur_diffusion_model.safetensors \
  --text_encoder nanosaur_text_encoder.safetensors \
  --vae nanosaur_vae_decoder.safetensors \
  --dataset_config dataset.toml \
  --output_dir output/ \
  --output_name nanosaur_ft \
  --learning_rate=3e-6 \
  --max_train_steps=5000 \
  --save_every_n_steps=1000 \
  --mixed_precision=bf16 \
  --gradient_checkpointing \
  --cache_latents_to_disk \
  --cache_text_encoder_outputs \
  --blocks_to_swap=12 \
  --optimizer_type=AdamW \
  --lr_scheduler=cosine
```

**Domain-specific fine-tuning with per-attribute LR:**

```bash
# Fine-tune the late DiT blocks and decoder more; keep early blocks conservative
--attr_lr "blocks.0-7=1e-6,blocks.8-25=5e-6,dec_net=1e-5" \
--learning_rate=1e-6
```

<details>
<summary>日本語</summary>

**保守的なファインチューニング（24GB VRAM、小規模データセットに安全）:**

```bash
accelerate launch --mixed_precision bf16 nanosaur_train.py \
  --pretrained_model_name_or_path nanosaur_diffusion_model.safetensors \
  --text_encoder nanosaur_text_encoder.safetensors \
  --vae nanosaur_vae_decoder.safetensors \
  --dataset_config dataset.toml \
  --output_dir output/ \
  --output_name nanosaur_ft \
  --learning_rate=3e-6 \
  --max_train_steps=5000 \
  --save_every_n_steps=1000 \
  --mixed_precision=bf16 \
  --gradient_checkpointing \
  --cache_latents_to_disk \
  --cache_text_encoder_outputs \
  --blocks_to_swap=12 \
  --optimizer_type=AdamW \
  --lr_scheduler=cosine
```

**アトリビュートごとのLRを使ったドメイン特化ファインチューニング:**

```bash
# 後期DiTブロックとデコーダーをより積極的に、初期ブロックは保守的に
--attr_lr "blocks.0-7=1e-6,blocks.8-25=5e-6,dec_net=1e-5" \
--learning_rate=1e-6
```

その他のオプションについては、スクリプトのヘルプ（`python nanosaur_train.py --help`）を参照してください。
</details>

## 8. Differences from `nanosaur_train_network.py` / `nanosaur_train_network.py` との違い

| Feature | `nanosaur_train.py` | `nanosaur_train_network.py` |
|---|---|---|
| Training scope | Full model weights | LoRA adapter weights only |
| File size (output) | Same as base model | Small (rank × layers × 2 × dtype) |
| VRAM requirement | Higher | Lower |
| Training speed | Slower (larger graph) | Faster |
| Composability | Not composable | Multiple LoRAs can be combined |
| Use case | Large-scale adaptation | Concept / style injection |
| ComfyUI | Drop-in model replacement | LoRA loader node |

<details>
<summary>日本語</summary>

| 機能 | `nanosaur_train.py` | `nanosaur_train_network.py` |
|---|---|---|
| 学習範囲 | フルモデルの重み | LoRAアダプターの重みのみ |
| ファイルサイズ（出力） | ベースモデルと同等 | 小さい（ランク × レイヤー × 2 × データ型） |
| VRAM要件 | 高い | 低い |
| 学習速度 | 遅い（グラフが大きい） | 速い |
| 組み合わせ可能性 | 不可 | 複数のLoRAを組み合わせ可能 |
| ユースケース | 大規模適応 | コンセプト/スタイル注入 |
| ComfyUI | モデルの直接置き換え | LoRAローダーノード |
</details>
