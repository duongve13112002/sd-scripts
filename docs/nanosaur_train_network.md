# LoRA Training Guide for NanoSaur using `nanosaur_train_network.py` / `nanosaur_train_network.py` を用いたNanoSaurモデルのLoRA学習ガイド

This document explains how to train LoRA (Low-Rank Adaptation) models for NanoSaur using `nanosaur_train_network.py` in the `sd-scripts` repository.

## 1. Introduction / はじめに

`nanosaur_train_network.py` trains additional networks such as LoRA for NanoSaur diffusion models. NanoSaur adopts a DiT (Diffusion Transformer) architecture with rectified flow matching and the SPRINT optimization (selective path dropping for unconditional guidance). It uses a Gemma3 270M text encoder and a DINOv3-based VAE with 16× spatial downscaling.

This guide assumes you already understand the basics of LoRA training. For common usage and options, see the [train_network.py guide](./train_network.md). Some parameters are similar to those in [`flux_train_network.py`](flux_train_network.md) and [`lumina_train_network.py`](lumina_train_network.md).

**Prerequisites:**

* The `sd-scripts` repository has been cloned and the Python environment is ready.
* A training dataset has been prepared. See the [Dataset Configuration Guide](./config_README-en.md).
* NanoSaur model files for training are available.

<details>
<summary>日本語</summary>

`nanosaur_train_network.py`は、NanoSaur拡散モデルに対してLoRAなどの追加ネットワークを学習させるためのスクリプトです。NanoSaurは、DiT (Diffusion Transformer) アーキテクチャを採用しており、Rectified Flow Matchingと、SPRINT (選択的パスドロップガイダンス) 最適化を使用します。テキストエンコーダーとしてGemma3 270Mを使用し、DINOv3ベースのVAEにより16倍の空間ダウンスケールを行います。

このガイドは、基本的なLoRA学習の手順を理解しているユーザーを対象としています。基本的な使い方や共通のオプションについては、`train_network.py`のガイドを参照してください。また一部のパラメータは [`flux_train_network.py`](flux_train_network.md) や [`lumina_train_network.py`](lumina_train_network.md) と同様のものがあるため、そちらも参考にしてください。

**前提条件:**

* `sd-scripts`リポジトリのクローンとPython環境のセットアップが完了していること。
* 学習用データセットの準備が完了していること。（詳細は[データセット設定ガイド](./config_README-en.md)を参照してください）
* 学習対象のNanoSaurモデルファイルが準備できていること。
</details>

## 2. Differences from `train_network.py` / `train_network.py` との違い

`nanosaur_train_network.py` is based on `train_network.py` but modified for NanoSaur. Main differences are:

* **Target models:** NanoSaur diffusion transformer models.
* **Model structure:** Uses a DiT (Transformer based) instead of U-Net, a Gemma3 270M text encoder with SentencePiece tokenizer (bundled inside the text encoder safetensors), and a 16× VAE (not the standard 8× used by SD/FLUX).
* **Arguments:** Separate options to specify the NanoSaur diffusion model (`--pretrained_model_name_or_path`), the text encoder (`--text_encoder`), and the VAE (`--vae`).
* **LoRA format:** Saved LoRA weights use ComfyUI-compatible key format (`diffusion_model.{path}.lora_{up|down}.weight`), compatible with ComfyUI NanoSaur nodes.
* **Incompatible arguments:** Stable Diffusion v1/v2 options such as `--v2`, `--v_parameterization`, and `--clip_skip` are not used.
* **NanoSaur-specific options:** Additional parameters for timestep sampling, sample generation guidance, SPRINT control, and block swapping.

<details>
<summary>日本語</summary>

`nanosaur_train_network.py`は`train_network.py`をベースに、NanoSaurモデルに対応するための変更が加えられています。主な違いは以下の通りです。

* **対象モデル:** NanoSaur拡散トランスフォーマーモデルを対象とします。
* **モデル構造:** U-Netの代わりにDiT (Transformerベース) を使用します。テキストエンコーダーとしてSentencePieceトークナイザー付きのGemma3 270Mを使用し（トークナイザーはテキストエンコーダーのsafetensors内に格納）、16倍VAE（SD/FLUXの標準的な8倍とは異なります）を使用します。
* **引数:** NanoSaur拡散モデル (`--pretrained_model_name_or_path`)、テキストエンコーダー (`--text_encoder`)、VAE (`--vae`) をそれぞれ個別に指定する引数があります。
* **LoRA形式:** 保存されるLoRAの重みはComfyUI互換のキー形式 (`diffusion_model.{path}.lora_{up|down}.weight`) を使用します。ComfyUI NanoSaurノードと互換性があります。
* **一部引数の非互換性:** Stable Diffusion v1/v2向けの引数（例: `--v2`, `--v_parameterization`, `--clip_skip`）はNanoSaur学習では使用されません。
* **NanoSaur特有の引数:** タイムステップサンプリング、サンプル生成ガイダンス、SPRINT制御、ブロックスワッピングに関する引数が追加されています。
</details>

## 3. Preparation / 準備

The following files are required before starting training:

1. **Training script:** `nanosaur_train_network.py`
2. **NanoSaur diffusion model file:** `.safetensors` file for the base diffusion model.
3. **NanoSaur text encoder file:** `.safetensors` file containing Gemma3 weights and the SentencePiece model.
4. **NanoSaur VAE file:** `.safetensors` file for the VAE decoder.
5. **Dataset definition file (.toml):** Dataset settings in TOML format. (See the [Dataset Configuration Guide](./config_README-en.md). In this document we use `my_nanosaur_dataset_config.toml` as an example.)

<details>
<summary>日本語</summary>

学習を開始する前に、以下のファイルが必要です。

1. **学習スクリプト:** `nanosaur_train_network.py`
2. **NanoSaur拡散モデルファイル:** ベースとなる拡散モデルの`.safetensors`ファイル。
3. **NanoSaurテキストエンコーダーファイル:** Gemma3の重みとSentencePieceモデルを含む`.safetensors`ファイル。
4. **NanoSaur VAEファイル:** VAEデコーダーの`.safetensors`ファイル。
5. **データセット定義ファイル (.toml):** 学習データセットの設定を記述したTOML形式のファイル。（詳細は[データセット設定ガイド](./config_README-en.md)を参照してください。このドキュメントでは`my_nanosaur_dataset_config.toml`を例として使用します。）
</details>

## 4. Running the Training / 学習の実行

Execute `nanosaur_train_network.py` from the terminal to start training. The overall command-line format is the same as `train_network.py`, but NanoSaur-specific options must be supplied.

Example command:

```bash
accelerate launch --mixed_precision bf16 nanosaur_train_network.py \
  --pretrained_model_name_or_path="nanosaur_diffusion_model.safetensors" \
  --text_encoder="nanosaur_text_encoder.safetensors" \
  --vae="nanosaur_vae_decoder.safetensors" \
  --dataset_config="my_nanosaur_dataset_config.toml" \
  --output_dir="./output" \
  --output_name="my_nanosaur_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_nanosaur \
  --network_dim=16 \
  --network_alpha=16 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW" \
  --lr_scheduler="cosine_with_restarts" \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents_to_disk \
  --cache_text_encoder_outputs \
  --blocks_to_swap=10
```

*(Write the command on one line or use `\` (Linux/macOS) or `^` (Windows) for line breaks.)*

<details>
<summary>日本語</summary>

学習は、ターミナルから`nanosaur_train_network.py`を実行することで開始します。基本的なコマンドラインの構造は`train_network.py`と同様ですが、NanoSaur特有の引数を指定する必要があります。

以下に基本的なコマンドライン実行例を示します。

```bash
accelerate launch --mixed_precision bf16 nanosaur_train_network.py \
  --pretrained_model_name_or_path="nanosaur_diffusion_model.safetensors" \
  --text_encoder="nanosaur_text_encoder.safetensors" \
  --vae="nanosaur_vae_decoder.safetensors" \
  --dataset_config="my_nanosaur_dataset_config.toml" \
  --output_dir="./output" \
  --output_name="my_nanosaur_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_nanosaur \
  --network_dim=16 \
  --network_alpha=16 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW" \
  --lr_scheduler="cosine_with_restarts" \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents_to_disk \
  --cache_text_encoder_outputs \
  --blocks_to_swap=10
```

※実際には1行で書くか、適切な改行文字（Linux/macOSでは `\`、Windowsでは `^`）を使用してください。
</details>

### 4.1. Explanation of Key Options / 主要なコマンドライン引数の解説

Besides the arguments explained in the [train_network.py guide](train_network.md), specify the following NanoSaur-specific options. For shared options (`--output_dir`, `--output_name`, etc.), see that guide.

#### Model Options / モデル関連

* `--pretrained_model_name_or_path="<path>"` **required** – Path to the NanoSaur diffusion model `.safetensors` file.
* `--text_encoder="<path>"` **required** – Path to the NanoSaur text encoder `.safetensors` file (contains Gemma3 weights and the SentencePiece tokenizer).
* `--vae="<path>"` **required** – Path to the NanoSaur VAE `.safetensors` file.

#### LoRA Network Options / LoRAネットワーク関連

* `--network_module=networks.lora_nanosaur` **required** – Use the NanoSaur LoRA module.
* `--network_dim=<integer>` – LoRA rank. Default `16`. Larger values capture more information but increase file size and VRAM usage.
* `--network_alpha=<float>` – LoRA alpha (scaling factor). Commonly set equal to `--network_dim`. Default `16`.
* `--network_train_unet_only` – If set, only trains LoRA for the diffusion model (DiT), skipping the Gemma3 text encoder. By default both diffusion model and text encoder are trained.
* `--network_args` – Additional network arguments. Supported keys:
  * `"target_modules=FlattenDiTBlock,TextRefineBlock"` – Comma-separated list of target module types for LoRA injection. Defaults to all trainable module types.

#### NanoSaur Training Parameters / NanoSaur 学習パラメータ

* `--time_sampling_alpha=<float>` – Alpha parameter for logistic-normal timestep sampling. Higher values concentrate sampling toward the midpoint (`t=0.5`). Default: `2.0`.
* `--sample_shift=<float>` – Timestep schedule shift used during sample image generation. Higher values bias toward high-signal (low-noise) steps. Default: `4.0`.
* `--sample_cfg=<float>` – CFG guidance scale for sample image generation. Default: `7.0`.
* `--sample_steps=<integer>` – Number of Euler denoising steps for sample image generation. Default: `40`.
* `--cfg_start=<float>` – Fraction of denoising steps from which CFG is applied during sampling. Default: `0.03`.
* `--cfg_end=<float>` – Fraction of denoising steps until which CFG is applied during sampling. Default: `0.80`.
* `--disable_sprint` – Disable SPRINT (path-drop guidance) optimization during sample generation. SPRINT alternates between full and sparse unconditional passes to speed up sampling. Disabled by default, meaning SPRINT is **active**.

#### Memory and Speed / メモリ・速度関連

* `--blocks_to_swap=<integer>` – Number of Transformer blocks to offload between CPU and GPU. Reduces VRAM usage at the cost of training speed. Recommended: `10–18` for 24 GB VRAM cards. Cannot be combined with `--cpu_offload_checkpointing`.
* `--cache_text_encoder_outputs` – Cache Gemma3 outputs to avoid repeated text encoder forward passes. Reduces VRAM and speeds up training when the text encoder is not being trained.
* `--cache_text_encoder_outputs_to_disk` – Cache text encoder outputs to disk. Automatically enables `--cache_text_encoder_outputs`.
* `--cache_latents` / `--cache_latents_to_disk` – Cache VAE latents in memory / on disk. Strongly recommended for multi-resolution bucket training.
* `--fp8_base` – Load the diffusion model in FP8 precision to save VRAM. Requires a compatible GPU (e.g., NVIDIA Ada or Hopper architecture).
* `--gradient_checkpointing` – Enable gradient checkpointing to reduce VRAM at the cost of slightly slower backward passes.

#### Incompatible or Deprecated Options / 非互換・非推奨の引数

* `--v2`, `--v_parameterization`, `--clip_skip` – Stable Diffusion v1/v2 options not used for NanoSaur.
* `--noise_offset` – Not applicable to NanoSaur's rectified flow matching.

<details>
<summary>日本語</summary>

[`train_network.py`のガイド](train_network.md)で説明されている引数に加え、以下のNanoSaur特有の引数を指定します。共通の引数については上記ガイドを参照してください。

#### モデル関連

* `--pretrained_model_name_or_path="<path>"` **[必須]** – NanoSaur拡散モデルの`.safetensors`ファイルのパスを指定します。
* `--text_encoder="<path>"` **[必須]** – NanoSaurテキストエンコーダーの`.safetensors`ファイルのパスを指定します（Gemma3の重みとSentencePieceトークナイザーが含まれています）。
* `--vae="<path>"` **[必須]** – NanoSaur VAEの`.safetensors`ファイルのパスを指定します。

#### LoRAネットワーク関連

* `--network_module=networks.lora_nanosaur` **[必須]** – NanoSaur LoRAモジュールを使用します。
* `--network_dim=<integer>` – LoRAランク。デフォルトは`16`。値が大きいほど多くの情報を捉えますが、ファイルサイズとVRAM使用量が増加します。
* `--network_alpha=<float>` – LoRAのアルファ（スケーリング係数）。通常は`--network_dim`と同じ値に設定します。デフォルトは`16`。
* `--network_train_unet_only` – 指定すると、拡散モデル（DiT）のLoRAのみを学習し、Gemma3テキストエンコーダーはスキップします。デフォルトでは拡散モデルとテキストエンコーダーの両方が学習されます。
* `--network_args` – 追加ネットワーク引数。サポートするキー:
  * `"target_modules=FlattenDiTBlock,TextRefineBlock"` – LoRA注入の対象となるモジュールタイプのカンマ区切りリスト。デフォルトは全ての学習可能なモジュールタイプです。

#### NanoSaur 学習パラメータ

* `--time_sampling_alpha=<float>` – ロジスティック正規分布タイムステップサンプリングのアルファパラメータ。値が大きいほど`t=0.5`付近にサンプリングが集中します。デフォルト: `2.0`。
* `--sample_shift=<float>` – サンプル画像生成中に使用されるタイムステップスケジュールシフト。値が大きいほど高信号（低ノイズ）のステップに偏ります。デフォルト: `4.0`。
* `--sample_cfg=<float>` – サンプル画像生成のCFGガイダンススケール。デフォルト: `7.0`。
* `--sample_steps=<integer>` – サンプル画像生成のオイラーノイズ除去ステップ数。デフォルト: `40`。
* `--cfg_start=<float>` – サンプリング中にCFGが適用される開始ステップの割合。デフォルト: `0.03`。
* `--cfg_end=<float>` – サンプリング中にCFGが適用される終了ステップの割合。デフォルト: `0.80`。
* `--disable_sprint` – サンプル生成中のSPRINT（パスドロップガイダンス）最適化を無効にします。SPRINTは全パスと疎な非条件パスを交互に使用してサンプリングを高速化します。デフォルトでは無効（SPRINTが有効）です。

#### メモリ・速度関連

* `--blocks_to_swap=<integer>` – CPUとGPU間でオフロードするTransformerブロック数。VRAMを節約できますが学習速度が低下します。24GB VRAMカードでは`10〜18`を推奨。`--cpu_offload_checkpointing`とは併用できません。
* `--cache_text_encoder_outputs` – Gemma3の出力をキャッシュして、繰り返しのテキストエンコーダー前向き計算を省略します。テキストエンコーダーを学習させない場合、VRAMを削減し学習を高速化します。
* `--cache_text_encoder_outputs_to_disk` – テキストエンコーダー出力をディスクにキャッシュします。`--cache_text_encoder_outputs`が自動的に有効になります。
* `--cache_latents` / `--cache_latents_to_disk` – VAEの潜在変数をメモリ/ディスクにキャッシュします。マルチ解像度バケット学習では強く推奨されます。
* `--fp8_base` – 拡散モデルをFP8精度でロードしてVRAMを節約します。対応するGPUが必要です（例: NVIDIA AdaまたはHopperアーキテクチャ）。
* `--gradient_checkpointing` – 勾配チェックポインティングを有効にして、バックワードパスが若干遅くなる代わりにVRAMを削減します。

#### 非互換・非推奨の引数

* `--v2`, `--v_parameterization`, `--clip_skip` – Stable Diffusion v1/v2向けの引数のため、NanoSaur学習では使用されません。
* `--noise_offset` – NanoSaurのRectified Flow Matchingには適用されません。
</details>

### 4.2. Multi-GPU Training / マルチGPU学習

NanoSaur training supports distributed training via `accelerate`. Configure the number of processes when launching:

```bash
accelerate launch --num_processes=4 --mixed_precision bf16 nanosaur_train_network.py \
  ...
```

Or configure using `accelerate config` beforehand.

<details>
<summary>日本語</summary>

NanoSauの学習は`accelerate`を使った分散学習をサポートしています。起動時にプロセス数を指定します：

```bash
accelerate launch --num_processes=4 --mixed_precision bf16 nanosaur_train_network.py \
  ...
```

または事前に`accelerate config`で設定することもできます。
</details>

### 4.3. Starting Training / 学習の開始

After setting the required arguments, run the command to begin training. The overall flow and how to check logs are the same as in the [train_network.py guide](train_network.md).

## 5. Using the Trained Model / 学習済みモデルの利用

When training finishes, a LoRA model file (e.g. `my_nanosaur_lora.safetensors`) is saved in the directory specified by `--output_dir`.

### ComfyUI

The saved LoRA uses ComfyUI-compatible key format (`diffusion_model.{path}.lora_up.weight` / `.lora_down.weight`). It can be loaded directly in ComfyUI with the appropriate NanoSaur LoRA loader node.

### Inference with this repository / このリポジトリでの推論

Use the `nanosaur_minimal_inference.py` script with the `--lora` option:

```bash
python nanosaur_minimal_inference.py \
  --model nanosaur_diffusion_model.safetensors \
  --text_encoder nanosaur_text_encoder.safetensors \
  --vae nanosaur_vae_decoder.safetensors \
  --lora my_nanosaur_lora.safetensors \
  --lora_scale 0.8 \
  --prompt "a photo of a cat sitting on a red sofa" \
  --output output.png
```

See `python nanosaur_minimal_inference.py --help` for all options.

<details>
<summary>日本語</summary>

学習が完了すると、指定した`--output_dir`にLoRAモデルファイル（例: `my_nanosaur_lora.safetensors`）が保存されます。

保存されるLoRAはComfyUI互換のキー形式（`diffusion_model.{path}.lora_up.weight` / `.lora_down.weight`）を使用しています。ComfyUIの適切なNanoSaur LoRAローダーノードで直接ロードできます。

当リポジトリの`nanosaur_minimal_inference.py`スクリプトで`--lora`オプションを指定して推論することも可能です。オプションの詳細は`python nanosaur_minimal_inference.py --help`で確認できます。
</details>

## 6. Others / その他

`nanosaur_train_network.py` shares many features with `train_network.py`, such as sample image generation (`--sample_prompts`, `--sample_every_n_steps`, etc.) and detailed optimizer settings. For these, see the [train_network.py guide](train_network.md) or run:

```bash
python nanosaur_train_network.py --help
```

### 6.1. Recommended Settings / 推奨設定

**Key Parameters:**
* `--network_module=networks.lora_nanosaur`
* `--network_dim=16` / `--network_alpha=16` (start here, increase for complex concepts)
* `--mixed_precision="bf16"`
* `--cache_latents_to_disk`
* `--cache_text_encoder_outputs` (when not training text encoder)
* `--gradient_checkpointing`

**VRAM Requirements (approximate):**
| Configuration | VRAM |
|---|---|
| Full precision, no swap | ~40 GB |
| bf16, gradient checkpointing | ~24 GB |
| bf16 + `--blocks_to_swap=10` | ~18 GB |
| bf16 + `--blocks_to_swap=18` | ~12 GB |

**Training the Text Encoder:**

By default, both the diffusion DiT and Gemma3 text encoder are trained with LoRA. To train only the DiT (faster, less VRAM), add `--network_train_unet_only`.

<details>
<summary>日本語</summary>

`nanosaur_train_network.py`には、サンプル画像生成（`--sample_prompts`、`--sample_every_n_steps`など）や詳細なオプティマイザ設定など、`train_network.py`と共通の機能が多くあります。これらについては、[`train_network.py`のガイド](train_network.md)やスクリプトのヘルプ（`python nanosaur_train_network.py --help`）を参照してください。

### 6.1. 推奨設定

**主要パラメータ:**
* `--network_module=networks.lora_nanosaur`
* `--network_dim=16` / `--network_alpha=16` (ここから始め、複雑なコンセプトには増加)
* `--mixed_precision="bf16"`
* `--cache_latents_to_disk`
* `--cache_text_encoder_outputs` (テキストエンコーダーを学習させない場合)
* `--gradient_checkpointing`

**VRAMの目安:**
| 設定 | VRAM |
|---|---|
| フル精度、スワップなし | ~40 GB |
| bf16、勾配チェックポインティング | ~24 GB |
| bf16 + `--blocks_to_swap=10` | ~18 GB |
| bf16 + `--blocks_to_swap=18` | ~12 GB |

**テキストエンコーダーの学習:**

デフォルトでは、拡散DiTとGemma3テキストエンコーダーの両方がLoRAで学習されます。DiTのみを学習する場合（高速化、VRAM削減）は `--network_train_unet_only` を追加してください。
</details>
