# Output Distillation (Anti Catastrophic-Forgetting) / 出力蒸留（破滅的忘却の抑制）

This document explains output distillation, a regularizer that reduces catastrophic forgetting when fine-tuning. It pulls the student's prediction toward the prediction of the frozen base model (the teacher), so the model keeps the base knowledge while still learning your data. This is the diffusion / flow-matching form of Learning without Forgetting (LwF).

It is available for **LoRA / network training** across all families (SD1.x, SDXL, SD3/3.5, FLUX.1, Lumina, HunyuanImage, Anima) and for **full fine-tuning** of SDXL, SD3/3.5, FLUX.1, Lumina, and Anima. It is disabled by default and adds no overhead unless a distillation weight is set.

<details>
<summary>日本語</summary>

このドキュメントでは、ファインチューニング時の破滅的忘却を抑制する正則化である出力蒸留について説明します。生徒（student）の予測を凍結したベースモデル（teacher）の予測へ近づけることで、データを学習しつつベースの知識を保持します。これは拡散 / フローマッチングにおける Learning without Forgetting (LwF) の形態です。

**LoRA / ネットワーク学習**（全ファミリー: SD1.x, SDXL, SD3/3.5, FLUX.1, Lumina, HunyuanImage, Anima）と、SDXL・SD3/3.5・FLUX.1・Lumina・Anima の **フルファインチューニング** で利用できます。デフォルトでは無効で、蒸留の重みを設定しない限りオーバーヘッドはありません。

</details>

## 1. How It Works / 仕組み

Each step the same noisy input, timestep, and conditioning are passed through both the student and the teacher, and a distillation term is added to the task loss:

```
L = L_task(student, real_target) + lambda(noise_level) * distance(student_pred, teacher_pred)
```

The teacher prediction is detached, so gradients flow only through the student.

- **LoRA / network training**: the teacher is the **same model with the adapter disabled** (`network.set_multiplier(0.0)`), so no second model copy is needed. Even though LoRA freezes the base weights, the adapter still changes the output, so distillation keeps the combined output close to the base.
- **Full fine-tuning**: there is no adapter, so a **separate frozen copy of the denoiser** is loaded as the teacher. Only the denoiser is duplicated; the VAE and text encoders are already frozen and shared.

`lambda` depends on the per-sample **noise level** (normalized to `[0, 1]`, where `1` is pure noise; flow-matching models use `sigmas`, DDPM models use `timesteps / num_train_timesteps`). Diffusion is coarse-to-fine: high noise carries global structure / concepts, low noise carries texture / style. Weighting distillation higher at high noise therefore **anchors concepts to the base while leaving detail/style learning free**.

<details>
<summary>日本語</summary>

各ステップで、同一のノイズ付き入力・タイムステップ・条件付けを student と teacher の両方に通し、タスク損失に蒸留項を加えます。

```
L = L_task(student, real_target) + lambda(noise_level) * distance(student_pred, teacher_pred)
```

teacher の予測は detach されるため、勾配は student のみを通ります。

- **LoRA / ネットワーク学習**: teacher は **アダプターを無効化した同一モデル**（`network.set_multiplier(0.0)`）であり、2 つ目のモデルコピーは不要です。LoRA はベースの重みを凍結しますが、アダプターは出力を変えるため、蒸留は合成出力をベースに近く保ちます。
- **フルファインチューニング**: アダプターがないため、teacher として **denoiser の凍結コピー** を別途ロードします。複製するのは denoiser のみで、VAE とテキストエンコーダは既に凍結・共有されています。

`lambda` はサンプルごとの **ノイズレベル**（`[0, 1]` に正規化、`1` が純ノイズ。フローマッチング系は `sigmas`、DDPM 系は `timesteps / num_train_timesteps` を使用）に依存します。拡散は粗→細であり、高ノイズは大域構造・概念を、低ノイズはテクスチャ・スタイルを担います。したがって高ノイズで蒸留を強くすると、**概念をベースに固定しつつ、細部・スタイルの学習は自由にできます**。

</details>

## 2. Command Line Arguments / コマンドライン引数

### Distillation weights (LoRA and full fine-tuning) / 蒸留の重み（LoRA・フルFT共通）

- `--distillation_weight_high`: Weight at high noise (noise level `1`). Anchors concepts/global structure to the base. `0` disables distillation (default: `0.0`).
- `--distillation_weight_low`: Weight at low noise (noise level `0`). Set lower than `high` to let the model learn detail/style freely (default: `0.0`).

The distance between the student and teacher predictions reuses the task `--loss_type` (and the same Huber threshold), so the two loss terms are always consistent; there is no separate distillation loss-type option. Distillation is enabled when either weight is greater than `0`. Use equal `high` and `low` for a constant weight across noise levels.

### Full fine-tune teacher options / フルFT の teacher オプション

These only apply to full fine-tuning (LoRA reuses the adapter-disabled model and ignores them):

- `--distillation_teacher_path`: Checkpoint for the teacher (default: `--pretrained_model_name_or_path`, i.e. the base being fine-tuned).
- `--distillation_teacher_fp8`: Load the teacher in fp8 to save VRAM (generic per-Linear fp8). Ignored for models that do not support fp8 (Anima, SDXL U-Net).
- `--distillation_teacher_blocks_to_swap`: Block-swap this many teacher blocks to CPU to save VRAM (transformer models: FLUX, SD3, Lumina, Anima; ignored for the SDXL U-Net) (default: `0`).

<details>
<summary>日本語</summary>

### 蒸留の重み（LoRA・フルFT共通）

- `--distillation_weight_high`: 高ノイズ（ノイズレベル `1`）での重み。概念・大域構造をベースに固定します。`0` で蒸留を無効化（デフォルト: `0.0`）。
- `--distillation_weight_low`: 低ノイズ（ノイズレベル `0`）での重み。`high` より小さくすると細部・スタイルを自由に学習できます（デフォルト: `0.0`）。

student と teacher の予測間の距離はタスクの `--loss_type`（および同じ Huber しきい値）を再利用するため、2 つの損失項は常に一貫します。蒸留専用の loss-type オプションはありません。

いずれかの重みが `0` より大きいとき蒸留が有効になります。`high` と `low` を同じ値にするとノイズレベルに依らない一定の重みになります。

### フルFT の teacher オプション

フルファインチューニングにのみ適用されます（LoRA はアダプター無効化モデルを使うため無視されます）。

- `--distillation_teacher_path`: teacher のチェックポイント（デフォルト: `--pretrained_model_name_or_path`、すなわちファインチューニング対象のベース）。
- `--distillation_teacher_fp8`: VRAM 節約のため teacher を fp8 でロード（汎用の per-Linear fp8）。fp8 非対応のモデル（Anima, SDXL U-Net）では無視されます。
- `--distillation_teacher_blocks_to_swap`: VRAM 節約のため teacher のブロックを指定数だけ CPU にスワップ（transformer 系: FLUX, SD3, Lumina, Anima。SDXL U-Net では無視）（デフォルト: `0`）。

</details>

## 3. Usage Example / 使用例

LoRA / network training (teacher is free — the adapter-disabled model):

```bash
accelerate launch --mixed_precision bf16 flux_train_network.py \
  --pretrained_model_name_or_path path/to/model.safetensors \
  --dataset_config path/to/config.toml \
  --output_dir path/to/output --output_name my_lora \
  (... other training args ...) \
  --distillation_weight_high 1.0 \
  --distillation_weight_low 0.0
```

Full fine-tuning (a frozen teacher copy is loaded; offload it to save VRAM):

```bash
accelerate launch --mixed_precision bf16 flux_train.py \
  --pretrained_model_name_or_path path/to/model.safetensors \
  --dataset_config path/to/config.toml \
  --output_dir path/to/output --output_name my_ft \
  (... other training args ...) \
  --distillation_weight_high 1.0 \
  --distillation_weight_low 0.0 \
  --distillation_teacher_fp8 \
  --distillation_teacher_blocks_to_swap 16
```

On Windows Command Prompt, use `^` instead of `\` for line continuation.

<details>
<summary>日本語</summary>

LoRA 学習では teacher は無償（アダプター無効化モデル）です。フルファインチューニングでは凍結 teacher コピーがロードされるため、VRAM 節約にオフロード（`--distillation_teacher_fp8` / `--distillation_teacher_blocks_to_swap`）を併用してください。Windows のコマンドプロンプトでは行末の継続文字に `\` ではなく `^` を使用してください。

</details>

## 4. Choosing the Weights / 重みの選び方

- Start with `--distillation_weight_high 1.0 --distillation_weight_low 0.0`: strong anchoring at high noise (keep concepts), free learning at low noise (adapt style/detail). This is the recommended default for "keep concepts, change style".
- Use a small constant weight (e.g. `high = low = 0.1`–`0.5`) if you want a gentle global regularizer.
- Larger weights keep more of the base behaviour but slow target adaptation. The printed loss value increases when distillation is on — this is expected (the optimizer minimizes the sum); log task and distillation terms separately if you want to monitor them.

<details>
<summary>日本語</summary>

- まず `--distillation_weight_high 1.0 --distillation_weight_low 0.0` から: 高ノイズで強く固定（概念を保持）、低ノイズで自由に学習（スタイル・細部を適応）。「概念は保持しつつスタイルを変える」用途の推奨デフォルトです。
- 緩やかな全体正則化が欲しい場合は小さい一定値（例: `high = low = 0.1`〜`0.5`）。
- 重みを大きくするとベースの挙動をより保持しますが、ターゲットへの適応は遅くなります。蒸留を有効にすると表示される損失値は増えますが、これは想定どおりです（オプティマイザは和を最小化します）。監視したい場合はタスク項と蒸留項を分けてログしてください。

</details>

## 5. Multi-GPU and Single-GPU / マルチ GPU とシングル GPU

Distillation works on both single-GPU and multi-GPU (DDP) runs.

- **LoRA**: the teacher is the DDP-wrapped model run with the adapter disabled under `no_grad`. There is no backward through it and no collective operation, so DDP is unaffected (this is the same mechanism as the existing differential output preservation).
- **Full fine-tuning**: the teacher is a separate frozen module that is **not** wrapped by `accelerate.prepare`; it is placed per rank and its forward is local with **no collective operation**. The distillation gradient flows through the student (DDP-wrapped), which all-reduces normally. Each rank holds its own teacher copy.

<details>
<summary>日本語</summary>

蒸留はシングル GPU・マルチ GPU（DDP）のどちらでも動作します。

- **LoRA**: teacher は DDP ラップされたモデルをアダプター無効化・`no_grad` で実行したものです。逆伝播も集合通信もないため DDP に影響しません（既存の differential output preservation と同じ仕組み）。
- **フルファインチューニング**: teacher は `accelerate.prepare` で **ラップしない** 独立した凍結モジュールで、各ランクに配置され、その forward はローカルで **集合通信を含みません**。蒸留の勾配は（DDP ラップされた）student を通り、通常どおり all-reduce されます。各ランクは自身の teacher コピーを保持します。

</details>

## 6. Notes and Limitations / 注意事項と制限

- **VRAM (full fine-tuning)**: the frozen teacher is an extra copy of the denoiser (about `1x` the transformer). Use `--distillation_teacher_fp8` and/or `--distillation_teacher_blocks_to_swap` for large models on limited VRAM. For a 12B-class model on a 24GB card, full fine-tuning with a live teacher likely needs fp8 and/or offload.
- **fp8 teacher** is a generic per-Linear quantization, an approximation chosen to save VRAM for a no-grad teacher; it is not the model's exact per-layer fp8 recipe. It is disabled for Anima and the SDXL U-Net.
- This is a separate term from the existing per-sample differential output preservation (DOP); both can be enabled, in which case the teacher runs twice.
- Combine with replay (mixing a small subset of the original data via multi-subset dataset config with `num_repeats`) and EMA for the strongest knowledge retention.

<details>
<summary>日本語</summary>

- **VRAM（フルFT）**: 凍結 teacher は denoiser の追加コピー（transformer のおよそ `1x`）です。大きなモデルを限られた VRAM で扱う場合は `--distillation_teacher_fp8` や `--distillation_teacher_blocks_to_swap` を使ってください。24GB で 12B 級のモデルをライブ teacher 付きフルFT する場合、fp8 やオフロードが必要になる可能性が高いです。
- **fp8 teacher** は汎用の per-Linear 量子化で、no-grad の teacher の VRAM を節約するための近似です。モデル本来の層別 fp8 レシピではありません。Anima と SDXL U-Net では無効です。
- これは既存のサンプル単位 differential output preservation (DOP) とは別の項です。両方を有効にでき、その場合 teacher は 2 回実行されます。
- 最大の知識保持には、リプレイ（multi-subset のデータセット設定で `num_repeats` を使い元データの小さなサブセットを混ぜる）や EMA と併用してください。

</details>

## 7. Recommended Settings / 推奨設定

```bash
# keep concepts, adapt style/detail (the distance follows the task --loss_type)
--distillation_weight_high 1.0 \
--distillation_weight_low 0.0

# full fine-tuning of a large model, save teacher VRAM
--distillation_teacher_fp8 \
--distillation_teacher_blocks_to_swap 16
```

<details>
<summary>日本語</summary>

上段は「概念を保持しつつスタイル・細部を適応」する設定、下段は大きなモデルのフルFT で teacher の VRAM を節約する設定です。

</details>

## Additional Resources / 追加リソース

- [EMA (Exponential Moving Average)](ema.md)
- [Anima LoRA Training Guide](anima_train_network.md)
- [Learning without Forgetting](https://arxiv.org/abs/1606.09282)
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)
