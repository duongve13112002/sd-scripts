# EMA (Exponential Moving Average) for Training / 学習における EMA（指数移動平均）

This document explains the EMA feature, which maintains a smoothed copy of the trained weights (a moving average over training steps). The EMA weights are often more stable and generalize better than the final training-step weights, so they are usually preferred for inference.

EMA is available across **all model families** for both **LoRA / network training** (`*_train_network.py`: SD1.x, SDXL, SD3/3.5, FLUX.1, Lumina, HunyuanImage, Anima) and **full fine-tuning** (`*_train.py`: SDXL, SD3/3.5, FLUX.1, Lumina, Anima). It is disabled by default and adds no overhead unless `--ema` is set.

<details>
<summary>日本語</summary>

このドキュメントでは、学習した重みの平滑化されたコピー（学習ステップ間の移動平均）を保持する EMA 機能について説明します。EMA の重みは最終ステップの重みより安定し、汎化性能が高いことが多いため、推論には通常 EMA を使うのが望ましいです。

EMA は **すべてのモデルファミリー** で、**LoRA / ネットワーク学習**（`*_train_network.py`: SD1.x, SDXL, SD3/3.5, FLUX.1, Lumina, HunyuanImage, Anima）と **フルファインチューニング**（`*_train.py`: SDXL, SD3/3.5, FLUX.1, Lumina, Anima）の両方で利用できます。デフォルトでは無効で、`--ema` を指定しない限りオーバーヘッドはありません。

</details>

## 1. How It Works / 仕組み

After each optimizer step, EMA updates a shadow copy of the trainable parameters:

```
shadow = decay * shadow + (1 - decay) * current_weights
```

For LoRA training the shadow covers the network (adapter) parameters; for full fine-tuning it covers the trainable denoiser parameters. When a checkpoint is saved, an additional EMA checkpoint is written with an `ema_` filename prefix (e.g. `ema_model-step00010.safetensors`); it has the same format as the regular checkpoint and can be used directly for inference. With `--ema_sample`, sample images are also generated from the EMA weights and saved with an `_ema` filename suffix.

<details>
<summary>日本語</summary>

各オプティマイザステップの後、EMA は学習対象パラメータのシャドウコピーを次式で更新します。

```
shadow = decay * shadow + (1 - decay) * current_weights
```

LoRA 学習ではシャドウはネットワーク（アダプター）パラメータを対象とし、フルファインチューニングでは学習対象の denoiser パラメータを対象とします。チェックポイント保存時には、ファイル名に `ema_` プレフィックスを付けた EMA チェックポイント（例: `ema_model-step00010.safetensors`）が追加で書き出されます。形式は通常のチェックポイントと同じで、そのまま推論に使えます。`--ema_sample` を指定すると、EMA の重みからもサンプル画像が生成され、`_ema` サフィックス付きで保存されます。

</details>

## 2. Command Line Arguments / コマンドライン引数

### Basic arguments / 基本的な引数

- `--ema`: Enable EMA. Saves an `ema_`-prefixed checkpoint alongside each regular checkpoint.
- `--ema_decay`: Decay rate (default: `0.9999`). Higher values are smoother but adapt more slowly. Typical range `0.999`–`0.99999`.
- `--ema_device`: Device for the shadow parameters, `cuda` or `cpu` (default: `cuda`). `cpu` saves GPU VRAM (the shadow uses about as much memory as the trained parameters) at the cost of slower updates.
- `--ema_use_num_updates`: Warm up the decay early in training using `min(decay, (1 + num_updates) / (10 + num_updates))`.
- `--ema_sample`: Also generate sample images from the EMA weights (saved with an `_ema` suffix). Requires `--ema`.
- `--ema_resume_path`: Path to an EMA checkpoint to resume the EMA state from a previous run.

### Experimental arguments (single-GPU only) / 実験的な引数（シングル GPU のみ）

- `--ema_use_feedback`: Feed the EMA result back into the training weights. **Rejected under multi-GPU.**
- `--ema_param_multiplier`: Multiply parameters each EMA update step (default: `1.0`, no effect). **Rejected under multi-GPU when not `1.0`.**

<details>
<summary>日本語</summary>

### 基本的な引数

- `--ema`: EMA を有効にする。通常のチェックポイントと並べて `ema_` プレフィックス付きチェックポイントを保存します。
- `--ema_decay`: 減衰率（デフォルト: `0.9999`）。値が大きいほど平滑ですが適応は遅くなります。一般的には `0.999`〜`0.99999`。
- `--ema_device`: シャドウパラメータのデバイス。`cuda` または `cpu`（デフォルト: `cuda`）。`cpu` は GPU VRAM を節約できます（シャドウは学習対象パラメータと同程度のメモリを使用）が、更新は遅くなります。
- `--ema_use_num_updates`: 学習初期の減衰を `min(decay, (1 + num_updates) / (10 + num_updates))` でウォームアップします。
- `--ema_sample`: EMA の重みからもサンプル画像を生成します（`_ema` サフィックス付きで保存）。`--ema` が必要です。
- `--ema_resume_path`: 以前の学習の EMA 状態を再開するための EMA チェックポイントへのパス。

### 実験的な引数（シングル GPU のみ）

- `--ema_use_feedback`: EMA の結果を学習中の重みにフィードバックします。**マルチ GPU では拒否されます。**
- `--ema_param_multiplier`: EMA 更新ごとにパラメータを乗算します（デフォルト: `1.0`、効果なし）。**`1.0` 以外の場合、マルチ GPU では拒否されます。**

</details>

## 3. Usage Example / 使用例

LoRA / network training:

```bash
accelerate launch --mixed_precision bf16 flux_train_network.py \
  --pretrained_model_name_or_path path/to/model.safetensors \
  --dataset_config path/to/config.toml \
  --output_dir path/to/output --output_name my_lora \
  (... other training args ...) \
  --ema --ema_decay 0.9999 --ema_device cpu --ema_sample
```

Full fine-tuning:

```bash
accelerate launch --mixed_precision bf16 flux_train.py \
  --pretrained_model_name_or_path path/to/model.safetensors \
  --dataset_config path/to/config.toml \
  --output_dir path/to/output --output_name my_ft \
  (... other training args ...) \
  --ema --ema_decay 0.9999 --ema_device cuda
```

On Windows Command Prompt, use `^` instead of `\` for line continuation.

<details>
<summary>日本語</summary>

LoRA / ネットワーク学習・フルファインチューニングともに上記のように `--ema` 系の引数を追加します。Windows のコマンドプロンプトでは行末の継続文字に `\` ではなく `^` を使用してください。

</details>

## 4. Multi-GPU and Single-GPU / マルチ GPU とシングル GPU

EMA works on both single-GPU and multi-GPU (DDP) runs.

- The shadow parameters live on the **main process only** (workers store shapes), saving worker memory.
- `update()` and the EMA checkpoint save run on the **main process only** and contain **no collective operation**, so they cannot deadlock.
- For EMA sampling (`--ema_sample`), the shadow weights are broadcast from the main process to the workers, generated, and then the live weights are restored. All ranks enter this path identically.
- `--ema_use_feedback` and `--ema_param_multiplier != 1.0` mutate parameters on the main process only, which would desynchronize DDP, so they are **rejected under multi-GPU**.

<details>
<summary>日本語</summary>

EMA はシングル GPU・マルチ GPU（DDP）のどちらでも動作します。

- シャドウパラメータは **メインプロセスのみ** に保持されます（ワーカーは形状のみ保持）。ワーカーのメモリを節約します。
- `update()` と EMA チェックポイント保存は **メインプロセスのみ** で実行され、**集合通信を含みません**。したがってデッドロックしません。
- EMA サンプリング（`--ema_sample`）では、シャドウの重みをメインプロセスからワーカーへブロードキャストして生成し、その後ライブの重みを復元します。すべてのランクが同一の経路に入ります。
- `--ema_use_feedback` と `--ema_param_multiplier != 1.0` はメインプロセスのみでパラメータを変更し DDP の同期を崩すため、**マルチ GPU では拒否されます**。

</details>

## 5. Notes and Limitations / 注意事項と制限

- For multi-model full fine-tuning (SD3 with MMDiT + text encoders, SDXL with U-Net + text encoders), EMA covers the **main transformer only** (MMDiT / U-Net). If the transformer is not trained (e.g. learning rate `0`), EMA is silently disabled with a warning.
- The EMA checkpoint is written each time a regular checkpoint is saved, so EMA respects the same `--save_every_n_*` cadence and rotation.
- Anima full fine-tuning builds the EMA checkpoint from the shadow state dict; the other models swap the shadow into the model before saving. Both produce an identical-format `ema_` checkpoint.
- Over very long runs a high EMA decay slowly dilutes the pretrained influence; if you want to maximize knowledge retention, do not over-train and keep the decay high.

<details>
<summary>日本語</summary>

- マルチモデルのフルファインチューニング（SD3 の MMDiT + テキストエンコーダ、SDXL の U-Net + テキストエンコーダ）では、EMA は **メインの transformer のみ**（MMDiT / U-Net）を対象とします。transformer を学習しない場合（学習率 `0` など）、EMA は警告とともに無効化されます。
- EMA チェックポイントは通常のチェックポイント保存のたびに書き出されるため、`--save_every_n_*` の間隔やローテーションに従います。
- Anima のフルファインチューニングはシャドウの state dict から EMA チェックポイントを構築し、他のモデルは保存前にシャドウをモデルへスワップします。どちらも同一形式の `ema_` チェックポイントを生成します。
- 非常に長い学習では、高い EMA 減衰が事前学習の影響を徐々に薄めます。知識保持を最大化したい場合は過学習を避け、減衰を高く保ってください。

</details>

## 6. Recommended Settings / 推奨設定

```bash
--ema \
--ema_decay 0.9999 \
--ema_device cpu \   # use cuda if you have spare VRAM
--ema_sample          # optional, to compare EMA vs non-EMA samples
```

<details>
<summary>日本語</summary>

VRAM に余裕があれば `--ema_device cuda` を使ってください。`--ema_sample` は EMA と非 EMA のサンプルを比較したい場合に任意で指定します。

</details>

## Additional Resources / 追加リソース

- [Output Distillation](distillation.md)
- [Anima LoRA Training Guide](anima_train_network.md)
- [Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits](https://arxiv.org/abs/2411.18704)
