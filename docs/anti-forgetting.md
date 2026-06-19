# Anti Catastrophic-Forgetting / 破滅的忘却の抑制

This document collects the techniques for keeping a model's base knowledge while fine-tuning on new data. They are independent and can be combined. Currently documented here:

- **Replay (`--replay_ratio`)** — rehearse a slice of the original/base data alongside the new data.
- **Adaptive λ (`--adaptive_lambda`)** — auto-tune the strength of a soft penalty (output distillation; later Rank-1 EWC) over time.

Related: [Output Distillation](./distillation.md) pulls the student's prediction toward the frozen base prediction. Replay and distillation are complementary (data-space vs output-space) and may be used together.

<details>
<summary>日本語</summary>

このドキュメントは、新しいデータでファインチューニングしつつモデルのベース知識を保持するための手法をまとめます。各手法は独立しており、組み合わせて使えます。現在ここで説明するもの:

- **リプレイ（`--replay_ratio`）** — 元データ／ベースデータの一部を新データと一緒に再学習する。

関連: [出力蒸留](./distillation.md) は生徒の予測を凍結ベースの予測へ近づけます。リプレイと蒸留は補完的（データ空間 対 出力空間）であり、併用できます。

</details>

## Replay (`--replay_ratio`)

### How It Works / 仕組み

Replay keeps mixing a portion of the **original/base data** into training so the model rehearses old knowledge while learning the new task. You mark the base-data subset with `is_replay = true`, and set `--replay_ratio` to the fraction of sampled images that should come from replay subsets.

The ratio is **epoch-level, not per-batch**. A batch is a slice of a single resolution bucket (all images in a batch must share a resolution so latents stack), so replay and new images cannot be guaranteed to share a batch. Instead, the rate is realized through `num_repeats`: the trainer scales the effective repeats of replay images so that, across an epoch, about `replay_ratio` of all sampled images are replay images.

Because `num_repeats` is an integer, the achieved ratio is **approximate**, and enabling replay **increases the effective epoch size** (more total samples per epoch), so set `--max_train_steps` accordingly.

<details>
<summary>日本語</summary>

リプレイは **元データ／ベースデータ** の一部を学習に混ぜ続け、新しいタスクを学習しながら旧知識を再学習させます。ベースデータのサブセットに `is_replay = true` を付け、`--replay_ratio` にリプレイサブセットから取るサンプルの割合を設定します。

この割合は **バッチ単位ではなくエポック単位** です。バッチは単一解像度バケットのスライス（同一バッチ内の画像は latent をスタックするため同一解像度である必要がある）なので、リプレイ画像と新規画像が同じバッチに入る保証はできません。代わりに `num_repeats` を通じて割合を実現します。すなわち、エポック全体でサンプリングされる画像のうち約 `replay_ratio` がリプレイ画像になるよう、リプレイ画像の実効リピート数をスケールします。

`num_repeats` は整数のため、達成される割合は **近似** であり、リプレイを有効にすると **実効エポックサイズが増加** します（エポックあたりの総サンプル数が増える）。`--max_train_steps` を適宜設定してください。

</details>

### Configuration / 設定

- `is_replay` (dataset config, per subset, default `false`): marks a subset as the base/original data slice. Ascendable, so it can also be set at the dataset or `[general]` level.
- `--replay_ratio` (CLI, float in `[0, 1)`, default `0.0` = disabled): target fraction of sampled images from replay subsets.

If `--replay_ratio > 0` but no subset has `is_replay = true`, training stops with an error.

Example dataset config:

```toml
[general]
resolution = 1024
[[datasets]]
batch_size = 2
  [[datasets.subsets]]
  image_dir = "path/to/new_data"   # the new task data
  num_repeats = 1
  [[datasets.subsets]]
  image_dir = "path/to/base_slice" # a slice of the original/base data
  num_repeats = 1
  is_replay = true
```

Example command (full fine-tuning, ~20% replay):

```bash
accelerate launch --mixed_precision bf16 flux_train.py \
  --pretrained_model_name_or_path model.safetensors \
  --dataset_config config.toml \
  --output_dir output --output_name my_model \
  --replay_ratio 0.2
```

The same `--replay_ratio` flag works for LoRA/network training (`*_train_network.py`) and for full fine-tuning of SDXL/SD3/3.5/FLUX.1/Lumina/Anima.

<details>
<summary>日本語</summary>

- `is_replay`（データセット設定、サブセット単位、既定 `false`）: そのサブセットをベース／元データとして扱う。上位継承可能で、データセットや `[general]` レベルでも設定できます。
- `--replay_ratio`（CLI、`[0, 1)` の float、既定 `0.0` = 無効）: リプレイサブセットから取るサンプルの目標割合。

`--replay_ratio > 0` なのに `is_replay = true` のサブセットが無い場合、エラーで停止します。

`--replay_ratio` は LoRA / ネットワーク学習（`*_train_network.py`）でも、SDXL/SD3/3.5/FLUX.1/Lumina/Anima のフルファインチューニングでも同様に使えます。

</details>

### Notes / 補足

- Replay images should cover the training resolutions; images that land in buckets with no new-data counterpart are still trained, but the per-batch composition will simply be replay-only for those buckets.
- Replay is the simplest and most robust anti-forgetting method, and it is cheap when you already have the base-data slice. It composes well with [output distillation](./distillation.md).

<details>
<summary>日本語</summary>

- リプレイ画像は学習解像度をカバーしているのが望ましいです。新規データと対応しないバケットに入る画像も学習されますが、そのバケットのバッチ構成はリプレイのみになります。
- リプレイは最もシンプルで頑健な忘却抑制手法であり、ベースデータの一部を既に持っているなら低コストです。[出力蒸留](./distillation.md) とよく組み合わせられます。

</details>

## Adaptive λ (`--adaptive_lambda`)

### How It Works / 仕組み

A static preservation weight is a poor compromise: too high and the model cannot learn the new task; too low and it forgets. The right value also changes during training. Adaptive λ is a thermostat for the soft-penalty strength. Each step it tracks the ratio of the preservation (penalty) loss to the task loss, smooths it with an EMA, and produces a coefficient:

```
r       = preserve_loss / task_loss
r_bar   = ema * r_bar + (1 - ema) * r
coeff   = clamp(base * r_bar, min, max)
```

The coefficient **multiplies the existing penalty**, so the noise profile from `--distillation_weight_high/low` is preserved; the controller only modulates the overall strength over time. When forgetting grows (preservation loss rises relative to the task), `coeff` rises to protect the base; when the new task is hard (task loss large), `coeff` falls to give the model room to learn. The two loss scalars are averaged across ranks before the ratio is computed, so every process applies the same coefficient (DDP-consistent).

Adaptive λ needs an active soft penalty to scale. It currently drives **output distillation** (and, in a later step, Rank-1 EWC). If `--adaptive_lambda` is set with no soft penalty active, it is disabled with a warning (e.g. it cannot drive OPLoRA, which has no λ — its knob is the projection rank).

<details>
<summary>日本語</summary>

固定の保存重みは妥協的です。高すぎると新タスクを学習できず、低すぎると忘却します。適切な値は学習中にも変化します。Adaptive λ はソフトペナルティ強度の「サーモスタット」です。各ステップで保存（ペナルティ）損失とタスク損失の比を追跡し、EMA で平滑化して係数を算出します。

```
r       = preserve_loss / task_loss
r_bar   = ema * r_bar + (1 - ema) * r
coeff   = clamp(base * r_bar, min, max)
```

係数は **既存のペナルティに乗算** されるため、`--distillation_weight_high/low` によるノイズプロファイルは保持され、全体強度のみを時間方向に調整します。忘却が進む（保存損失がタスクに対して上昇）と `coeff` が上がってベースを保護し、新タスクが難しい（タスク損失が大きい）と `coeff` が下がって学習の余地を与えます。2 つの損失スカラーは比の計算前にランク間で平均されるため、全プロセスが同じ係数を適用します（DDP 一貫）。

Adaptive λ はスケール対象のソフトペナルティが必要です。現在は **出力蒸留**（および後のステップで Rank-1 EWC）を駆動します。ソフトペナルティが無効なまま `--adaptive_lambda` を指定すると、警告とともに無効化されます（λ を持たない OPLoRA は駆動できません。OPLoRA のつまみは射影ランクです）。

</details>

### Configuration / 設定

- `--adaptive_lambda` (flag, default off): enable the controller.
- `--adaptive_lambda_ema` (default `0.99`): EMA decay for smoothing the loss ratio.
- `--adaptive_lambda_base` (default `1.0`): base multiplier; `1.0` keeps the penalty near its configured strength at start.
- `--adaptive_lambda_min` / `--adaptive_lambda_max` (default `0.0` / `10.0`): clamp range for the coefficient.

Example (full fine-tuning, distillation with adaptive strength):

```bash
accelerate launch --mixed_precision bf16 flux_train.py \
  --pretrained_model_name_or_path model.safetensors \
  --dataset_config config.toml \
  --output_dir output --output_name my_model \
  --distillation_weight_high 1.0 --distillation_weight_low 0.0 \
  --adaptive_lambda
```

<details>
<summary>日本語</summary>

- `--adaptive_lambda`（フラグ、既定オフ）: コントローラを有効化。
- `--adaptive_lambda_ema`（既定 `0.99`）: 損失比平滑化の EMA 減衰。
- `--adaptive_lambda_base`（既定 `1.0`）: ベース倍率。`1.0` は開始時にペナルティを設定値付近に保ちます。
- `--adaptive_lambda_min` / `--adaptive_lambda_max`（既定 `0.0` / `10.0`）: 係数のクランプ範囲。

</details>
