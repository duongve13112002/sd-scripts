# Anti Catastrophic-Forgetting / 破滅的忘却の抑制

This document collects the techniques for keeping a model's base knowledge while fine-tuning on new data. They are independent and can be combined. Currently documented here:

- **Replay (`--replay_ratio`)** — rehearse a slice of the original/base data alongside the new data.
- **Adaptive λ (`--adaptive_lambda`)** — auto-tune the strength of a soft penalty (output distillation or Rank-1 EWC) over time.
- **Rank-1 EWC (`--ewc_lambda`)** — penalize weight drift along the dominant Fisher direction (full fine-tuning only).
- **OPLoRA (`--oplora`)** — confine LoRA updates to the orthogonal complement of the base's top-k singular subspace (LoRA training only).

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

## Rank-1 EWC (`--ewc_lambda`)

### How It Works / 仕組み

Elastic Weight Consolidation penalizes drift of important weights away from their pre-trained values. In diffusion, per-sample gradients are strongly collinear at low SNR, so the empirical Fisher information is approximately rank-1, `F ~ u u^T`, where `u` is the mean gradient. The penalty then collapses to a single global scalar:

```
L_ewc = lambda * (u^T (theta - theta*))^2
```

where `theta*` is a snapshot of the initial weights. Unlike output distillation, EWC constrains the **parameter space** directly and needs **no teacher model and no extra forward pass** at train time — only one inner product.

`u` is estimated up front during a short **Fisher phase**: the first `--ewc_fisher_samples` micro-batches run the normal training loss and backward, their gradients are averaged (and reduced across ranks), and no optimizer step is taken (so `theta*` stays at the initial weights). After that, training proceeds normally with the EWC penalty added to the loss each step.

EWC is **full fine-tuning only** — it lives in base-weight space, whereas LoRA optimizes a separate low-rank delta, so passing `--ewc_lambda` to LoRA/network training raises an error. When both EWC and output distillation are enabled, **EWC supersedes distillation** (parameter-space penalty with no resident teacher), and distillation is disabled with a warning. Adaptive λ (`--adaptive_lambda`) can drive the EWC strength over time, just as it does for distillation.

<details>
<summary>日本語</summary>

EWC（Elastic Weight Consolidation）は、重要な重みが事前学習時の値から離れることを抑制します。拡散モデルでは低 SNR でサンプルごとの勾配が強く同一方向に揃うため、経験的フィッシャー情報はほぼランク1（`F ~ u u^T`、`u` は平均勾配）になります。よってペナルティは単一のグローバルスカラーに帰着します。

```
L_ewc = lambda * (u^T (theta - theta*))^2
```

ここで `theta*` は初期重みのスナップショットです。出力蒸留と異なり、EWC は **パラメータ空間** を直接制約し、学習時に **teacher モデルも追加の forward も不要** で、内積1回だけです。

`u` は学習開始前の短い **Fisher フェーズ** で推定します。最初の `--ewc_fisher_samples` 個のマイクロバッチで通常の学習損失と backward を行い、その勾配を平均（ランク間でも平均）し、オプティマイザのステップは踏みません（`theta*` は初期重みのまま）。以降は通常学習に EWC ペナルティを毎ステップ加えます。

EWC は **フルファインチューニング専用** です（ベース重み空間で動作するため）。LoRA/ネットワーク学習に `--ewc_lambda` を渡すとエラーになります。EWC と出力蒸留を同時に有効にした場合は **EWC が蒸留より優先**（teacher 常駐不要のパラメータ空間ペナルティ）され、蒸留は警告とともに無効化されます。Adaptive λ（`--adaptive_lambda`）は蒸留と同様に EWC の強度も時間方向に調整できます。

</details>

### Configuration / 設定

- `--ewc_lambda` (float, default `0.0` = off): EWC penalty weight (full fine-tuning only).
- `--ewc_fisher_samples` (int, default `100`): number of micro-batches used to estimate the Fisher direction before training. Must be smaller than your total steps so the Fisher phase finishes.
- `--ewc_buffers_on_cpu` (flag, default off): store the reference weights (`theta*`) and Fisher vector (`u`) on CPU to save VRAM. This adds a per-step host-device transfer, so it is slower; recommended only for very large models.

Memory and compatibility notes:

- EWC keeps two extra fp32 buffers the size of the trainable weights (`theta*` and `u`). On GPU (default) this costs roughly 2× the parameter memory but every step is cheap; on CPU it frees VRAM but transfers those buffers each step. This is comfortable for SDXL/SD3/Lumina and heavy for FLUX.1 (12B).
- EWC is **incompatible with `--fused_backward_pass` / `--blockwise_fused_optimizers`** (the optimizer steps inside the backward hook, which would update weights during the Fisher phase); enabling both raises an error.
- Works on single and multi-GPU: `u` is averaged across ranks, so every process applies the same penalty (DDP-consistent).

Example (FLUX.1 full fine-tuning with EWC):

```bash
accelerate launch --mixed_precision bf16 flux_train.py \
  --pretrained_model_name_or_path model.safetensors \
  --dataset_config config.toml \
  --output_dir output --output_name my_model \
  --ewc_lambda 0.1 --ewc_fisher_samples 100 --ewc_buffers_on_cpu
```

<details>
<summary>日本語</summary>

- `--ewc_lambda`（float、既定 `0.0` = 無効）: EWC ペナルティ重み（フルファインチューニング専用）。
- `--ewc_fisher_samples`（int、既定 `100`）: 学習前に Fisher 方向を推定するマイクロバッチ数。Fisher フェーズが終わるよう、総ステップ数より小さくしてください。
- `--ewc_buffers_on_cpu`（フラグ、既定オフ）: 参照重み（`theta*`）と Fisher ベクトル（`u`）を CPU に置いて VRAM を節約します。毎ステップの転送が増えるため遅くなります。非常に大きいモデルにのみ推奨。

メモリ・互換性:

- EWC は学習対象重みと同サイズの fp32 バッファを 2 つ（`theta*` と `u`）保持します。GPU（既定）ではおよそパラメータ 2 倍のメモリですが毎ステップは軽量、CPU では VRAM を解放する代わりに毎ステップ転送が発生します。SDXL/SD3/Lumina では余裕があり、FLUX.1（12B）では重くなります。
- EWC は **`--fused_backward_pass` / `--blockwise_fused_optimizers` と非互換** です（オプティマイザが backward フック内でステップし、Fisher フェーズ中に重みを更新してしまうため）。同時指定はエラーになります。
- シングル/マルチ GPU 対応: `u` はランク間で平均されるため、全プロセスが同じペナルティを適用します（DDP 一貫）。

</details>

## OPLoRA (`--oplora`)

### How It Works / 仕組み

OPLoRA (orthogonal-projection LoRA) keeps a LoRA adapter from ever writing over the base model's most important directions. A base weight `W = U S V^T` has its strongest behavior in the top-k singular directions (`U_k`, `V_k`). If the LoRA update `ΔW = up·down` has a component along those directions it overwrites base knowledge. OPLoRA projects every update into the **orthogonal complement** of that top-k subspace:

```
ΔW' = P_L ΔW P_R,   P_L = I - U_k U_k^T,   P_R = I - V_k V_k^T
```

which factors into a cheap low-rank correction of the LoRA factors, applied after each optimizer step:

```
up'  = up   - U_k (U_k^T up)
down' = down - (down V_k) V_k^T
```

After projection `U_k^T up' = 0` and `down' V_k = 0`, so `W + ΔW'` keeps `W`'s top-k singular triples **exactly** unchanged — a hard mathematical guarantee, unlike the soft pull of distillation. There is **no teacher and no extra forward pass**: the bases `U_k`, `V_k` are computed once by SVD of each base weight at startup (before the adapter is attached), and the per-step projection is a few small matmuls.

OPLoRA is **LoRA/network training only** (it operates on the LoRA factors); its arguments are not registered by the full fine-tune scripts, so passing them there is rejected. It has no `λ` to tune — its knob is the subspace size `--oplora_rank` — so adaptive λ does not apply to it. When both OPLoRA and output distillation are enabled, **OPLoRA supersedes distillation** (hard guarantee, no teacher forward) and distillation is disabled with a warning.

<details>
<summary>日本語</summary>

OPLoRA（直交射影 LoRA）は、LoRA アダプターがベースモデルの最も重要な方向を上書きしないようにします。ベース重み `W = U S V^T` は上位 k 個の特異方向（`U_k`, `V_k`）に最も強い振る舞いを持ちます。LoRA 更新 `ΔW = up·down` がその方向に成分を持つとベース知識を上書きします。OPLoRA は更新をその上位 k 部分空間の **直交補空間** に射影します。

```
ΔW' = P_L ΔW P_R,   P_L = I - U_k U_k^T,   P_R = I - V_k V_k^T
```

これは LoRA 因子の安価な低ランク補正に分解でき、各オプティマイザステップ後に適用します。

```
up'  = up   - U_k (U_k^T up)
down' = down - (down V_k) V_k^T
```

射影後は `U_k^T up' = 0`、`down' V_k = 0` となり、`W + ΔW'` は `W` の上位 k 特異三つ組を **厳密に** 保持します（蒸留のソフトな引き戻しと異なり、ハードな数学的保証）。**teacher も追加 forward も不要** で、基底 `U_k`, `V_k` は起動時（アダプター接続前）に各ベース重みの SVD で一度だけ計算し、毎ステップの射影は小さな行列積数回です。

OPLoRA は **LoRA/ネットワーク学習専用** です（LoRA 因子に作用するため）。引数はフルファインチューニングのスクリプトには登録されないため、そこへ渡すと拒否されます。調整する `λ` はなく、つまみは部分空間サイズ `--oplora_rank` なので、adaptive λ は適用されません。OPLoRA と出力蒸留を同時に有効にした場合は **OPLoRA が蒸留より優先**（ハード保証、teacher forward 不要）され、蒸留は警告とともに無効化されます。

</details>

### Configuration / 設定

- `--oplora` (flag, default off): enable orthogonal projection (LoRA training only).
- `--oplora_rank` (int, required when `--oplora` is set): the top-k singular subspace of each base weight to protect. Larger preserves more base knowledge but leaves less room to learn the new task.
- `--oplora_full_svd` (flag, default off): use full SVD instead of fast randomized low-rank SVD when building the bases (slower and more exact).

Notes:

- The one-time SVD of every target weight at startup adds some startup time on large models; randomized SVD (the default) keeps it fast.
- Split-qkv LoRA modules (FLUX, when `split_dims` is used) cannot be projected exactly and are left unprojected (logged at startup).
- Works on single and multi-GPU: the projection is applied identically on every rank after the synchronized optimizer step.

Example (FLUX.1 LoRA with OPLoRA):

```bash
accelerate launch --mixed_precision bf16 flux_train_network.py \
  --pretrained_model_name_or_path model.safetensors \
  --dataset_config config.toml \
  --output_dir output --output_name my_lora \
  --network_module networks.lora_flux --network_dim 16 \
  --oplora --oplora_rank 16
```

<details>
<summary>日本語</summary>

- `--oplora`（フラグ、既定オフ）: 直交射影を有効化（LoRA 学習専用）。
- `--oplora_rank`（int、`--oplora` 指定時は必須）: 各ベース重みで保護する上位 k 特異部分空間。大きいほどベース知識を多く保持しますが、新タスク学習の余地は減ります。
- `--oplora_full_svd`（フラグ、既定オフ）: 基底計算に高速なランダム化低ランク SVD ではなくフル SVD を使う（遅く、より厳密）。

補足:

- 起動時に全対象重みの SVD を一度行うため、大きいモデルでは起動が少し遅くなります。既定のランダム化 SVD で高速に保てます。
- split-qkv LoRA モジュール（FLUX で `split_dims` 使用時）は厳密に射影できないため、射影せずに残します（起動時にログ出力）。
- シングル/マルチ GPU 対応: 同期されたオプティマイザステップ後に、全ランクで同一の射影を適用します。

</details>
