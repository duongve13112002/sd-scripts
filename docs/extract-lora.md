# Extracting a LoRA from two models (`networks/extract_lora.py`)

`networks/extract_lora.py` extracts an approximating LoRA from the **difference between two models of
the same architecture** (an original model and a tuned one) by truncated SVD, at a **customizable rank**.
One tool covers every supported family through a registry, selected with `--model_type`.

It generalizes the older per-architecture tools (`networks/extract_lora_from_models.py` for SD/SDXL and
`networks/flux_extract_lora.py` for FLUX), which remain available unchanged.

## How it works

For each LoRA target layer (a Linear or Conv weight `W`), the tool computes the difference and factorizes
it with SVD, keeping the top `r` singular directions:

```
ΔW = W_tuned − W_org
U, S, Vᵀ = SVD(ΔW)
lora_up   = U[:, :r] · diag(S[:r])
lora_down = Vᵀ[:r, :]
alpha     = r
```

So `lora_up · lora_down ≈ ΔW`. This is the best rank-`r` approximation of the difference (Eckart–Young).
It is **lossy** unless `r` reaches the layer's full rank — higher rank tracks `ΔW` more closely but produces
a larger file.

Which layers become LoRA, and the exact key names, are taken from each model's own
`networks.lora_{model}.create_network`, so the extracted LoRA uses the **same naming as training** and loads
in the matching trainer and in ComfyUI. (The model's own include/exclude rules — e.g. Anima skipping
`modulation`/`norm`/`embedder`/`final_layer` — are honored automatically.)

## Usage

```bash
python networks/extract_lora.py \
  --model_type anima \
  --model_org  A.safetensors \
  --model_tuned B.safetensors \
  --save_to    B_minus_A_lora.safetensors \
  --dim 32 --device cuda --save_precision bf16
```

`--model_type` is one of: `sd`, `sdxl`, `sd3`, `flux`, `lumina`, `anima`, `hunyuan_image`.

Extract a **LoKr** (smaller file) instead of a LoRA:

```bash
python networks/extract_lora.py --model_type anima \
  --model_org A.safetensors --model_tuned B.safetensors \
  --save_to B_minus_A_lokr.safetensors --extract_as lokr --dim 32 --lokr_factor -1 --device cuda
```

Extract only the part of `B − A` that does **not** overwrite the base's core directions (anti-forgetting):

```bash
python networks/extract_lora.py --model_type anima \
  --model_org A.safetensors --model_tuned B.safetensors \
  --save_to B_minus_A_ortho.safetensors --orthogonal_to_base --orthogonal_rank 16 --dim 32 --device cuda
```

### Output formats and projection

- `--extract_as` (`lora`|`lokr`, default `lora`): the adapter format to produce.
  - `lora` — SVD of the difference (the default; closed-form optimal rank-`r` fit).
  - `lokr` — **LoKr** (Kronecker product) via nearest-Kronecker-product (Van Loan–Pitsianis): a smaller
    file at comparable expressiveness. `--lokr_factor` (default `-1` = balanced, as in training) sets the
    Kronecker split. Covers Linear, conv-1×1, and conv-3×3 **flat** mode — i.e. every DiT family
    (FLUX/SD3/Lumina/Anima/HunyuanImage, all Linear) fully, plus SD/SDXL (conv-3×3 only appears with
    `--conv_dim`). The conv **tucker** mode and full-conv-`w2` are **not** produced (niche, SD/SDXL-only);
    conv-3×3 is always extracted as flat low-rank, which is valid and loadable.
- `--orthogonal_to_base` (flag, **`lora` only**): after the SVD, project the LoRA onto the orthogonal
  complement of the **base model's** top-k singular subspace (OPLoRA-style), keeping only the part of
  `B − A` that does **not** overwrite the base's most important directions. This is **lossy by design** (it
  drops the top-k component), so it does not reconstruct `B − A` — use it when you want "the part B learned
  without overwriting A's core knowledge". `--orthogonal_rank` (default 16) sets k; `--orthogonal_full_svd`
  uses full SVD for the basis (default: randomized). It cannot be combined with `--extract_as lokr` (the
  projection breaks the Kronecker structure), and the run errors if both are set.

### Key options

- `--dim` (int, default 4): rank of the extracted adapter (LoRA rank, or LoKr inner/`w2` rank).
- `--conv_dim` (int, default unset): rank for conv-3×3 layers; when set, conv layers are also extracted.
  Only SD/SDXL have conv layers; the DiT families are Linear-only, so this is usually irrelevant for them.
- `--include_text_encoder` (flag): also extract a LoRA for the text encoder(s). **SD/SDXL only** — for the
  DiT families the text encoders are shared/frozen, so only the denoiser is extracted and this flag is ignored.
- `--device` (e.g. `cuda`): device for the SVD; the difference is always computed in float for precision.
- `--load_precision` (`float`|`fp16`|`bf16`, default `float`): precision the two models are loaded in.
  Use `bf16` for very large models (e.g. FLUX 12B) to halve load memory.
- `--save_precision` (`float`|`fp16`|`bf16`): precision of the saved LoRA.
- `--clamp_quantile` (default 0.99): clamps SVD outliers for numerical stability.
- `--no_metadata`: write only the minimal `ss_` metadata (skip SAI model-spec tags).
- Loader options reused from the trainers, read only when relevant to the chosen `--model_type`:
  `--v2` (sd), `--disable_mmap_load_safetensors`, `--attn_mode` / `--split_attn` (anima, hunyuan_image),
  `--use_flash_attn` (lumina).

### Continue fine-tuning while staying anchored to a base

If you fine-tuned `A → B` and want the difference as a reusable, adjustable module, extract `B − A` as a LoRA.
Applying it on top of a frozen `A` at an adjustable strength is an "add-and-freeze" way to keep the base
intact while layering the learned change — complementary to the anti-forgetting methods in
[anti-forgetting.md](./anti-forgetting.md).

## Validation note

The SVD core, the LoRA state-dict assembly, the org→tuned module name-mapping, the LoKr
nearest-Kronecker-product reconstruction (Linear / conv-1×1 / conv-3×3 flat), and the orthogonal-to-base
projection are unit-tested on CPU (`tests/test_extract_lora.py`). Loading a real checkpoint for each
architecture needs a GPU/large-RAM host, so validate a new `--model_type` on a real pair of checkpoints
there (the run prints how many modules were extracted; load the result in the matching trainer or ComfyUI
to confirm the keys match).

## Adding a new model

When a new architecture is added to the repo, register it in `networks/extract_lora.py` so extraction
supports it. Add one entry to `MODEL_REGISTRY`:

```python
"my_model": ModelEntry(
    "networks.lora_my_model",   # module exposing create_network(mult, dim, alpha, vae, text_encoders, denoiser, **kwargs)
    _load_my_model,             # loader: (args, path, dtype, with_te) -> (denoiser, [text_encoders])
    supports_text_encoder=False,
),
```

and a small loader that mirrors how `{model}_train.py` loads the denoiser. Nothing else changes — the SVD
core and naming are shared.

<details>
<summary>日本語</summary>

`networks/extract_lora.py` は、**同一アーキテクチャの2モデル**（元モデルと派生モデル）の差分を切り捨て SVD で近似し、
**任意のランク**で LoRA として抽出します。`--model_type` で対応する全モデルを1つのツールで扱えます（レジストリ方式）。
既存の `extract_lora_from_models.py`（SD/SDXL）と `flux_extract_lora.py`（FLUX）はそのまま残ります。

各対象層 `W` について `ΔW = W_tuned − W_org` を SVD し、上位 `r` 特異方向で `lora_up`・`lora_down` を作ります
（`alpha = r`）。`r` が大きいほど差分に近づきますがファイルは大きくなります（フルランク未満は損失あり）。
対象層と鍵名は各モデルの `create_network` から取得するため、**学習時と同じ命名**になり、対応トレーナーや ComfyUI で読み込めます。

使い方:

```bash
python networks/extract_lora.py --model_type anima \
  --model_org A.safetensors --model_tuned B.safetensors \
  --save_to B_minus_A_lora.safetensors --dim 32 --device cuda --save_precision bf16
```

主なオプション: `--dim`（ランク）、`--conv_dim`（conv-3×3 のランク。SD/SDXL のみ）、`--include_text_encoder`
（SD/SDXL のみ。DiT 系では無視）、`--device`、`--load_precision`（大規模モデルは `bf16` 推奨）、`--save_precision`、
`--clamp_quantile`、`--no_metadata`、および各トレーナー由来の読み込みオプション
（`--v2`、`--disable_mmap_load_safetensors`、`--attn_mode`/`--split_attn`、`--use_flash_attn`）。

出力形式と射影:
- `--extract_as lokr`: LoRA の代わりに **LoKr**（Kronecker、ファイル小）を最近接 Kronecker 積で抽出。`--lokr_factor`
  で分解を指定（既定 -1）。Linear・conv-1×1・conv-3×3 flat に対応（DiT 系は全て Linear なので完全対応）。conv の
  tucker / full-conv-`w2` は対象外（SD/SDXL の特殊設定のみ。conv-3×3 は flat 低ランクで抽出）。
- `--orthogonal_to_base`（lora のみ）: 抽出した LoRA を **ベースの top-k 特異部分空間の直交補空間**へ射影（OPLoRA 流）。
  ベースの重要方向を上書きしない成分だけを残す（意図的に損失あり）。`--orthogonal_rank`（既定 16）、`--orthogonal_full_svd`。
  `--extract_as lokr` とは併用不可（Kronecker 構造が壊れるためエラー）。

SVD コア・state-dict 構築・org→tuned の名前対応は CPU でユニットテスト済み
（`tests/test_extract_lora.py`）。各アーキテクチャの実チェックポイント読み込みは GPU/大容量メモリ環境で検証してください。

新しいモデルを追加するとき: `networks/extract_lora.py` の `MODEL_REGISTRY` にエントリ（ローダ + `networks.lora_{model}`）
を1つ足すだけで対応できます。

</details>
