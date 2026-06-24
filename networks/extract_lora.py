# Unified LoRA extractor: approximate a LoRA by SVD of the weight difference between two models
# of the same architecture (org -> tuned). Works for every supported model family through a
# registry; the rank is customizable via --dim (and --conv_dim for conv layers).
#
# The code is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo! It generalizes networks/extract_lora_from_models.py (SD/SDXL) and
# networks/flux_extract_lora.py (FLUX) so one tool covers all families.
#
# Adding a new model architecture: add one entry to MODEL_REGISTRY below (a loader that returns the
# denoiser and any text encoders, plus the networks.lora_{model} module used for naming). Nothing
# else needs to change. See docs/extract-lora.md.

import argparse
import json
import os
import time
import importlib
import sys
from typing import Callable, Dict, List, Optional, Tuple

import torch
from safetensors.torch import save_file
from tqdm import tqdm

# Allow running directly as `python networks/extract_lora.py` from the repo root: put the repo root
# (the parent of this file's directory) on sys.path so `library` and `networks` import. Without this,
# running a script that lives in networks/ puts only networks/ on the path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library import sai_model_spec
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def str_to_dtype(p: Optional[str]) -> Optional[torch.dtype]:
    if p == "float" or p == "fp32":
        return torch.float32
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    return None


# Each loader returns (denoiser_module, [text_encoder_modules...]). Text encoders are only loaded
# when --include_text_encoder is set, so denoiser-only extraction (the common case for DiT models,
# whose text encoders are shared/frozen) stays cheap.
def _load_sd(args, path: str, dtype: Optional[torch.dtype], with_te: bool):
    from library import model_util

    text_encoder, _, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, path, device="cpu", dtype=dtype)
    return unet, ([text_encoder] if with_te else [])


def _load_sdxl(args, path: str, dtype: Optional[torch.dtype], with_te: bool):
    from library import sdxl_model_util

    te1, te2, _, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
        sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, path, "cpu", dtype=dtype, disable_mmap=args.disable_mmap_load_safetensors
    )
    return unet, ([te1, te2] if with_te else [])


def _load_flux(args, path: str, dtype: Optional[torch.dtype], with_te: bool):
    from library import flux_utils

    _, model = flux_utils.load_flow_model(path, dtype, "cpu", args.disable_mmap_load_safetensors, model_type="flux")
    return model, []


def _load_sd3(args, path: str, dtype: Optional[torch.dtype], with_te: bool):
    from library import sd3_utils
    from library.safetensors_utils import load_safetensors

    state_dict = load_safetensors(path, "cpu", args.disable_mmap_load_safetensors, dtype)
    mmdit = sd3_utils.load_mmdit(state_dict, dtype, "cpu")
    return mmdit, []


def _load_lumina(args, path: str, dtype: Optional[torch.dtype], with_te: bool):
    from library import lumina_util

    lumina = lumina_util.load_lumina_model(
        path, dtype, torch.device("cpu"), disable_mmap=args.disable_mmap_load_safetensors, use_flash_attn=args.use_flash_attn
    )
    return lumina, []


def _load_anima(args, path: str, dtype: Optional[torch.dtype], with_te: bool):
    from library import anima_utils

    dit = anima_utils.load_anima_model("cpu", path, args.attn_mode, args.split_attn, "cpu", dit_weight_dtype=dtype)
    return dit, []


def _load_hunyuan_image(args, path: str, dtype: Optional[torch.dtype], with_te: bool):
    from library import hunyuan_image_models

    dit = hunyuan_image_models.load_hunyuan_image_model(
        "cpu", path, args.attn_mode, args.split_attn, "cpu", dtype
    )
    return dit, []


class ModelEntry:
    """One supported architecture: how to load it and which LoRA network module names its modules."""

    def __init__(self, network_module: str, loader: Callable, supports_text_encoder: bool):
        self.network_module = network_module  # e.g. "networks.lora_anima"; also written as ss_network_module
        self.loader = loader  # (args, path, dtype, with_te) -> (denoiser, [text_encoders])
        self.supports_text_encoder = supports_text_encoder  # whether --include_text_encoder is meaningful


# To support a new architecture, add an entry here. The network module must expose create_network
# with the shared signature (multiplier, network_dim, network_alpha, vae, text_encoders, denoiser, **kwargs).
MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "sd": ModelEntry("networks.lora", _load_sd, True),
    "sdxl": ModelEntry("networks.lora", _load_sdxl, True),
    "sd3": ModelEntry("networks.lora_sd3", _load_sd3, False),
    "flux": ModelEntry("networks.lora_flux", _load_flux, False),
    "lumina": ModelEntry("networks.lora_lumina", _load_lumina, False),
    "anima": ModelEntry("networks.lora_anima", _load_anima, False),
    "hunyuan_image": ModelEntry("networks.lora_hunyuan_image", _load_hunyuan_image, False),
}


def _qualified_names(collections: List[torch.nn.Module]) -> Dict[int, Tuple[int, str]]:
    """Map id(submodule) -> (collection index, qualified name) across the denoiser and text encoders,
    so a module found in the org model can be located by name in the tuned model."""
    mapping: Dict[int, Tuple[int, str]] = {}
    for ci, root in enumerate(collections):
        for name, sub in root.named_modules():
            mapping[id(sub)] = (ci, name)
    return mapping


def _module_by_name(collections: List[torch.nn.Module]) -> Dict[Tuple[int, str], torch.nn.Module]:
    table: Dict[Tuple[int, str], torch.nn.Module] = {}
    for ci, root in enumerate(collections):
        for name, sub in root.named_modules():
            table[(ci, name)] = sub
    return table


def collect_weight_pairs(args, dim: int, with_te: bool) -> Tuple[List[Tuple[str, torch.Tensor, torch.Tensor]], str]:
    """Load org and tuned models, enumerate the LoRA target modules with the model's own
    create_network (so names match training exactly), and return (lora_name, W_org, W_tuned) triples."""
    entry = MODEL_REGISTRY[args.model_type]
    load_dtype = str_to_dtype(args.load_precision)

    logger.info(f"loading original model: {args.model_org}")
    org_denoiser, org_tes = entry.loader(args, args.model_org, load_dtype, with_te)
    logger.info(f"loading tuned model: {args.model_tuned}")
    tuned_denoiser, tuned_tes = entry.loader(args, args.model_tuned, load_dtype, with_te)

    org_collections = [org_denoiser] + org_tes
    tuned_collections = [tuned_denoiser] + tuned_tes

    net_module = importlib.import_module(entry.network_module)
    # Build a network on the org model only to enumerate target modules and their LoRA names.
    # dim here only controls the throwaway LoRA modules' size (we never use their weights); pass the
    # requested dim so any dim-dependent module selection matches. conv_dim is forwarded so conv
    # layers are enumerated when requested.
    net_kwargs = {}
    if args.conv_dim is not None:
        net_kwargs["conv_dim"] = str(args.conv_dim)
        net_kwargs["conv_alpha"] = str(args.conv_dim)
    network = net_module.create_network(1.0, dim, float(dim), None, org_tes, org_denoiser, **net_kwargs)

    loras = list(network.unet_loras)
    if with_te:
        loras += list(getattr(network, "text_encoder_loras", []))
    if len(loras) == 0:
        raise ValueError(
            f"No LoRA target modules were found for model_type={args.model_type}. Check that the checkpoints "
            f"are {args.model_type} models of the same architecture."
        )

    id2name = _qualified_names(org_collections)
    tuned_table = _module_by_name(tuned_collections)

    pairs: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
    for lora in loras:
        org_module = lora.org_module  # intact because create_network does not call apply_to()
        loc = id2name.get(id(org_module))
        if loc is None:
            raise RuntimeError(f"Could not locate module for {lora.lora_name} in the original model.")
        tuned_module = tuned_table.get(loc)
        if tuned_module is None:
            raise RuntimeError(
                f"Module '{loc[1]}' for {lora.lora_name} is missing from the tuned model; the two models "
                f"do not share the same architecture."
            )
        w_org = org_module.weight
        w_tuned = tuned_module.weight
        if w_org.shape != w_tuned.shape:
            raise RuntimeError(
                f"Shape mismatch for {lora.lora_name}: org {tuple(w_org.shape)} vs tuned {tuple(w_tuned.shape)}; "
                f"the two models do not share the same architecture."
            )
        pairs.append((lora.lora_name, w_org, w_tuned))

    return pairs, entry.network_module


def extract_up_down(
    diff: torch.Tensor, rank: int, clamp_quantile: float, device: Optional[str], save_dtype: Optional[torch.dtype]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SVD of a single weight difference into LoRA up/down factors of the given rank.
    Handles Linear (2D) and Conv2d (4D) weights. Returns (up, down) on CPU."""
    mat = diff.to(torch.float)
    if device:
        mat = mat.to(device)

    conv2d = mat.dim() == 4
    kernel_size = None if not conv2d else tuple(mat.size()[2:4])
    out_dim, in_dim = mat.size()[0], mat.size()[1]
    rank = min(rank, in_dim, out_dim)  # LoRA rank cannot exceed the layer dims

    if conv2d:
        if kernel_size == (1, 1):
            mat = mat.squeeze()
        else:
            mat = mat.flatten(start_dim=1)

    U, S, Vh = torch.linalg.svd(mat)
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)
    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, clamp_quantile)
    low_val = -hi_val
    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)

    if conv2d:
        U = U.reshape(out_dim, rank, 1, 1)
        Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])

    up = U.to("cpu", dtype=save_dtype).contiguous()
    down = Vh.to("cpu", dtype=save_dtype).contiguous()
    return up, down


def build_lora_state_dict(
    weight_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]],
    dim: int,
    conv_dim: Optional[int],
    clamp_quantile: float,
    device: Optional[str],
    save_dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Build the LoRA state dict from (lora_name, W_org, W_tuned) triples by SVD of each difference."""
    lora_sd: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for lora_name, w_org, w_tuned in tqdm(weight_pairs):
            diff = w_tuned.to(torch.float) - w_org.to(torch.float)
            conv2d_3x3 = diff.dim() == 4 and tuple(diff.size()[2:4]) != (1, 1)
            rank = conv_dim if (conv2d_3x3 and conv_dim is not None) else dim
            up, down = extract_up_down(diff, rank, clamp_quantile, device, save_dtype)
            lora_sd[lora_name + ".lora_up.weight"] = up
            lora_sd[lora_name + ".lora_down.weight"] = down
            lora_sd[lora_name + ".alpha"] = torch.tensor(down.size()[0]).to(torch.float)  # alpha = rank
    return lora_sd


def orthogonalize_diff(diff: torch.Tensor, w_org: torch.Tensor, rank: int, full_svd: bool) -> torch.Tensor:
    """Project a weight difference onto the orthogonal complement of the base weight's top-k singular
    subspace (OPLoRA-style): the result keeps only the part of ΔW that does NOT move along the base's
    most important directions. This is lossy by design (it drops the top-k component), so the extracted
    LoRA preserves the base's top-k singular triples. Reuses oplora's basis computation."""
    from networks.oplora import _compute_basis

    basis = _compute_basis(w_org, rank, use_lowrank_svd=not full_svd)
    if basis is None:
        return diff
    u_k, v_k = basis  # (out, k), (in_flat, k), orthonormal columns
    shape = diff.shape
    d = diff.reshape(shape[0], -1).to(torch.float)
    u_k = u_k.to(d.dtype)
    v_k = v_k.to(d.dtype)
    # P_L d P_R with P_L = I - u_k u_k^T, P_R = I - v_k v_k^T, computed without forming full projectors
    ut_d = u_k.t() @ d            # (k, in_flat)
    d_v = d @ v_k                 # (out, k)
    ut_d_v = ut_d @ v_k           # (k, k)
    proj = d - u_k @ ut_d - d_v @ v_k.t() + u_k @ (ut_d_v @ v_k.t())
    return proj.reshape(shape)


def _nearest_kron(mat: torch.Tensor, out_l: int, out_k: int, c1: int, c2: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Best rank-1 nearest-Kronecker-product factorization (Van Loan–Pitsianis): find w1 (out_l, c1)
    and w2 (out_k, c2) minimizing ||mat - kron(w1, w2)||, where mat is (out_l*out_k, c1*c2) laid out
    in torch.kron order. Returns (w1, w2)."""
    rearranged = mat.reshape(out_l, out_k, c1, c2).permute(0, 2, 1, 3).reshape(out_l * c1, out_k * c2)
    U, S, Vh = torch.linalg.svd(rearranged, full_matrices=False)
    s = torch.sqrt(S[0])
    w1 = (U[:, 0] * s).reshape(out_l, c1)
    w2 = (Vh[0, :] * s).reshape(out_k, c2)
    return w1, w2


def _low_rank(mat: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factor mat (a, b) into a@b with inner dim rank (balanced sqrt-singular split)."""
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    rank = min(rank, S.shape[0])
    sqrt_s = torch.sqrt(S[:rank])
    a = U[:, :rank] * sqrt_s  # (a, rank)
    b = sqrt_s.unsqueeze(1) * Vh[:rank, :]  # (rank, b)
    return a, b


def extract_lokr_keys(diff: torch.Tensor, factor: int, dim: int) -> Dict[str, torch.Tensor]:
    """Extract LoKr factors (suffix-keyed, e.g. '.lokr_w1') from a single weight difference by
    nearest-Kronecker-product, with low-rank w2 when the rank is small. Covers Linear, conv-1x1, and
    conv-3x3 'flat' mode (the LoKr default). conv tucker / full-conv-w2 are not produced here."""
    from networks.lokr import factorization

    is_conv = diff.dim() == 4
    out_dim = diff.shape[0]
    if is_conv:
        in_ch, k1, k2 = diff.shape[1], diff.shape[2], diff.shape[3]
        conv1x1 = k1 == 1 and k2 == 1
        kprod = k1 * k2
    else:
        in_ch = diff.shape[1]
        conv1x1 = False
        kprod = 1

    in_m, in_n = factorization(in_ch, factor)
    out_l, out_k = factorization(out_dim, factor)
    diff = diff.to(torch.float)
    keys: Dict[str, torch.Tensor] = {}

    if (not is_conv) or conv1x1:
        # Linear and conv-1x1: 2D Kronecker (conv-1x1 squeezes; the module re-expands at load)
        mat = diff.reshape(out_dim, in_ch)
        w1, w2 = _nearest_kron(mat, out_l, out_k, in_m, in_n)
        keys[".lokr_w1"] = w1
        if dim < max(out_k, in_n) / 2:
            d = min(dim, out_k, in_n)
            w2a, w2b = _low_rank(w2, d)
            keys[".lokr_w2_a"] = w2a
            keys[".lokr_w2_b"] = w2b
            keys[".alpha"] = torch.tensor(float(d))  # scale = alpha / lora_dim = 1
        else:
            keys[".lokr_w2"] = w2  # full w2: the module forces scale = 1
            keys[".alpha"] = torch.tensor(float(min(out_k, in_n)))
    else:
        # conv-3x3 'flat': fold the kernel into w2's columns, factor, then low-rank w2
        mat = diff.reshape(out_dim, in_ch * kprod)
        w1, w2 = _nearest_kron(mat, out_l, out_k, in_m, in_n * kprod)
        d = min(dim, out_k, in_n * kprod)
        w2a, w2b = _low_rank(w2, d)
        keys[".lokr_w1"] = w1
        keys[".lokr_w2_a"] = w2a
        keys[".lokr_w2_b"] = w2b
        keys[".alpha"] = torch.tensor(float(d))
    return keys


def build_state_dict(args, weight_pairs, save_dtype) -> Dict[str, torch.Tensor]:
    """Assemble the adapter state dict from (lora_name, W_org, W_tuned) triples, honoring
    --extract_as (lora|lokr) and --orthogonal_to_base."""
    sd: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for lora_name, w_org, w_tuned in tqdm(weight_pairs):
            diff = w_tuned.to(torch.float) - w_org.to(torch.float)
            if args.orthogonal_to_base:
                diff = orthogonalize_diff(diff, w_org, args.orthogonal_rank, args.orthogonal_full_svd)

            if args.extract_as == "lokr":
                for suffix, tensor in extract_lokr_keys(diff, args.lokr_factor, args.dim).items():
                    out = tensor if suffix == ".alpha" else tensor.to("cpu", dtype=save_dtype)
                    sd[lora_name + suffix] = out.contiguous() if torch.is_tensor(out) else out
            else:
                conv2d_3x3 = diff.dim() == 4 and tuple(diff.size()[2:4]) != (1, 1)
                rank = args.conv_dim if (conv2d_3x3 and args.conv_dim is not None) else args.dim
                up, down = extract_up_down(diff, rank, args.clamp_quantile, args.device, save_dtype)
                sd[lora_name + ".lora_up.weight"] = up
                sd[lora_name + ".lora_down.weight"] = down
                sd[lora_name + ".alpha"] = torch.tensor(down.size()[0]).to(torch.float)
    return sd


def svd(args):
    if args.model_type not in MODEL_REGISTRY:
        raise ValueError(f"unknown --model_type {args.model_type}; choices: {', '.join(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[args.model_type]
    with_te = args.include_text_encoder and entry.supports_text_encoder
    if args.include_text_encoder and not entry.supports_text_encoder:
        logger.warning(
            f"--include_text_encoder is ignored for model_type={args.model_type}: only the denoiser is extracted "
            f"for this family (its text encoders are shared/frozen)."
        )

    if args.extract_as == "lokr" and args.orthogonal_to_base:
        raise ValueError(
            "--orthogonal_to_base cannot be combined with --extract_as lokr: the orthogonal projection "
            "breaks the Kronecker structure (it does not factor into w1 x w2). Use --extract_as lora with "
            "--orthogonal_to_base, or drop --orthogonal_to_base for LoKr."
        )

    save_dtype = str_to_dtype(args.save_precision)

    weight_pairs, network_module = collect_weight_pairs(args, args.dim, with_te)
    logger.info(
        f"extracting {args.extract_as} for {len(weight_pairs)} modules "
        f"(rank={args.dim}, conv_dim={args.conv_dim}, orthogonal_to_base={args.orthogonal_to_base})"
    )
    state_dict = build_state_dict(args, weight_pairs, save_dtype)

    # minimum metadata so the adapter loads in the matching trainer / inference
    if args.extract_as == "lokr":
        ss_network_module = "networks.lokr"
        net_kwargs = {"factor": str(args.lokr_factor)}
    else:
        ss_network_module = network_module
        net_kwargs = {}
        if args.conv_dim is not None:
            net_kwargs["conv_dim"] = str(args.conv_dim)
            net_kwargs["conv_alpha"] = str(args.conv_dim)
    metadata = {
        "ss_network_module": ss_network_module,
        "ss_network_dim": str(args.dim),
        "ss_network_alpha": str(float(args.dim)),
        "ss_network_args": json.dumps(net_kwargs),
    }
    if not args.no_metadata:
        title = os.path.splitext(os.path.basename(args.save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(
            state_dict, False, False, False, True, False, time.time(), title, None
        )
        metadata.update(sai_metadata)

    if save_dtype is not None:
        for key in list(state_dict.keys()):
            state_dict[key] = state_dict[key].to(save_dtype)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_to)), exist_ok=True)
    save_file(state_dict, args.save_to, metadata=metadata)
    logger.info(f"{args.extract_as} weights saved to {args.save_to} ({len(weight_pairs)} modules)")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type", type=str, required=True, choices=list(MODEL_REGISTRY.keys()),
        help="architecture of the two models / 2モデルのアーキテクチャ",
    )
    parser.add_argument("--model_org", type=str, required=True, help="original (base) model / 元モデル")
    parser.add_argument(
        "--model_tuned", type=str, required=True,
        help="tuned model; the LoRA is the difference org -> tuned / 派生モデル（LoRAは元→派生の差分）",
    )
    parser.add_argument("--save_to", type=str, required=True, help="destination .safetensors file / 保存先")
    parser.add_argument(
        "--extract_as", type=str, default="lora", choices=["lora", "lokr"],
        help="adapter format to extract: 'lora' (SVD of the difference) or 'lokr' (Kronecker, smaller file) / "
        "抽出する形式: lora か lokr",
    )
    parser.add_argument("--dim", type=int, default=4, help="rank of the extracted adapter (default 4) / ランク")
    parser.add_argument(
        "--lokr_factor", type=int, default=-1,
        help="LoKr Kronecker factorization factor (-1 = balanced, as in training) / LoKrの分解factor（-1で自動）",
    )
    parser.add_argument(
        "--orthogonal_to_base", action="store_true",
        help="(lora only) project the extracted LoRA onto the orthogonal complement of the base model's top-k "
        "singular subspace (OPLoRA-style), keeping only the part that does not overwrite the base. Lossy by "
        "design / (loraのみ) ベースのtop-k特異部分空間の直交補空間へ射影（OPLoRA流）。base上書き部分を捨てる",
    )
    parser.add_argument(
        "--orthogonal_rank", type=int, default=16,
        help="top-k of the base weight protected by --orthogonal_to_base (default 16) / 保護するtop-k",
    )
    parser.add_argument(
        "--orthogonal_full_svd", action="store_true",
        help="use full SVD (not randomized) when computing the base top-k basis for --orthogonal_to_base / 直交基底に完全SVDを使う",
    )
    parser.add_argument(
        "--conv_dim", type=int, default=None,
        help="rank for conv (3x3) layers; when set, conv layers are also extracted (SD/SDXL only have conv). "
        "If omitted, conv-3x3 layers are not extracted / conv層のランク（指定時はconv層も抽出）",
    )
    parser.add_argument(
        "--include_text_encoder", action="store_true",
        help="also extract a LoRA for the text encoder(s) (SD/SDXL only; ignored for DiT families) / "
        "テキストエンコーダのLoRAも抽出（SD/SDXLのみ）",
    )
    parser.add_argument("--device", type=str, default=None, help="device for the SVD, e.g. cuda / SVDの計算デバイス")
    parser.add_argument(
        "--load_precision", type=str, default="float", choices=["float", "fp16", "bf16"],
        help="precision to load the models in; the difference is always computed in float (default: float) / 読み込み精度",
    )
    parser.add_argument(
        "--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"],
        help="precision to save the LoRA in / 保存精度",
    )
    parser.add_argument(
        "--clamp_quantile", type=float, default=0.99,
        help="quantile (0-1) for clamping SVD outliers (default 0.99) / SVD外れ値クランプの分位点",
    )
    parser.add_argument(
        "--no_metadata", action="store_true",
        help="do not write SAI model-spec metadata (minimum ss_ metadata is still written) / SAIメタデータを書かない",
    )
    # model-load options reused from the trainers (only the relevant ones are read per model_type)
    parser.add_argument("--v2", action="store_true", help="SD v2.x base model (model_type=sd) / SD v2.x")
    parser.add_argument(
        "--disable_mmap_load_safetensors", action="store_true", help="disable mmap when loading safetensors"
    )
    parser.add_argument("--attn_mode", type=str, default="torch", help="attention mode for Anima/HunyuanImage loaders")
    parser.add_argument("--split_attn", action="store_true", help="split attention for Anima/HunyuanImage loaders")
    parser.add_argument("--use_flash_attn", action="store_true", help="use flash attention for the Lumina loader")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    svd(args)
