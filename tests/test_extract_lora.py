"""CPU tests for the unified LoRA extractor (networks/extract_lora.py).

These cover the model-agnostic core (SVD of a weight difference, state-dict assembly) and the
org->tuned module name-mapping, using tiny synthetic modules. Real per-architecture checkpoint
loading is validated separately on GPU (see docs/extract-lora.md), since the dev machine is CPU-only.
"""

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "networks"))

import networks.extract_lora as extract_lora


def test_extract_up_down_reconstructs_low_rank_linear():
    torch.manual_seed(0)
    out_dim, in_dim, r = 6, 5, 2
    # a rank-r difference: up @ down
    up_true = torch.randn(out_dim, r)
    down_true = torch.randn(r, in_dim)
    diff = up_true @ down_true

    up, down = extract_lora.extract_up_down(diff, rank=r, clamp_quantile=1.0, device=None, save_dtype=None)
    assert up.shape == (out_dim, r)
    assert down.shape == (r, in_dim)
    # SVD gives a different factorization but the product must match the rank-r difference
    assert torch.allclose(up @ down, diff, atol=1e-4)


def test_extract_up_down_rank_capped_to_layer_dims():
    diff = torch.randn(3, 4)
    up, down = extract_lora.extract_up_down(diff, rank=100, clamp_quantile=1.0, device=None, save_dtype=None)
    # rank cannot exceed min(in, out) = 3
    assert up.shape[1] == 3 and down.shape[0] == 3
    # full-rank extraction reconstructs the full difference
    assert torch.allclose(up @ down, diff, atol=1e-4)


def test_extract_up_down_handles_conv2d_3x3():
    out_ch, in_ch, k = 4, 3, 3
    diff = torch.randn(out_ch, in_ch, k, k)
    up, down = extract_lora.extract_up_down(diff, rank=2, clamp_quantile=1.0, device=None, save_dtype=None)
    assert up.shape == (out_ch, 2, 1, 1)
    assert down.shape == (2, in_ch, k, k)


def test_build_lora_state_dict_keys_alpha_and_reconstruction():
    out_dim, in_dim, r = 5, 4, 2
    up_true = torch.randn(out_dim, r)
    down_true = torch.randn(r, in_dim)
    w_org = torch.randn(out_dim, in_dim)
    w_tuned = w_org + up_true @ down_true  # difference is exactly rank r

    pairs = [("lora_unet_fc", w_org, w_tuned)]
    sd = extract_lora.build_lora_state_dict(pairs, dim=r, conv_dim=None, clamp_quantile=1.0, device=None, save_dtype=None)

    assert set(sd.keys()) == {"lora_unet_fc.lora_up.weight", "lora_unet_fc.lora_down.weight", "lora_unet_fc.alpha"}
    assert int(sd["lora_unet_fc.alpha"].item()) == r  # alpha == rank
    recon = sd["lora_unet_fc.lora_up.weight"] @ sd["lora_unet_fc.lora_down.weight"]
    assert torch.allclose(recon, w_tuned - w_org, atol=1e-4)


def test_build_lora_state_dict_uses_conv_dim_for_3x3():
    diff_conv = torch.randn(4, 3, 3, 3)
    w_org = torch.zeros(4, 3, 3, 3)
    pairs = [("lora_unet_conv", w_org, diff_conv)]
    sd = extract_lora.build_lora_state_dict(pairs, dim=8, conv_dim=2, clamp_quantile=1.0, device=None, save_dtype=None)
    # conv-3x3 uses conv_dim (2), not dim (8)
    assert sd["lora_unet_conv.lora_down.weight"].shape[0] == 2


class _TinyDenoiser(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)


def _make_args(**kw):
    base = dict(
        model_type="faketype", model_org="ORG", model_tuned="TUNED",
        conv_dim=None, load_precision="float", include_text_encoder=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _register_fake(monkeypatch, org_model, tuned_model):
    """Register a fake architecture whose loader returns the given org/tuned models and whose
    create_network names the single Linear as 'lora_unet_fc' pointing at the passed denoiser's module."""
    def loader(args, path, dtype, with_te):
        return (org_model if path == "ORG" else tuned_model), []

    def create_network(multiplier, dim, alpha, vae, text_encoders, denoiser, **kwargs):
        lora = types.SimpleNamespace(lora_name="lora_unet_fc", org_module=denoiser.fc)
        return types.SimpleNamespace(unet_loras=[lora], text_encoder_loras=[])

    fake_module = types.SimpleNamespace(create_network=create_network)
    monkeypatch.setitem(extract_lora.MODEL_REGISTRY, "faketype", extract_lora.ModelEntry("fake.module", loader, False))
    monkeypatch.setattr(extract_lora.importlib, "import_module", lambda name: fake_module)


def test_collect_weight_pairs_maps_org_to_tuned_by_name(monkeypatch):
    torch.manual_seed(1)
    org = _TinyDenoiser(4, 3)
    tuned = _TinyDenoiser(4, 3)
    with torch.no_grad():
        tuned.fc.weight += 1.0  # make them differ
    _register_fake(monkeypatch, org, tuned)

    pairs, network_module = extract_lora.collect_weight_pairs(_make_args(), dim=2, with_te=False)
    assert network_module == "fake.module"
    assert len(pairs) == 1
    name, w_org, w_tuned = pairs[0]
    assert name == "lora_unet_fc"
    # the org weight comes from the org model, the tuned weight from the tuned model (mapped by name)
    assert torch.equal(w_org, org.fc.weight)
    assert torch.equal(w_tuned, tuned.fc.weight)
    assert not torch.equal(w_org, w_tuned)


def test_collect_weight_pairs_rejects_shape_mismatch(monkeypatch):
    org = _TinyDenoiser(4, 3)
    tuned = _TinyDenoiser(5, 3)  # different in_dim -> shape mismatch on fc.weight
    _register_fake(monkeypatch, org, tuned)
    with pytest.raises(RuntimeError, match="same architecture"):
        extract_lora.collect_weight_pairs(_make_args(), dim=2, with_te=False)


def test_registry_has_all_supported_models():
    for mt in ["sd", "sdxl", "sd3", "flux", "lumina", "anima", "hunyuan_image"]:
        assert mt in extract_lora.MODEL_REGISTRY
        assert extract_lora.MODEL_REGISTRY[mt].network_module.startswith("networks.")


# --- orthogonal-to-base projection ---

def test_orthogonalize_diff_is_orthogonal_to_base_topk():
    from networks.oplora import _compute_basis

    torch.manual_seed(0)
    w_org = torch.randn(8, 8)
    diff = torch.randn(8, 8)
    proj = extract_lora.orthogonalize_diff(diff, w_org, rank=2, full_svd=True)

    u_k, v_k = _compute_basis(w_org, 2, use_lowrank_svd=False)
    # the projected difference must not move along the base's top-k left/right singular directions
    assert torch.allclose(u_k.t() @ proj, torch.zeros(2, 8), atol=1e-4)
    assert torch.allclose(proj @ v_k, torch.zeros(8, 2), atol=1e-4)
    # and it must actually change the diff (it dropped the top-k component)
    assert not torch.allclose(proj, diff, atol=1e-3)


# --- LoKr (nearest Kronecker product) extraction ---

def _kron(w1, w2):
    return torch.kron(w1, w2)


def test_lokr_linear_full_reconstructs_kron():
    from networks.lokr import factorization

    out_dim, in_dim = 8, 8
    out_l, out_k = factorization(out_dim, -1)  # (2, 4)
    in_m, in_n = factorization(in_dim, -1)     # (2, 4)
    torch.manual_seed(1)
    w1 = torch.randn(out_l, in_m)
    w2 = torch.randn(out_k, in_n)
    diff = _kron(w1, w2)  # exactly a Kronecker product

    keys = extract_lora.extract_lokr_keys(diff, factor=-1, dim=4)  # dim>=2 -> full w2
    assert ".lokr_w1" in keys and ".lokr_w2" in keys
    recon = _kron(keys[".lokr_w1"], keys[".lokr_w2"])
    assert torch.allclose(recon, diff, atol=1e-4)


def test_lokr_linear_lowrank_reconstructs_kron():
    from networks.lokr import factorization

    out_dim, in_dim = 8, 8
    out_l, out_k = factorization(out_dim, -1)
    in_m, in_n = factorization(in_dim, -1)
    torch.manual_seed(2)
    w1 = torch.randn(out_l, in_m)
    # rank-1 w2 so the low-rank (dim=1) extraction is exact
    w2 = torch.randn(out_k, 1) @ torch.randn(1, in_n)
    diff = _kron(w1, w2)

    keys = extract_lora.extract_lokr_keys(diff, factor=-1, dim=1)  # dim<2 -> low-rank w2
    assert ".lokr_w2_a" in keys and ".lokr_w2_b" in keys
    assert int(keys[".alpha"].item()) == 1  # scale = alpha / lora_dim = 1
    recon = _kron(keys[".lokr_w1"], keys[".lokr_w2_a"] @ keys[".lokr_w2_b"])
    assert torch.allclose(recon, diff, atol=1e-4)


def test_lokr_conv1x1_reconstructs():
    from networks.lokr import factorization

    out_dim, in_dim = 8, 8
    out_l, out_k = factorization(out_dim, -1)
    in_m, in_n = factorization(in_dim, -1)
    torch.manual_seed(3)
    w1 = torch.randn(out_l, in_m)
    w2 = torch.randn(out_k, in_n)
    diff2d = _kron(w1, w2)
    diff = diff2d.unsqueeze(-1).unsqueeze(-1)  # (out, in, 1, 1)

    keys = extract_lora.extract_lokr_keys(diff, factor=-1, dim=4)
    recon = _kron(keys[".lokr_w1"], keys[".lokr_w2"])  # 2D, module re-expands to conv at load
    assert torch.allclose(recon, diff2d, atol=1e-4)


def test_lokr_conv3x3_flat_reconstructs():
    from networks.lokr import factorization

    out_dim, in_ch, k = 8, 8, 3
    out_l, out_k = factorization(out_dim, -1)
    in_m, in_n = factorization(in_ch, -1)
    kprod = k * k
    torch.manual_seed(4)
    w1 = torch.randn(out_l, in_m)
    # rank-2 w2 (out_k, in_n*kprod) so dim=2 low-rank extraction is exact
    w2 = torch.randn(out_k, 2) @ torch.randn(2, in_n * kprod)
    diff_2d = _kron(w1, w2)  # (out_dim, in_ch*kprod)
    diff = diff_2d.reshape(out_dim, in_ch, k, k)

    keys = extract_lora.extract_lokr_keys(diff, factor=-1, dim=2)
    assert ".lokr_w2_a" in keys and ".lokr_w2_b" in keys
    recon_2d = _kron(keys[".lokr_w1"], keys[".lokr_w2_a"] @ keys[".lokr_w2_b"])
    recon = recon_2d.reshape(out_dim, in_ch, k, k)
    assert torch.allclose(recon, diff, atol=1e-4)


# --- build_state_dict dispatch ---

def _build_args(**kw):
    base = dict(
        extract_as="lora", dim=4, conv_dim=None, clamp_quantile=0.99, device=None,
        orthogonal_to_base=False, orthogonal_rank=16, orthogonal_full_svd=False, lokr_factor=-1,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_build_state_dict_lokr_keys():
    w_org = torch.zeros(8, 8)
    w_tuned = torch.randn(8, 8)
    sd = extract_lora.build_state_dict(_build_args(extract_as="lokr", dim=4), [("lora_unet_fc", w_org, w_tuned)], None)
    assert "lora_unet_fc.lokr_w1" in sd
    assert "lora_unet_fc.lokr_w2" in sd or "lora_unet_fc.lokr_w2_a" in sd
    assert "lora_unet_fc.alpha" in sd


def test_build_state_dict_lora_orthogonal_keys_and_property():
    from networks.oplora import _compute_basis

    torch.manual_seed(5)
    w_org = torch.randn(8, 8)
    w_tuned = w_org + torch.randn(8, 8)
    sd = extract_lora.build_state_dict(
        _build_args(extract_as="lora", dim=8, orthogonal_to_base=True, orthogonal_rank=2, orthogonal_full_svd=True),
        [("lora_unet_fc", w_org, w_tuned)], None,
    )
    up = sd["lora_unet_fc.lora_up.weight"]
    down = sd["lora_unet_fc.lora_down.weight"]
    recon = up @ down  # the extracted delta
    u_k, _ = _compute_basis(w_org, 2, use_lowrank_svd=False)
    # the extracted delta is (nearly) orthogonal to the base's top-k left subspace: its top-k
    # component is far smaller than that of the raw difference. (SVD outlier clamping keeps it
    # approximate rather than exactly zero.)
    proj_component = (u_k.t() @ recon).norm()
    raw_component = (u_k.t() @ (w_tuned - w_org)).norm()
    assert proj_component < 0.1 * raw_component
