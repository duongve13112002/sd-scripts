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
