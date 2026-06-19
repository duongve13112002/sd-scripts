"""CPU unit tests for OPLoRA (networks/oplora.py).

These exercise the orthogonal-projection math on mock LoRA modules: after projection the
updated factors lie in the orthogonal complement of the base weight's top-k singular subspace,
the base's top-k singular triples are preserved, and the rank/shapes are unchanged. They also
cover the split-qkv skip, the enable switch, and the rank validation. No real model is run.
"""

import argparse
import types

import pytest
import torch
import torch.nn as nn

import networks.oplora as oplora


def _mock_lora(in_dim, out_dim, r, conv=False, split_dims=None):
    m = types.SimpleNamespace()
    if conv:
        m.org_module = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False)
        m.lora_down = nn.Conv2d(in_dim, r, 3, 1, 1, bias=False)  # weight (r, in, 3, 3)
        m.lora_up = nn.Conv2d(r, out_dim, 1, 1, bias=False)      # weight (out, r, 1, 1)
    else:
        m.org_module = nn.Linear(in_dim, out_dim, bias=False)
        m.lora_down = nn.Linear(in_dim, r, bias=False)  # weight (r, in)
        m.lora_up = nn.Linear(r, out_dim, bias=False)   # weight (out, r)
    m.split_dims = split_dims
    # give the adapter a non-trivial (trained-like) state
    with torch.no_grad():
        m.lora_up.weight.normal_()
        m.lora_down.weight.normal_()
    return m


def _network(*loras):
    return types.SimpleNamespace(text_encoder_loras=[], unet_loras=list(loras))


def test_projection_makes_factors_orthogonal_to_topk():
    torch.manual_seed(0)
    lora = _mock_lora(16, 12, r=4)
    mgr = oplora.OPLoRAManager(_network(lora), rank=3, use_lowrank_svd=False)
    mgr.project()
    u_k, v_k = mgr.bases[id(lora)]
    up2d = lora.lora_up.weight.detach().reshape(12, -1).float()
    down2d = lora.lora_down.weight.detach().reshape(4, -1).float()
    # U_k^T up' == 0 and down' V_k == 0
    assert torch.allclose(u_k.t() @ up2d, torch.zeros(3, 4), atol=1e-4)
    assert torch.allclose(down2d @ v_k, torch.zeros(4, 3), atol=1e-4)


def test_projection_preserves_base_topk_singular_triples():
    torch.manual_seed(1)
    lora = _mock_lora(16, 12, r=4)
    w = lora.org_module.weight.detach().float().clone()
    mgr = oplora.OPLoRAManager(_network(lora), rank=3, use_lowrank_svd=False)
    mgr.project()
    u_k, v_k = mgr.bases[id(lora)]
    up2d = lora.lora_up.weight.detach().reshape(12, -1).float()
    down2d = lora.lora_down.weight.detach().reshape(4, -1).float()
    delta = up2d @ down2d
    # (W + dW') acts on the top-k right singular subspace exactly like W
    assert torch.allclose((w + delta) @ v_k, w @ v_k, atol=1e-4)


def test_projection_preserves_shapes():
    lora = _mock_lora(16, 12, r=4)
    up_shape, down_shape = lora.lora_up.weight.shape, lora.lora_down.weight.shape
    mgr = oplora.OPLoRAManager(_network(lora), rank=3, use_lowrank_svd=False)
    mgr.project()
    assert lora.lora_up.weight.shape == up_shape
    assert lora.lora_down.weight.shape == down_shape


def test_projection_conv_module():
    torch.manual_seed(2)
    lora = _mock_lora(8, 6, r=2, conv=True)
    mgr = oplora.OPLoRAManager(_network(lora), rank=2, use_lowrank_svd=False)
    mgr.project()
    u_k, _ = mgr.bases[id(lora)]
    up2d = lora.lora_up.weight.detach().reshape(6, -1).float()  # (out, r)
    assert torch.allclose(u_k.t() @ up2d, torch.zeros(2, 2), atol=1e-4)


def test_lowrank_svd_basis_also_orthogonalizes():
    torch.manual_seed(3)
    lora = _mock_lora(32, 24, r=4)
    mgr = oplora.OPLoRAManager(_network(lora), rank=4, use_lowrank_svd=True)
    mgr.project()
    u_k, v_k = mgr.bases[id(lora)]
    up2d = lora.lora_up.weight.detach().reshape(24, -1).float()
    assert torch.allclose(u_k.t() @ up2d, torch.zeros(4, 4), atol=1e-3)


def test_split_dims_module_is_skipped():
    lora = _mock_lora(16, 12, r=4, split_dims=[6, 6])
    mgr = oplora.OPLoRAManager(_network(lora), rank=3, use_lowrank_svd=False)
    assert id(lora) not in mgr.bases  # split-qkv modules are not projected


def test_create_manager_enable_and_rank_validation():
    lora = _mock_lora(16, 12, r=4)
    net = _network(lora)
    assert oplora.create_oplora_manager(argparse.Namespace(oplora=False), net) is None
    with pytest.raises(ValueError):
        oplora.create_oplora_manager(argparse.Namespace(oplora=True, oplora_rank=0), net)
    mgr = oplora.create_oplora_manager(
        argparse.Namespace(oplora=True, oplora_rank=2, oplora_full_svd=True), net
    )
    assert mgr is not None and id(lora) in mgr.bases


def test_oplora_args_registered_on_lora_parser_only():
    import train_network

    parser = train_network.setup_parser()
    args = parser.parse_args([])
    assert args.oplora is False and args.oplora_rank == 0 and args.oplora_full_svd is False
    on = parser.parse_args(["--oplora", "--oplora_rank", "16", "--oplora_full_svd"])
    assert on.oplora is True and on.oplora_rank == 16 and on.oplora_full_svd is True
