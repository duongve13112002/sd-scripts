"""CPU unit tests for the EMA implementation and the Anima EMA checkpoint helpers.

These are intentionally lightweight so they run on a CPU-only machine. They exercise the
EMA math and the filename helpers, not any real training.
"""

import os

import pytest
import torch
import torch.nn as nn

from library.ema import ExponentialMovingAverage
from library import anima_train_utils


def _make_model():
    torch.manual_seed(0)
    return nn.Linear(8, 8)


def test_update_moves_shadow_towards_params():
    model = _make_model()
    params = [p for p in model.parameters() if p.requires_grad]
    ema = ExponentialMovingAverage(parameters=params, decay=0.5, device=torch.device("cpu"))

    shadow_before = ema.shadow_params[0].clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update()
    shadow_after = ema.shadow_params[0]

    # With decay 0.5 the shadow should move halfway towards the new weights, so it changes
    # but stays strictly between the old shadow and the updated weights.
    assert not torch.allclose(shadow_before, shadow_after)
    assert torch.all(shadow_after <= torch.maximum(shadow_before, model.weight.detach()) + 1e-6)


def test_average_parameters_restores_live_weights():
    model = _make_model()
    params = [p for p in model.parameters() if p.requires_grad]
    ema = ExponentialMovingAverage(parameters=params, decay=0.9, device=torch.device("cpu"))

    with torch.no_grad():
        model.weight.add_(2.0)
    ema.update()

    live_weight = model.weight.detach().clone()
    shadow = ema.shadow_params[0].detach().clone()

    with ema.average_parameters():
        assert torch.allclose(model.weight.detach(), shadow)

    # Live weights must be restored exactly after the context exits.
    assert torch.allclose(model.weight.detach(), live_weight)


def test_decay_validation():
    model = _make_model()
    params = list(model.parameters())
    with pytest.raises(ValueError):
        ExponentialMovingAverage(parameters=params, decay=1.5, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        ExponentialMovingAverage(parameters=params, decay=0.9, param_multiplier=0.0, device=torch.device("cpu"))


def test_state_dict_roundtrips_decay():
    model = _make_model()
    ema = ExponentialMovingAverage(
        parameters=list(model.parameters()), decay=0.99, use_num_updates=True, device=torch.device("cpu")
    )
    ema.update()
    sd = ema.state_dict()
    assert sd["decay"] == pytest.approx(0.99)
    assert sd["num_updates"] == 1


def test_ema_filename_helper():
    path = os.path.join("output", "model-step00010.safetensors")
    ema_path = anima_train_utils._get_ema_filename(path)
    assert os.path.basename(ema_path) == "ema_model-step00010.safetensors"
    assert os.path.dirname(ema_path) == os.path.dirname(path)


def test_remove_old_ema_file(tmp_path):
    ckpt = tmp_path / "model-step00010.safetensors"
    ema_file = tmp_path / "ema_model-step00010.safetensors"
    ema_file.write_bytes(b"dummy")

    # None is a no-op and must not raise
    anima_train_utils._remove_old_ema_file(None)

    anima_train_utils._remove_old_ema_file(str(ckpt))
    assert not ema_file.exists()
