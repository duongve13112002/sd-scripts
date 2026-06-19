"""CPU unit tests for the shared anti-forgetting utilities (library/anti_forgetting.py).

Covers the adaptive-lambda controller math (ratio tracking, EMA smoothing, clamps,
monotonicity), the penalty-combining helper, the enable switch, the conflict validator
(adaptive lambda needs an active soft penalty), and CLI arg registration. No model is run.
"""

import argparse

import pytest
import torch

from library import anti_forgetting


def _ctrl_args(ema=0.0, base=1.0, lo=0.0, hi=10.0, enabled=True):
    return argparse.Namespace(
        adaptive_lambda=enabled,
        adaptive_lambda_ema=ema,
        adaptive_lambda_base=base,
        adaptive_lambda_min=lo,
        adaptive_lambda_max=hi,
    )


def _penalty_args(high=0.0, low=0.0, adaptive=False, ewc_lambda=0.0):
    return argparse.Namespace(
        distillation_weight_high=high,
        distillation_weight_low=low,
        adaptive_lambda=adaptive,
        adaptive_lambda_ema=0.99,
        adaptive_lambda_base=1.0,
        adaptive_lambda_min=0.0,
        adaptive_lambda_max=10.0,
        ewc_lambda=ewc_lambda,
    )


def test_controller_tracks_ratio_without_smoothing():
    c = anti_forgetting.AdaptiveLambdaController(_ctrl_args(ema=0.0))
    coeff = c.update(torch.tensor(2.0), torch.tensor(1.0))
    assert coeff == pytest.approx(2.0, abs=1e-5)
    coeff = c.update(torch.tensor(0.5), torch.tensor(1.0))
    assert coeff == pytest.approx(0.5, abs=1e-5)


def test_controller_ema_smooths():
    c = anti_forgetting.AdaptiveLambdaController(_ctrl_args(ema=0.5))
    assert c.update(torch.tensor(2.0), torch.tensor(1.0)) == pytest.approx(2.0, abs=1e-5)  # first sets r_bar
    # r=4, r_bar = 0.5*2 + 0.5*4 = 3
    assert c.update(torch.tensor(4.0), torch.tensor(1.0)) == pytest.approx(3.0, abs=1e-5)


def test_controller_clamps():
    c_hi = anti_forgetting.AdaptiveLambdaController(_ctrl_args(ema=0.0, hi=10.0))
    assert c_hi.update(torch.tensor(100.0), torch.tensor(1.0)) == pytest.approx(10.0, abs=1e-5)
    c_lo = anti_forgetting.AdaptiveLambdaController(_ctrl_args(ema=0.0, lo=0.5))
    assert c_lo.update(torch.tensor(0.0), torch.tensor(1.0)) == pytest.approx(0.5, abs=1e-5)


def test_controller_monotone_in_forgetting():
    low = anti_forgetting.AdaptiveLambdaController(_ctrl_args(ema=0.0)).update(torch.tensor(1.0), torch.tensor(1.0))
    high = anti_forgetting.AdaptiveLambdaController(_ctrl_args(ema=0.0)).update(torch.tensor(3.0), torch.tensor(1.0))
    assert high > low


def test_add_adaptive_penalty_without_controller_is_plain_sum():
    loss = torch.tensor(1.0)
    penalty = torch.tensor(2.0)
    out = anti_forgetting.add_adaptive_penalty(loss, penalty, None)
    assert out.item() == pytest.approx(3.0, abs=1e-6)


def test_add_adaptive_penalty_with_controller_scales():
    c = anti_forgetting.AdaptiveLambdaController(_ctrl_args(ema=0.0, base=1.0))
    loss = torch.tensor(1.0)
    penalty = torch.tensor(2.0)
    # coeff = clamp(base * (penalty/task)) = 2.0 ; result = 1 + 2*2 = 5
    out = anti_forgetting.add_adaptive_penalty(loss, penalty, c)
    assert out.item() == pytest.approx(5.0, abs=1e-5)


def test_create_controller_respects_enable():
    assert anti_forgetting.create_adaptive_lambda_controller(_ctrl_args(enabled=False)) is None
    assert anti_forgetting.create_adaptive_lambda_controller(_ctrl_args(enabled=True)) is not None


def test_is_adaptive_lambda_enabled_defaults_off():
    assert anti_forgetting.is_adaptive_lambda_enabled(argparse.Namespace()) is False
    assert anti_forgetting.is_adaptive_lambda_enabled(_ctrl_args(enabled=True)) is True


def test_validator_disables_adaptive_lambda_without_penalty():
    # adaptive lambda on, no distillation, no ewc -> disabled
    args = _penalty_args(high=0.0, low=0.0, adaptive=True)
    anti_forgetting.verify_anti_forgetting_args(args)
    assert args.adaptive_lambda is False


def test_validator_keeps_adaptive_lambda_with_distillation():
    args = _penalty_args(high=1.0, low=0.0, adaptive=True)
    anti_forgetting.verify_anti_forgetting_args(args)
    assert args.adaptive_lambda is True


def test_validator_keeps_adaptive_lambda_with_ewc():
    # ewc_lambda counts as a soft penalty (forward-compatible with the EWC step)
    args = _penalty_args(high=0.0, low=0.0, adaptive=True, ewc_lambda=0.5)
    anti_forgetting.verify_anti_forgetting_args(args)
    assert args.adaptive_lambda is True


def test_adaptive_lambda_args_registered():
    import library.args as args_util

    parser = argparse.ArgumentParser()
    args_util.add_training_arguments(parser, False)
    defaults = parser.parse_args([])
    assert defaults.adaptive_lambda is False
    assert defaults.adaptive_lambda_ema == 0.99
    assert defaults.adaptive_lambda_base == 1.0
    assert defaults.adaptive_lambda_min == 0.0
    assert defaults.adaptive_lambda_max == 10.0
    on = parser.parse_args(["--adaptive_lambda", "--adaptive_lambda_max", "5.0"])
    assert on.adaptive_lambda is True and on.adaptive_lambda_max == 5.0
