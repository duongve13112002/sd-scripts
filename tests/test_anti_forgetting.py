"""CPU unit tests for the shared anti-forgetting utilities (library/anti_forgetting.py).

Covers the adaptive-lambda controller math (ratio tracking, EMA smoothing, clamps,
monotonicity), the penalty-combining helper, the enable switch, the conflict validator
(adaptive lambda needs an active soft penalty), and CLI arg registration. No model is run.
"""

import argparse

import pytest
import torch
import torch.nn as nn

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
        ewc_fisher_samples=100,
        fused_backward_pass=False,
        blockwise_fused_optimizers=False,
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


def test_ewc_penalty_math_and_gradient():
    m = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.0)  # theta* = 0
    reg = anti_forgetting.EWCRegularizer(m.named_parameters(), lam=2.0, num_fisher_samples=0, store_on_cpu=True)
    reg.u["weight"] = torch.tensor([[3.0]])
    reg.ready = True
    with torch.no_grad():
        m.weight.fill_(5.0)  # delta = 5, s = u*delta = 15
    pen = reg.penalty()
    assert pen.item() == pytest.approx(2.0 * 15.0**2, abs=1e-4)  # lam * s^2 = 450
    pen.backward()
    # d/dtheta [lam s^2] = 2 lam s u = 2*2*15*3 = 180
    assert m.weight.grad.item() == pytest.approx(180.0, abs=1e-3)


def test_ewc_penalty_zero_at_init():
    m = nn.Linear(2, 2)
    reg = anti_forgetting.EWCRegularizer(m.named_parameters(), lam=1.0, num_fisher_samples=0, store_on_cpu=True)
    reg.u = {n: torch.ones_like(v) for n, v in reg.u.items()}
    reg.ready = True
    # theta == theta* (no update yet) -> penalty 0 regardless of u
    assert reg.penalty().item() == pytest.approx(0.0, abs=1e-6)


def test_ewc_accumulate_and_finalize_averages():
    m = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.0)
    reg = anti_forgetting.EWCRegularizer(m.named_parameters(), lam=1.0, num_fisher_samples=2, store_on_cpu=True)
    assert reg.collecting and not reg.ready
    m.weight.grad = torch.tensor([[2.0]])
    reg.accumulate()
    assert reg.count == 1 and not reg.maybe_finalize()  # not enough samples yet
    m.weight.grad = torch.tensor([[4.0]])
    reg.accumulate()
    assert reg.maybe_finalize()  # count == 2 -> finalize
    assert reg.ready and not reg.collecting
    assert reg.u["weight"].item() == pytest.approx(3.0, abs=1e-6)  # (2 + 4) / 2


def test_ewc_end_to_end_fisher_then_penalty():
    torch.manual_seed(0)
    m = nn.Linear(4, 4, bias=False)
    reg = anti_forgetting.EWCRegularizer(m.named_parameters(), lam=10.0, num_fisher_samples=3, store_on_cpu=True)
    # Fisher phase: accumulate gradients over a few batches, then finalize
    for _ in range(3):
        m(torch.randn(8, 4)).sum().backward()
        reg.accumulate()
        m.weight.grad = None
    assert reg.maybe_finalize() and reg.ready
    # no drift yet -> penalty is exactly zero
    assert reg.penalty().item() == pytest.approx(0.0, abs=1e-6)
    # drift the weights; penalty becomes positive and produces a gradient that opposes the drift
    with torch.no_grad():
        m.weight += 1.0
    pen = reg.penalty()
    assert pen.item() > 0.0
    pen.backward()
    assert m.weight.grad is not None and torch.any(m.weight.grad != 0)


def test_create_ewc_regularizer_respects_enable_and_spans_models():
    m1, m2 = nn.Linear(2, 2), nn.Linear(3, 3)
    off = argparse.Namespace(ewc_lambda=0.0, ewc_fisher_samples=10, ewc_buffers_on_cpu=True)
    assert anti_forgetting.create_ewc_regularizer(off, [m1, m2]) is None
    on = argparse.Namespace(ewc_lambda=0.5, ewc_fisher_samples=10, ewc_buffers_on_cpu=True)
    reg = anti_forgetting.create_ewc_regularizer(on, [m1, m2])
    assert reg is not None
    # params from both models are tracked
    assert len(reg.params) == len(list(m1.parameters())) + len(list(m2.parameters()))


def test_ewc_fisher_logging_gated_to_main_process():
    import types

    m = nn.Linear(2, 2)
    # No accelerator (single process / CPU) -> the Fisher-phase log is enabled.
    reg = anti_forgetting.EWCRegularizer(m.named_parameters(), lam=1.0, num_fisher_samples=40, store_on_cpu=True)
    assert reg._log_enabled is True
    assert reg._log_every == 4  # max(1, 40 // 10)
    # Non-main rank of a multi-GPU run -> suppressed so the line is not repeated once per rank.
    non_main = types.SimpleNamespace(is_main_process=False, num_processes=4)
    reg2 = anti_forgetting.EWCRegularizer(
        m.named_parameters(), lam=1.0, num_fisher_samples=40, store_on_cpu=True, accelerator=non_main
    )
    assert reg2._log_enabled is False


def test_ewc_buffer_dtype_bf16_stores_bf16_and_computes_difference_in_fp32():
    m = nn.Linear(1, 1, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        m.weight.fill_(0.0)
    reg = anti_forgetting.EWCRegularizer(
        m.named_parameters(), lam=1.0, num_fisher_samples=2, store_on_cpu=True, buffer_dtype=torch.bfloat16
    )
    # both reference buffers live at the buffer dtype (half the VRAM); accumulation is at that dtype
    assert reg.theta_star["weight"].dtype == torch.bfloat16
    assert reg.u["weight"].dtype == torch.bfloat16
    m.weight.grad = torch.tensor([[2.0]], dtype=torch.bfloat16)
    reg.accumulate()
    m.weight.grad = torch.tensor([[4.0]], dtype=torch.bfloat16)
    reg.accumulate()
    assert reg.maybe_finalize()
    assert reg.u["weight"].dtype == torch.bfloat16
    assert reg.u["weight"].float().item() == pytest.approx(3.0, abs=1e-2)  # (2 + 4) / 2
    # penalty still upcasts to fp32 for the theta - theta* difference (the precision that matters):
    # the returned scalar is fp32 even though the buffers are bf16
    with torch.no_grad():
        m.weight += 1.0
    pen = reg.penalty()
    assert pen.dtype == torch.float32
    assert pen.item() > 0.0


def test_create_ewc_regularizer_maps_buffer_dtype_and_warns_for_fp32_weights(caplog):
    import logging

    on_bf16 = argparse.Namespace(
        ewc_lambda=0.5, ewc_fisher_samples=10, ewc_buffers_on_cpu=True, ewc_buffer_dtype="bf16"
    )
    # bf16 weights -> no warning, buffers are bf16
    mbf16 = nn.Linear(2, 2).to(torch.bfloat16)
    with caplog.at_level(logging.WARNING, logger="library.anti_forgetting"):
        reg = anti_forgetting.create_ewc_regularizer(on_bf16, [mbf16])
    assert reg.buffer_dtype == torch.bfloat16
    assert not any("reduced precision" in r.getMessage() for r in caplog.records)
    # fp32 weights + bf16 buffers -> warn about losing drift resolution
    caplog.clear()
    mfp32 = nn.Linear(2, 2)  # fp32 weights
    with caplog.at_level(logging.WARNING, logger="library.anti_forgetting"):
        anti_forgetting.create_ewc_regularizer(on_bf16, [mfp32])
    assert any("fp32" in r.getMessage() for r in caplog.records)


def test_ewc_default_buffer_dtype_is_fp32():
    on = argparse.Namespace(ewc_lambda=0.5, ewc_fisher_samples=10, ewc_buffers_on_cpu=True)  # no ewc_buffer_dtype
    reg = anti_forgetting.create_ewc_regularizer(on, [nn.Linear(2, 2)])
    assert reg.buffer_dtype == torch.float32
    assert reg.theta_star[next(iter(reg.theta_star))].dtype == torch.float32


def test_ewc_fisher_phase_emits_start_progress_and_done_logs(caplog):
    import logging

    m = nn.Linear(1, 1, bias=False)
    with caplog.at_level(logging.INFO, logger="library.anti_forgetting"):
        reg = anti_forgetting.EWCRegularizer(m.named_parameters(), lam=1.0, num_fisher_samples=2, store_on_cpu=True)
        for _ in range(2):
            m.weight.grad = torch.tensor([[1.0]])
            reg.accumulate()
        assert reg.maybe_finalize()
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "Fisher phase started" in msgs
    assert "Fisher phase 1/2" in msgs  # progress line before the final micro-batch
    assert "Fisher phase done" in msgs


def test_is_ewc_enabled():
    assert anti_forgetting.is_ewc_enabled(argparse.Namespace()) is False
    assert anti_forgetting.is_ewc_enabled(argparse.Namespace(ewc_lambda=0.0)) is False
    assert anti_forgetting.is_ewc_enabled(argparse.Namespace(ewc_lambda=0.3)) is True


def test_validator_ewc_supersedes_distillation():
    args = _penalty_args(high=1.0, low=0.5, adaptive=False, ewc_lambda=0.5)
    args.ewc_fisher_samples = 100
    anti_forgetting.verify_anti_forgetting_args(args)
    # EWC wins: distillation weights zeroed
    assert args.distillation_weight_high == 0.0 and args.distillation_weight_low == 0.0


def _ewc_args(lam=0.5, fisher_samples=100, fused=False, blockwise=False, blocks_to_swap=0):
    return argparse.Namespace(
        ewc_lambda=lam,
        ewc_fisher_samples=fisher_samples,
        distillation_weight_high=0.0,
        distillation_weight_low=0.0,
        adaptive_lambda=False,
        fused_backward_pass=fused,
        blockwise_fused_optimizers=blockwise,
        blocks_to_swap=blocks_to_swap,
    )


def test_validator_rejects_ewc_with_zero_fisher_samples():
    with pytest.raises(ValueError):
        anti_forgetting.verify_anti_forgetting_args(_ewc_args(fisher_samples=0))


def test_validator_rejects_ewc_with_fused_optimizer():
    with pytest.raises(ValueError):
        anti_forgetting.verify_anti_forgetting_args(_ewc_args(fused=True))
    with pytest.raises(ValueError):
        anti_forgetting.verify_anti_forgetting_args(_ewc_args(blockwise=True))


def test_validator_rejects_ewc_with_block_swap():
    with pytest.raises(ValueError):
        anti_forgetting.verify_anti_forgetting_args(_ewc_args(blocks_to_swap=10))


def test_validator_allows_ewc_normal_optimizer():
    args = _ewc_args(fused=False, blockwise=False, blocks_to_swap=0)
    anti_forgetting.verify_anti_forgetting_args(args)  # must not raise


def test_validator_oplora_supersedes_distillation():
    args = _penalty_args(high=1.0, low=0.5, adaptive=False)
    args.oplora = True
    anti_forgetting.verify_anti_forgetting_args(args)
    assert args.distillation_weight_high == 0.0 and args.distillation_weight_low == 0.0


def test_validator_disables_adaptive_lambda_with_oplora_only():
    # OPLoRA has no lambda to scale, so adaptive lambda with only OPLoRA active is disabled
    args = _penalty_args(high=0.0, low=0.0, adaptive=True)
    args.oplora = True
    anti_forgetting.verify_anti_forgetting_args(args)
    assert args.adaptive_lambda is False


def test_ewc_args_registered():
    import library.args as args_util

    parser = argparse.ArgumentParser()
    args_util.add_training_arguments(parser, False)
    defaults = parser.parse_args([])
    assert defaults.ewc_lambda == 0.0
    assert defaults.ewc_fisher_samples == 100
    assert defaults.ewc_buffers_on_cpu is False
    on = parser.parse_args(["--ewc_lambda", "0.5", "--ewc_fisher_samples", "50", "--ewc_buffers_on_cpu"])
    assert on.ewc_lambda == 0.5 and on.ewc_fisher_samples == 50 and on.ewc_buffers_on_cpu is True


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


def test_ewc_reference_model_anchors_theta_star_to_separate_model():
    # The trained weight starts at 2.0 but theta* must be the reference weight (5.0), so the penalty
    # measures the distance to the reference, not to the trained model's own starting point.
    trained = nn.Linear(1, 1, bias=False)
    reference = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trained.weight.fill_(2.0)
        reference.weight.fill_(5.0)
    reg = anti_forgetting.EWCRegularizer(
        trained.named_parameters(), lam=1.0, num_fisher_samples=1, store_on_cpu=True,
        reference_params=list(reference.named_parameters()),
    )
    assert reg.theta_star["weight"].item() == pytest.approx(5.0)
    # estimate u = 1, then penalty = (u^T (theta - theta*))^2 = (1 * (2 - 5))^2 = 9, evaluated at the
    # trained start weight (2.0): nonzero from the first step because the anchor is a different model
    trained.weight.grad = torch.tensor([[1.0]])
    reg.accumulate()
    assert reg.maybe_finalize()
    assert reg.penalty().item() == pytest.approx(9.0, abs=1e-4)


def test_ewc_reference_missing_parameter_raises():
    trained = nn.Linear(2, 2)  # has weight and bias
    reference = nn.Linear(2, 2)
    ref_without_bias = [(n, p) for n, p in reference.named_parameters() if n != "bias"]
    with pytest.raises(ValueError, match="missing the trained parameter"):
        anti_forgetting.EWCRegularizer(
            trained.named_parameters(), lam=1.0, num_fisher_samples=1, store_on_cpu=True,
            reference_params=ref_without_bias,
        )


def test_ewc_reference_shape_mismatch_raises():
    trained = nn.Linear(2, 2)
    reference = nn.Linear(3, 3)  # same param names, different shapes
    with pytest.raises(ValueError, match="does not match"):
        anti_forgetting.EWCRegularizer(
            trained.named_parameters(), lam=1.0, num_fisher_samples=1, store_on_cpu=True,
            reference_params=list(reference.named_parameters()),
        )


def test_create_ewc_regularizer_uses_reference_named_params():
    trained = nn.Linear(1, 1, bias=False)
    reference = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trained.weight.fill_(2.0)
        reference.weight.fill_(7.0)
    args = argparse.Namespace(
        ewc_lambda=0.5, ewc_fisher_samples=10, ewc_buffers_on_cpu=True,
        ewc_reference_model_path="/path/to/base_A.safetensors",
    )
    reg = anti_forgetting.create_ewc_regularizer(
        args, [trained], reference_named_params=list(reference.named_parameters())
    )
    assert reg.theta_star["weight"].item() == pytest.approx(7.0)


def test_ewc_reference_model_arg_registered():
    import library.args as args_util

    parser = argparse.ArgumentParser()
    args_util.add_training_arguments(parser, False)
    assert parser.parse_args([]).ewc_reference_model_path is None
    on = parser.parse_args(["--ewc_reference_model_path", "/models/base_A.safetensors"])
    assert on.ewc_reference_model_path == "/models/base_A.safetensors"


def test_validator_warns_when_reference_set_without_ewc(caplog):
    import logging

    args = argparse.Namespace(
        ewc_lambda=0.0,
        ewc_reference_model_path="/models/base_A.safetensors",
        distillation_weight_high=0.0,
        distillation_weight_low=0.0,
        adaptive_lambda=False,
    )
    with caplog.at_level(logging.WARNING, logger="library.anti_forgetting"):
        anti_forgetting.verify_anti_forgetting_args(args)
    assert any("reference model is ignored" in r.getMessage() for r in caplog.records)
