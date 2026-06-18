"""CPU unit tests for the output distillation helper (library/distillation.py).

These are lightweight and run on a CPU-only machine. They exercise the lambda
schedule, the loss math, the noise-level helpers, the enable switch, and that
gradients flow only through the student (teacher is detached). They do not run
any real model; per-model wiring is validated by the GPU smoke script.
"""

import argparse

import pytest
import torch

from library import distillation


def _args(high=0.0, low=0.0, loss_type="l2", huber_c=1.0):
    return argparse.Namespace(
        distillation_weight_high=high,
        distillation_weight_low=low,
        distillation_loss_type=loss_type,
        distillation_huber_c=huber_c,
    )


def test_is_enabled():
    assert not distillation.is_enabled(_args(0.0, 0.0))
    assert distillation.is_enabled(_args(1.0, 0.0))
    assert distillation.is_enabled(_args(0.0, 0.5))
    # missing attributes must default to disabled, not raise
    assert not distillation.is_enabled(argparse.Namespace())


def test_lambda_interpolates_and_clamps():
    args = _args(high=1.0, low=0.2)
    level = torch.tensor([0.0, 0.5, 1.0])
    lam = distillation.lambda_for_noise_level(level, args)
    assert torch.allclose(lam, torch.tensor([0.2, 0.6, 1.0]))
    # out-of-range noise levels are clamped to [0, 1]
    clamped = distillation.lambda_for_noise_level(torch.tensor([-1.0, 2.0]), args)
    assert torch.allclose(clamped, torch.tensor([0.2, 1.0]))


def test_loss_zero_when_predictions_match():
    args = _args(high=1.0, low=1.0)
    pred = torch.randn(4, 3, 8, 8)
    loss = distillation.distillation_loss(
        pred, pred.clone(), torch.ones(4), torch.ones(4), args
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_positive_when_predictions_differ():
    args = _args(high=1.0, low=1.0)
    student = torch.zeros(2, 3, 4, 4)
    teacher = torch.ones(2, 3, 4, 4)
    loss = distillation.distillation_loss(student, teacher, torch.ones(2), torch.ones(2), args)
    # l2 of (0-1)^2 averaged = 1.0, lambda=1, weights=1
    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_higher_noise_gets_more_distillation():
    # high noise weight 1.0, low noise weight 0.0 -> a noisy sample is penalized,
    # a clean sample is not, for the same per-element difference.
    args = _args(high=1.0, low=0.0)
    student = torch.zeros(2, 3, 4, 4)
    teacher = torch.ones(2, 3, 4, 4)
    noisy = distillation.distillation_loss(student, teacher, torch.tensor([1.0, 1.0]), torch.ones(2), args)
    clean = distillation.distillation_loss(student, teacher, torch.tensor([0.0, 0.0]), torch.ones(2), args)
    assert noisy.item() > clean.item()
    assert clean.item() == pytest.approx(0.0, abs=1e-6)


def test_gradient_flows_only_through_student():
    args = _args(high=1.0, low=1.0)
    student = torch.zeros(2, 3, 4, 4, requires_grad=True)
    teacher = torch.ones(2, 3, 4, 4, requires_grad=True)
    loss = distillation.distillation_loss(student, teacher, torch.ones(2), torch.ones(2), args)
    loss.backward()
    assert student.grad is not None and torch.any(student.grad != 0)
    # teacher is detached inside the helper, so no gradient reaches it
    assert teacher.grad is None


def test_loss_weights_scale_the_term():
    args = _args(high=1.0, low=1.0)
    student = torch.zeros(2, 3, 4, 4)
    teacher = torch.ones(2, 3, 4, 4)
    base = distillation.distillation_loss(student, teacher, torch.ones(2), torch.ones(2), args)
    scaled = distillation.distillation_loss(student, teacher, torch.ones(2), torch.full((2,), 2.0), args)
    assert scaled.item() == pytest.approx(2.0 * base.item(), rel=1e-5)


def test_huber_matches_l2_for_small_diff():
    # For |diff| < delta, Huber == 0.5 * diff^2, so it is half the plain L2 term.
    student = torch.zeros(2, 3, 4, 4)
    teacher = torch.full((2, 3, 4, 4), 0.5)
    l2 = distillation.distillation_loss(student, teacher, torch.ones(2), torch.ones(2), _args(1.0, 1.0, "l2"))
    huber = distillation.distillation_loss(
        student, teacher, torch.ones(2), torch.ones(2), _args(1.0, 1.0, "huber", huber_c=1.0)
    )
    assert huber.item() == pytest.approx(0.5 * l2.item(), rel=1e-5)


def test_noise_level_from_sigmas_collapses_to_batch():
    sigmas = torch.tensor([0.1, 0.9]).reshape(2, 1, 1, 1)
    level = distillation.normalized_noise_level_from_sigmas(sigmas)
    assert level.shape == (2,)
    assert torch.allclose(level, torch.tensor([0.1, 0.9]))


def test_noise_level_from_timesteps_normalizes():
    timesteps = torch.tensor([0, 500, 999])
    level = distillation.normalized_noise_level_from_timesteps(timesteps, 1000)
    assert torch.allclose(level, torch.tensor([0.0, 0.5, 0.999]))


def test_distillation_args_registered_on_training_parser():
    # add_training_arguments is called by every network trainer's setup_parser
    # (and by the full-FT scripts), so registering here covers all families.
    import argparse as _argparse

    import library.args as args_util

    parser = _argparse.ArgumentParser()
    args_util.add_training_arguments(parser, False)
    args = parser.parse_args(
        ["--distillation_weight_high", "1.0", "--distillation_weight_low", "0.1", "--distillation_loss_type", "huber"]
    )
    assert args.distillation_weight_high == 1.0
    assert args.distillation_weight_low == 0.1
    assert args.distillation_loss_type == "huber"
    assert args.distillation_huber_c == 1.0
    assert distillation.is_enabled(args)
    # defaults keep distillation disabled
    assert not distillation.is_enabled(parser.parse_args([]))
