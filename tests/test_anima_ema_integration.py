"""CPU integration smoke tests for wiring the Anima EMA feature into the training scripts.

No model is loaded; these only check that the argument parsers expose the EMA options with
sane defaults and that the trainer classes expose the expected EMA hooks.
"""

import argparse

import pytest

import anima_train
import anima_train_network
import train_network


EMA_FLAGS = [
    "--ema",
    "--ema_decay",
    "--ema_device",
    "--ema_use_num_updates",
    "--ema_use_feedback",
    "--ema_param_multiplier",
    "--ema_resume_path",
    "--ema_sample",
]


def _option_strings(parser: argparse.ArgumentParser):
    options = set()
    for action in parser._actions:
        options.update(action.option_strings)
    return options


@pytest.mark.parametrize("setup_parser", [anima_train.setup_parser, anima_train_network.setup_parser])
def test_ema_args_present(setup_parser):
    options = _option_strings(setup_parser())
    missing = [flag for flag in EMA_FLAGS if flag not in options]
    assert not missing, f"missing EMA args: {missing}"


def test_ema_defaults():
    parser = anima_train.setup_parser()
    args, _ = parser.parse_known_args([])
    assert args.ema is False
    assert args.ema_decay == pytest.approx(0.9999)
    assert args.ema_device == "cuda"
    assert args.ema_param_multiplier == pytest.approx(1.0)
    assert args.ema_resume_path is None


def test_base_trainer_ema_hooks_are_noops():
    trainer = train_network.NetworkTrainer()
    # Base hooks must default to disabled / no-op so non-Anima models are unaffected.
    assert trainer.create_ema(argparse.Namespace(ema=False), None, None, None) is None
    assert trainer.save_ema_network(None, "x", None, None, {}) is None
    assert trainer.remove_ema_network("x") is None


def test_anima_network_trainer_overrides_hooks():
    trainer = anima_train_network.AnimaNetworkTrainer()
    base = train_network.NetworkTrainer
    for name in ("create_ema", "save_ema_network", "remove_ema_network", "sample_ema_images"):
        assert type(trainer).__dict__.get(name) is not None, f"{name} not overridden on AnimaNetworkTrainer"
        assert getattr(type(trainer), name) is not getattr(base, name), f"{name} still points at the base no-op"

    # create_ema must still return None when EMA is disabled.
    assert trainer.create_ema(argparse.Namespace(ema=False), None, None, None) is None
