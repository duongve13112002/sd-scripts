"""CPU integration smoke tests for the generalized EMA wiring across model families.

No model is loaded; these only check that every training entry point exposes the EMA
options with sane defaults and that the EMA hooks are provided generically by the base
network trainer (so all network families inherit them).
"""

import argparse

import pytest

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


def _network_setup_parsers():
    import anima_train_network
    import flux_train_network
    import sd3_train_network
    import sdxl_train_network
    import lumina_train_network
    import hunyuan_image_train_network

    return {
        "train_network": train_network.setup_parser,
        "anima_train_network": anima_train_network.setup_parser,
        "flux_train_network": flux_train_network.setup_parser,
        "sd3_train_network": sd3_train_network.setup_parser,
        "sdxl_train_network": sdxl_train_network.setup_parser,
        "lumina_train_network": lumina_train_network.setup_parser,
        "hunyuan_image_train_network": hunyuan_image_train_network.setup_parser,
    }


def _full_finetune_setup_parsers():
    import anima_train
    import flux_train
    import sd3_train
    import sdxl_train
    import lumina_train

    return {
        "anima_train": anima_train.setup_parser,
        "flux_train": flux_train.setup_parser,
        "sd3_train": sd3_train.setup_parser,
        "sdxl_train": sdxl_train.setup_parser,
        "lumina_train": lumina_train.setup_parser,
    }


def _option_strings(parser: argparse.ArgumentParser):
    options = set()
    for action in parser._actions:
        options.update(action.option_strings)
    return options


def _all_setup_parsers():
    parsers = {}
    parsers.update(_network_setup_parsers())
    parsers.update(_full_finetune_setup_parsers())
    return parsers


def test_ema_args_present_on_every_trainer():
    for name, setup_parser in _all_setup_parsers().items():
        options = _option_strings(setup_parser())
        missing = [flag for flag in EMA_FLAGS if flag not in options]
        assert not missing, f"{name} is missing EMA args: {missing}"


def test_ema_defaults():
    parser = train_network.setup_parser()
    args, _ = parser.parse_known_args([])
    assert args.ema is False
    assert args.ema_decay == pytest.approx(0.9999)
    assert args.ema_device == "cuda"
    assert args.ema_param_multiplier == pytest.approx(1.0)
    assert args.ema_resume_path is None


def test_base_create_ema_disabled_by_default():
    trainer = train_network.NetworkTrainer()
    # EMA is opt-in: with --ema off, no EMA instance is created.
    assert trainer.create_ema(argparse.Namespace(ema=False), None, None, None) is None


def test_network_families_inherit_generic_ema_hooks():
    # The base trainer provides the EMA hooks generically; model trainers should not need
    # to override them. This guards against accidental divergence per model.
    base = train_network.NetworkTrainer
    for name, setup_parser in _network_setup_parsers().items():
        if name == "train_network":
            continue
        module = __import__(name)
        trainer_cls = next(
            obj
            for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, base) and obj is not base
        )
        for hook in ("create_ema", "save_ema_network", "remove_ema_network", "sample_ema_images"):
            assert getattr(trainer_cls, hook) is getattr(base, hook), f"{name}.{hook} unexpectedly overrides the base hook"


def test_sample_ema_images_noop_without_ema():
    trainer = train_network.NetworkTrainer()
    # Must be a no-op (and must not raise) when ema is None.
    trainer.sample_ema_images(None, None, argparse.Namespace(ema_sample=True), None, 10, None, None, None, None, None)
