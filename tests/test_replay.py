"""CPU unit tests for anti-forgetting replay (--replay_ratio + is_replay subset flag).

These are lightweight and CPU-only. They exercise the epoch-level replay-ratio math in
``config_util.apply_replay_ratio`` (scaling replay images' num_repeats), the validation
errors, the ``is_replay`` subset flag plumbing, and the CLI arg registration. No real
images or models are loaded.
"""

import argparse
import types

import pytest

from library import config_util
from library.dataset import ImageInfo


def _img(key, num_repeats, is_replay):
    info = ImageInfo(key, num_repeats, "caption", False, key)
    info.is_replay = is_replay
    return info


def _dataset(infos):
    # apply_replay_ratio only touches dataset.image_data.values()
    return types.SimpleNamespace(image_data={info.image_key: info for info in infos})


def _totals(datasets):
    new_total = replay_total = 0
    for d in datasets:
        for info in d.image_data.values():
            if info.is_replay:
                replay_total += info.num_repeats
            else:
                new_total += info.num_repeats
    return new_total, replay_total


def test_replay_ratio_zero_is_noop():
    infos = [_img("n1", 1, False), _img("r1", 1, True)]
    ds = _dataset(infos)
    config_util.apply_replay_ratio([ds], 0.0)
    assert infos[0].num_repeats == 1 and infos[1].num_repeats == 1


def test_replay_ratio_half_balances_counts():
    new = [_img(f"n{i}", 1, False) for i in range(100)]
    replay = [_img(f"r{i}", 1, True) for i in range(10)]
    ds = _dataset(new + replay)
    config_util.apply_replay_ratio([ds], 0.5)
    new_total, replay_total = _totals([ds])
    # target 0.5 with these sizes is exactly reachable: 10 replay images -> 10 repeats each
    assert replay_total / (replay_total + new_total) == pytest.approx(0.5, abs=1e-6)


def test_replay_ratio_is_approximate_but_close():
    new = [_img(f"n{i}", 1, False) for i in range(100)]
    replay = [_img(f"r{i}", 1, True) for i in range(10)]
    ds = _dataset(new + replay)
    config_util.apply_replay_ratio([ds], 0.3)
    new_total, replay_total = _totals([ds])
    achieved = replay_total / (replay_total + new_total)
    # integer num_repeats makes it approximate; should still be near the target
    assert achieved == pytest.approx(0.3, abs=0.05)


def test_replay_spans_multiple_datasets_globally():
    ds1 = _dataset([_img(f"n{i}", 1, False) for i in range(50)])
    ds2 = _dataset([_img(f"r{i}", 1, True) for i in range(5)])
    config_util.apply_replay_ratio([ds1, ds2], 0.5)
    new_total, replay_total = _totals([ds1, ds2])
    assert replay_total / (replay_total + new_total) == pytest.approx(0.5, abs=1e-6)


def test_replay_never_drops_below_one_repeat():
    # tiny target with many replay images must not zero anyone out
    new = [_img(f"n{i}", 1, False) for i in range(10)]
    replay = [_img(f"r{i}", 1, True) for i in range(100)]
    ds = _dataset(new + replay)
    config_util.apply_replay_ratio([ds], 0.01)
    assert all(info.num_repeats >= 1 for info in ds.image_data.values())


def test_replay_ratio_out_of_range_raises():
    ds = _dataset([_img("n1", 1, False), _img("r1", 1, True)])
    with pytest.raises(ValueError):
        config_util.apply_replay_ratio([ds], 1.0)


def test_replay_ratio_without_replay_subset_raises():
    ds = _dataset([_img("n1", 1, False), _img("n2", 1, False)])
    with pytest.raises(ValueError):
        config_util.apply_replay_ratio([ds], 0.5)


def test_replay_ratio_without_new_images_raises():
    ds = _dataset([_img("r1", 1, True), _img("r2", 1, True)])
    with pytest.raises(ValueError):
        config_util.apply_replay_ratio([ds], 0.5)


def test_is_replay_flag_threads_through_subsets():
    from library.subset import DreamBoothSubset, FineTuningSubset

    common = dict(
        num_repeats=1,
        shuffle_caption=False,
        caption_separator=",",
        keep_tokens=0,
        keep_tokens_separator="",
        secondary_separator=None,
        enable_wildcard=False,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=1,
        token_warmup_step=0,
    )
    db = DreamBoothSubset(
        image_dir="dir", is_reg=False, class_tokens=None, caption_extension=".txt",
        cache_info=False, alpha_mask=False, is_replay=True, **common,
    )
    assert db.is_replay is True
    ft = FineTuningSubset(image_dir="dir", metadata_file="m.json", alpha_mask=False, **common)
    assert ft.is_replay is False  # default off


def test_base_subset_params_and_schema_have_is_replay():
    assert hasattr(config_util.BaseSubsetParams(), "is_replay")
    assert config_util.BaseSubsetParams().is_replay is False
    assert config_util.ConfigSanitizer.SUBSET_ASCENDABLE_SCHEMA.get("is_replay") is bool


def test_is_replay_threads_through_blueprint():
    # TOML subset key -> schema -> blueprint -> subset params
    san = config_util.ConfigSanitizer(True, True, False, True)
    gen = config_util.BlueprintGenerator(san)
    user_config = {
        "datasets": [
            {
                "resolution": 512,
                "batch_size": 1,
                "subsets": [
                    {"image_dir": "new_imgs", "num_repeats": 1},
                    {"image_dir": "old_imgs", "num_repeats": 1, "is_replay": True},
                ],
            }
        ]
    }
    blueprint = gen.generate(user_config, argparse.Namespace())
    flags = [s.params.is_replay for s in blueprint.dataset_group.datasets[0].subsets]
    assert flags == [False, True]


def test_replay_ratio_arg_registered():
    import library.args as args_util

    parser = argparse.ArgumentParser()
    args_util.add_dataset_arguments(parser, True, True, True)
    args = parser.parse_args([])
    assert args.replay_ratio == 0.0
    args = parser.parse_args(["--replay_ratio", "0.25"])
    assert args.replay_ratio == 0.25
