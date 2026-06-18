"""CPU unit tests for the async disk-write caching helpers on the base caching strategies.

These verify the overlap-write mechanism (and its sync fallback) without needing a VAE,
a text encoder or a GPU.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from library.strategy_base import LatentsCachingStrategy, TextEncoderOutputsCachingStrategy


def _new_latents_strategy():
    # Bypass the abstract __init__; the async helpers and save path do not depend on it.
    return LatentsCachingStrategy.__new__(LatentsCachingStrategy)


def _new_te_strategy():
    return TextEncoderOutputsCachingStrategy.__new__(TextEncoderOutputsCachingStrategy)


def test_save_latents_async_writes_file(tmp_path):
    strat = _new_latents_strategy()
    executor = ThreadPoolExecutor(max_workers=2)
    strat.set_async_write_executor(executor)
    try:
        npz_path = str(tmp_path / "a.npz")
        strat.save_latents_to_disk(npz_path, torch.randn(4, 8, 8), [64, 64], [0, 0, 64, 64])
        strat.wait_for_async_writes()
    finally:
        strat.set_async_write_executor(None)
        executor.shutdown(wait=True)

    data = np.load(npz_path)
    assert data["latents"].shape == (4, 8, 8)
    assert tuple(data["original_size"]) == (64, 64)


def test_save_latents_sync_fallback(tmp_path):
    strat = _new_latents_strategy()  # no executor set -> writes inline
    npz_path = str(tmp_path / "b.npz")
    strat.save_latents_to_disk(npz_path, torch.randn(4, 8, 8), [64, 64], [0, 0, 64, 64])
    assert (tmp_path / "b.npz").exists()


def test_save_outputs_copies_before_handoff(tmp_path):
    strat = _new_te_strategy()
    executor = ThreadPoolExecutor(max_workers=2)
    strat.set_async_write_executor(executor)
    try:
        npz_path = str(tmp_path / "c.npz")
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        strat.save_outputs_npz(npz_path, emb=arr)
        # Mutate the source immediately: a correct implementation copied it before handoff.
        arr[:] = -999.0
        strat.wait_for_async_writes()
    finally:
        strat.set_async_write_executor(None)
        executor.shutdown(wait=True)

    data = np.load(npz_path)
    assert data["emb"][0, 0] == 0.0


def test_wait_reraises_write_errors():
    strat = _new_te_strategy()
    executor = ThreadPoolExecutor(max_workers=1)
    strat.set_async_write_executor(executor)
    try:
        # An unwritable path makes the background write raise; wait must surface it.
        strat.save_outputs_npz("/nonexistent_dir_xyz/should_fail.npz", emb=np.zeros(2, dtype=np.float32))
        with pytest.raises(Exception):
            strat.wait_for_async_writes()
    finally:
        strat.set_async_write_executor(None)
        executor.shutdown(wait=True)
