"""Tests for nanosaur_train_network.py (no GPU required, mock-based)."""

import argparse
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from library import nanosaur_models, nanosaur_utils
from nanosaur_train_network import NanoSaurNetworkTrainer


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def trainer():
    return NanoSaurNetworkTrainer()


@pytest.fixture
def mock_args():
    args = MagicMock()
    args.pretrained_model_name_or_path = "test_model.safetensors"
    args.text_encoder = "test_te.safetensors"
    args.vae = "test_vae.safetensors"
    args.disable_mmap_load_safetensors = False
    args.fp8_base = False
    args.blocks_to_swap = None
    args.network_train_unet_only = False
    args.cache_text_encoder_outputs = True
    args.cache_text_encoder_outputs_to_disk = False
    args.text_encoder_batch_size = 4
    args.skip_cache_check = False
    args.sample_prompts = None
    args.sample_at_first = False
    args.sample_every_n_steps = None
    args.sample_every_n_epochs = None
    args.gradient_checkpointing = False
    args.time_sampling_alpha = 2.0
    return args


@pytest.fixture
def mock_accelerator():
    acc = MagicMock()
    acc.device = torch.device("cpu")
    acc.prepare.side_effect = lambda *args, **kwargs: args[0] if len(args) == 1 else args
    acc.unwrap_model.side_effect = lambda x: x
    return acc


@pytest.fixture
def small_model():
    """A tiny NanoSaur model for testing without full weights."""
    return nanosaur_models.NanoSaurTransformer2DModel(
        in_channels=8,
        num_groups=4,
        hidden_size=32,
        decoder_hidden_size=64,
        num_encoder_blocks=4,
        num_decoder_blocks=1,
        num_text_blocks=1,
        patch_size=1,
        txt_embed_dim=16,
    )


# ─── assert_extra_args ────────────────────────────────────────────────────────


class TestAssertExtraArgs:
    def test_sets_train_gemma3(self, trainer, mock_args):
        train_dg = MagicMock()
        train_dg.verify_bucket_reso_steps = MagicMock()
        val_dg = MagicMock()
        val_dg.verify_bucket_reso_steps = MagicMock()

        trainer.assert_extra_args(mock_args, train_dg, val_dg)

        # train_gemma3 is the inverse of network_train_unet_only
        assert trainer.train_gemma3 is (not mock_args.network_train_unet_only)

    def test_verifies_bucket_reso_steps(self, trainer, mock_args):
        train_dg = MagicMock()
        train_dg.verify_bucket_reso_steps = MagicMock()
        val_dg = MagicMock()
        val_dg.verify_bucket_reso_steps = MagicMock()

        trainer.assert_extra_args(mock_args, train_dg, val_dg)

        # NanoSaur must call verify_bucket_reso_steps with stride=16 (16x VAE downscale).
        # The base class may also call it with a different stride first, so use assert_any_call.
        train_dg.verify_bucket_reso_steps.assert_any_call(16)
        val_dg.verify_bucket_reso_steps.assert_any_call(16)

    def test_enables_cache_when_disk_cache_set(self, trainer, mock_args):
        mock_args.cache_text_encoder_outputs = False
        mock_args.cache_text_encoder_outputs_to_disk = True
        train_dg = MagicMock()
        train_dg.verify_bucket_reso_steps = MagicMock()
        val_dg = MagicMock()
        val_dg.verify_bucket_reso_steps = MagicMock()

        trainer.assert_extra_args(mock_args, train_dg, val_dg)

        assert mock_args.cache_text_encoder_outputs is True

    def test_val_dataset_group_optional(self, trainer, mock_args):
        train_dg = MagicMock()
        train_dg.verify_bucket_reso_steps = MagicMock()

        # Should not raise when val_dataset_group is None
        trainer.assert_extra_args(mock_args, train_dg, None)


# ─── load_target_model ────────────────────────────────────────────────────────


class TestLoadTargetModel:
    def test_returns_correct_structure(self, trainer, mock_args, mock_accelerator):
        mock_model = MagicMock(spec=nanosaur_models.NanoSaurTransformer2DModel)
        mock_model.dtype = torch.float32
        mock_gemma3 = MagicMock()
        mock_vae = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch("library.nanosaur_utils.load_nanosaur_model", return_value=mock_model) as mock_load_model,
            patch("library.nanosaur_utils.load_nanosaur_text_encoder", return_value=(mock_tokenizer, mock_gemma3)),
            patch("library.nanosaur_utils.load_nanosaur_vae", return_value=mock_vae),
        ):
            version, te_list, vae, model = trainer.load_target_model(
                mock_args, torch.float32, mock_accelerator
            )

        assert version == nanosaur_utils.MODEL_VERSION_NANOSAUR
        assert te_list == [mock_gemma3]
        assert vae is mock_vae
        assert model is mock_model
        mock_load_model.assert_called_once()

    def test_block_swap_enabled_when_requested(self, trainer, mock_args, mock_accelerator):
        mock_args.blocks_to_swap = 5
        mock_model = MagicMock(spec=nanosaur_models.NanoSaurTransformer2DModel)
        mock_model.dtype = torch.float32
        mock_gemma3 = MagicMock()
        mock_tokenizer = MagicMock()
        mock_vae = MagicMock()

        with (
            patch("library.nanosaur_utils.load_nanosaur_model", return_value=mock_model),
            patch("library.nanosaur_utils.load_nanosaur_text_encoder", return_value=(mock_tokenizer, mock_gemma3)),
            patch("library.nanosaur_utils.load_nanosaur_vae", return_value=mock_vae),
        ):
            trainer.load_target_model(mock_args, torch.float32, mock_accelerator)

        mock_model.enable_block_swap.assert_called_once_with(5, mock_accelerator.device)
        assert trainer.is_swapping_blocks is True


# ─── Strategy factories ───────────────────────────────────────────────────────


class TestStrategyFactories:
    def setup_method(self):
        # Simulate load_target_model having set the tokenizer
        from library.nanosaur_utils import NanoSaurSentencePieceTokenizer
        mock_sp = MagicMock()
        mock_sp.EncodeAsIds.return_value = [10, 20, 30]
        self._tokenizer = MagicMock(spec=NanoSaurSentencePieceTokenizer)
        self._tokenizer.max_length = 128
        self._tokenizer.bos_token_id = 2
        self._tokenizer.pad_token_id = 0

    def test_get_latents_caching_strategy(self):
        from library.strategy_nanosaur import NanoSaurLatentsCachingStrategy
        trainer = NanoSaurNetworkTrainer()
        args = MagicMock()
        args.cache_latents_to_disk = False
        args.vae_batch_size = 4
        strategy = trainer.get_latents_caching_strategy(args)
        assert isinstance(strategy, NanoSaurLatentsCachingStrategy)

    def test_get_text_encoding_strategy(self):
        from library.strategy_nanosaur import NanoSaurTextEncodingStrategy
        trainer = NanoSaurNetworkTrainer()
        args = MagicMock()
        strategy = trainer.get_text_encoding_strategy(args)
        assert isinstance(strategy, NanoSaurTextEncodingStrategy)

    def test_get_text_encoder_outputs_caching_strategy_enabled(self):
        from library.strategy_nanosaur import NanoSaurTextEncoderOutputsCachingStrategy
        trainer = NanoSaurNetworkTrainer()
        trainer.train_gemma3 = False
        args = MagicMock()
        args.cache_text_encoder_outputs = True
        args.cache_text_encoder_outputs_to_disk = False
        args.text_encoder_batch_size = 8
        args.skip_cache_check = False
        strategy = trainer.get_text_encoder_outputs_caching_strategy(args)
        assert isinstance(strategy, NanoSaurTextEncoderOutputsCachingStrategy)

    def test_get_text_encoder_outputs_caching_strategy_disabled(self):
        trainer = NanoSaurNetworkTrainer()
        args = MagicMock()
        args.cache_text_encoder_outputs = False
        strategy = trainer.get_text_encoder_outputs_caching_strategy(args)
        assert strategy is None

    def test_get_noise_scheduler(self):
        from nanosaur_train_network import _NanoSaurFlowScheduler
        trainer = NanoSaurNetworkTrainer()
        args = MagicMock()
        scheduler = trainer.get_noise_scheduler(args, torch.device("cpu"))
        assert isinstance(scheduler, _NanoSaurFlowScheduler)
        assert scheduler.num_train_timesteps == 1000


# ─── Noise prediction and loss ────────────────────────────────────────────────


class TestGetNoisePredAndTarget:
    def test_velocity_shapes_match(self, trainer, mock_args, mock_accelerator, small_model):
        trainer.train_gemma3 = False
        noise_sched = MagicMock()
        noise_sched.num_train_timesteps = 1000

        batch_size, ch, h, w = 1, 8, 8, 8
        latents = torch.randn(batch_size, ch, h, w)
        batch = {}
        hidden_states = torch.randn(batch_size, 4, 16)  # (B, L, D)

        with mock_accelerator.autocast():
            pass  # mock context manager

        mock_network = MagicMock()
        mock_network.set_multiplier = MagicMock()

        pred, target, timesteps, weighting = trainer.get_noise_pred_and_target(
            args=mock_args,
            accelerator=mock_accelerator,
            noise_scheduler=noise_sched,
            latents=latents,
            batch=batch,
            text_encoder_conds=(hidden_states, None, None),
            dit=small_model,
            network=mock_network,
            weight_dtype=torch.float32,
            train_unet=True,
            is_train=False,
        )

        assert pred.shape == (batch_size, ch, h, w)
        assert target.shape == (batch_size, ch, h, w)
        assert timesteps.shape == (batch_size,)
        assert weighting is None

    def test_velocity_pred_is_finite(self, trainer, mock_args, mock_accelerator, small_model):
        noise_sched = MagicMock()
        noise_sched.num_train_timesteps = 1000
        mock_network = MagicMock()

        latents = torch.randn(1, 8, 8, 8)
        hidden_states = torch.randn(1, 4, 16)

        pred, target, _, _ = trainer.get_noise_pred_and_target(
            args=mock_args,
            accelerator=mock_accelerator,
            noise_scheduler=noise_sched,
            latents=latents,
            batch={},
            text_encoder_conds=(hidden_states, None, None),
            dit=small_model,
            network=mock_network,
            weight_dtype=torch.float32,
            train_unet=True,
            is_train=False,
        )

        assert torch.isfinite(pred).all()
        assert torch.isfinite(target).all()


# ─── Metadata ─────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_update_metadata(self, trainer, mock_args):
        metadata = {}
        trainer.update_metadata(metadata, mock_args)
        assert "ss_model_type" in metadata
        assert metadata["ss_model_type"] == "nanosaur"

    def test_is_text_encoder_not_needed_with_cache(self, trainer, mock_args):
        mock_args.cache_text_encoder_outputs = True
        with patch.object(trainer, "is_train_text_encoder", return_value=False):
            result = trainer.is_text_encoder_not_needed_for_training(mock_args)
        assert result is True

    def test_is_text_encoder_needed_when_training(self, trainer, mock_args):
        mock_args.cache_text_encoder_outputs = True
        with patch.object(trainer, "is_train_text_encoder", return_value=True):
            result = trainer.is_text_encoder_not_needed_for_training(mock_args)
        assert result is False
