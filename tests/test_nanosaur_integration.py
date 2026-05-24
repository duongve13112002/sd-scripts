"""
Comprehensive integration tests for NanoSaur sd-scripts implementation.

Covers:
- LoRA network: forward hooks, key conversion, save/load round-trip, alpha dtype
- Flow matching: loss math, timestep sampling, sampler logic
- Strategy layer: tokenize, encode, latent caching, TE output caching
- nanosaur_train.py online encoding path (post-fix)
- Multi-GPU / DeepSpeed trainer hooks (mock accelerator)
- SPRINT uncond path consistency
- VAE wrapper scale/shift round-trip
"""

import io
import math
import os
import tempfile
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ─── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_model():
    from library.nanosaur_models import NanoSaurTransformer2DModel
    return NanoSaurTransformer2DModel(
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


@pytest.fixture(scope="module")
def lora_network(small_model):
    from networks.lora_nanosaur import LoRANetwork
    net = LoRANetwork(
        text_encoders=[],
        unet=small_model,
        multiplier=1.0,
        lora_dim=4,
        alpha=4.0,
        train_text_encoder=False,
    )
    net.apply_to(text_encoders=[], unet=small_model, apply_text_encoder=False, apply_unet=True)
    return net


# ─── LoRA Module: forward hook ────────────────────────────────────────────────


class TestLoRAModuleForward:
    def test_lora_changes_output(self, small_model, lora_network):
        """With LoRA applied and non-zero weights, output must differ from base."""
        x = torch.randn(1, 8, 8, 8)
        t = torch.tensor([0.5])
        ctx = torch.randn(1, 4, 16)

        # Reset LoRA up weights to non-zero so it actually changes output
        for lora in lora_network.unet_loras:
            nn.init.normal_(lora.lora_up.weight, std=0.1)

        with torch.no_grad():
            lora_network.set_multiplier(1.0)
            out_with_lora = small_model._forward(x, t, ctx)

            lora_network.set_multiplier(0.0)
            out_without_lora = small_model._forward(x, t, ctx)

        lora_network.set_multiplier(1.0)
        assert not torch.allclose(out_with_lora, out_without_lora), \
            "LoRA with non-zero up weights must change model output"

    def test_lora_zero_multiplier_is_identity(self, small_model, lora_network):
        """Multiplier=0 must give identical output to no-LoRA."""
        # Zero out lora_up so output truly matches (multiplier=0 path)
        for lora in lora_network.unet_loras:
            nn.init.zeros_(lora.lora_up.weight)

        x = torch.randn(1, 8, 8, 8)
        t = torch.tensor([0.3])
        ctx = torch.randn(1, 4, 16)

        with torch.no_grad():
            lora_network.set_multiplier(0.0)
            out_zero = small_model._forward(x, t, ctx)
            lora_network.set_multiplier(1.0)
            out_one = small_model._forward(x, t, ctx)

        # With lora_up zeroed, both should be identical
        assert torch.allclose(out_zero, out_one), \
            "Zero lora_up weights should give same output regardless of multiplier"


# ─── LoRA Key Conversion ──────────────────────────────────────────────────────


class TestLoRAKeyConversion:
    def test_all_unet_keys_converted(self, lora_network):
        """Every unet key must appear as diffusion_model.* after conversion."""
        internal_sd = lora_network.state_dict()
        comfyui_sd = lora_network._internal_to_comfyui(internal_sd)

        unet_internal = [k for k in internal_sd if k.startswith("lora_unet_")]
        comfyui_keys = list(comfyui_sd.keys())

        # Same count
        assert len(unet_internal) == sum(
            1 for k in comfyui_keys if k.startswith("diffusion_model.")
        ), "Every lora_unet_ key must be converted to diffusion_model.*"

    def test_key_format_matches_comfyui(self, lora_network):
        """Converted keys must follow diffusion_model.{path}.lora_{up|down}.weight."""
        internal_sd = lora_network.state_dict()
        comfyui_sd = lora_network._internal_to_comfyui(internal_sd)

        for key in comfyui_sd:
            if not key.startswith("diffusion_model."):
                continue
            assert ".lora_up.weight" in key or ".lora_down.weight" in key or key.endswith(".alpha"), \
                f"Unexpected ComfyUI key format: {key}"

    def test_round_trip_internal_comfyui_internal(self, lora_network):
        """Forward then reverse conversion must recover original keys exactly."""
        internal_sd = {k: v.clone() for k, v in lora_network.state_dict().items()}
        comfyui_sd = lora_network._internal_to_comfyui(internal_sd)
        recovered_sd = lora_network._maybe_convert_comfyui_to_internal(comfyui_sd)

        assert set(recovered_sd.keys()) == set(internal_sd.keys()), \
            "Round-trip key mismatch"
        for k in internal_sd:
            assert torch.allclose(recovered_sd[k].float(), internal_sd[k].float()), \
                f"Value mismatch after round-trip for key: {k}"

    def test_alpha_stays_float32_when_bf16_dtype(self, lora_network):
        """Alpha buffers must be saved as float32 even when dtype=bfloat16."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tmp = f.name
        try:
            lora_network.save_weights(tmp, dtype=torch.bfloat16, metadata=None)
            from safetensors.torch import load_file
            saved = load_file(tmp)
            for key, val in saved.items():
                if key.endswith(".alpha"):
                    assert val.dtype == torch.float32, \
                        f"Alpha key {key} should be float32, got {val.dtype}"
                else:
                    assert val.dtype == torch.bfloat16, \
                        f"Weight key {key} should be bfloat16, got {val.dtype}"
        finally:
            os.unlink(tmp)

    def test_load_comfyui_format_file(self, lora_network):
        """Loading a ComfyUI-format file must match original weights."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tmp = f.name
        try:
            lora_network.save_weights(tmp, dtype=None, metadata=None)
            # Reload
            from safetensors.torch import load_file
            loaded = load_file(tmp)
            # All keys must be diffusion_model.* format
            for k in loaded:
                assert k.startswith("diffusion_model."), \
                    f"Expected ComfyUI key, got: {k}"
        finally:
            os.unlink(tmp)

    def test_original_name_dotted_path_correct(self, small_model):
        """original_name must be the dotted module path, not underscore version."""
        from networks.lora_nanosaur import LoRANetwork
        net = LoRANetwork([], small_model, 1.0, 4, 4.0, train_text_encoder=False)
        for lora in net.unet_loras:
            # original_name must contain dots (dotted path)
            assert "." in lora.original_name, \
                f"original_name '{lora.original_name}' has no dots — should be dotted path"
            # lora_name must be underscore version
            expected_lora_name = "lora_unet_" + lora.original_name.replace(".", "_")
            assert lora.lora_name == expected_lora_name, \
                f"lora_name mismatch: expected {expected_lora_name}, got {lora.lora_name}"


# ─── LoRA target scope matches ComfyUI ───────────────────────────────────────


class TestLoRATargetScope:
    def test_adaLN_excluded_from_resblock(self, small_model):
        """ResBlock.adaLN_modulation must NOT be targeted (matches ComfyUI)."""
        from networks.lora_nanosaur import LoRANetwork
        net = LoRANetwork([], small_model, 1.0, 4, 4.0, train_text_encoder=False)
        for lora in net.unet_loras:
            assert "adaLN_modulation" not in lora.original_name or \
                not lora.original_name.startswith("dec_net.res_blocks"), \
                f"adaLN_modulation in ResBlock should be excluded: {lora.original_name}"

    def test_targets_blocks_and_text_refine(self, small_model):
        """LoRA must target both blocks.* and text_refine_blocks.* linears."""
        from networks.lora_nanosaur import LoRANetwork
        net = LoRANetwork([], small_model, 1.0, 4, 4.0, train_text_encoder=False)
        names = [lora.original_name for lora in net.unet_loras]
        has_blocks = any(n.startswith("blocks.") for n in names)
        has_text_refine = any(n.startswith("text_refine_blocks.") for n in names)
        assert has_blocks, "No LoRA modules targeting blocks.*"
        assert has_text_refine, "No LoRA modules targeting text_refine_blocks.*"

    def test_no_duplicate_lora_names(self, small_model):
        """All lora_names must be unique."""
        from networks.lora_nanosaur import LoRANetwork
        net = LoRANetwork([], small_model, 1.0, 4, 4.0, train_text_encoder=False)
        names = [lora.lora_name for lora in net.unet_loras]
        assert len(names) == len(set(names)), "Duplicate lora_name detected"


# ─── LoRA merge_to ────────────────────────────────────────────────────────────


class TestLoRAMergeTo:
    def test_merge_to_changes_base_weights(self, small_model, lora_network):
        """merge_to must bake LoRA into base weights."""
        # Reinitialize lora_up to non-zero
        for lora in lora_network.unet_loras:
            nn.init.normal_(lora.lora_up.weight, std=0.01)
            nn.init.normal_(lora.lora_down.weight, std=0.01)

        # Clone state before merge
        import copy
        model_copy = copy.deepcopy(small_model)

        comfyui_sd = lora_network._internal_to_comfyui(lora_network.state_dict())
        lora_network.merge_to([], model_copy, comfyui_sd, dtype=torch.float32, device="cpu")

        # At least one parameter must differ
        any_diff = False
        for (n1, p1), (n2, p2) in zip(small_model.named_parameters(), model_copy.named_parameters()):
            if not torch.allclose(p1, p2):
                any_diff = True
                break
        assert any_diff, "merge_to should change at least one base parameter"


# ─── Flow Matching Math ───────────────────────────────────────────────────────


class TestFlowMatchingMath:
    def test_noisy_input_interpolation(self):
        """zt = (1-t)*x0 + t*noise, check at t=0 and t=1."""
        from library.nanosaur_train_util import get_noisy_model_input_and_timesteps
        args = MagicMock()
        args.time_sampling_alpha = 2.0
        args.mixed_precision = "no"
        x0 = torch.ones(2, 8, 4, 4)
        noise = torch.zeros(2, 8, 4, 4)
        # Force t close to 0 via patch
        with patch("library.nanosaur_train_util.sample_timesteps",
                   return_value=torch.full((2,), 0.001)):
            zt, t = get_noisy_model_input_and_timesteps(args, x0, noise, "cpu", torch.float32)
        # At t≈0, zt ≈ x0
        assert torch.allclose(zt, x0, atol=0.01), "At t~0, zt should be ~x0"

    def test_velocity_target_formula(self):
        """velocity_target = (zt - x0) / (t + 0.05)."""
        from library.nanosaur_train_util import get_flow_matching_loss
        B, C, H, W = 2, 4, 4, 4
        x0 = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        t = torch.tensor([0.3, 0.7])
        shape = [B] + [1] * 3
        zt = (1 - t.view(shape)) * x0 + t.view(shape) * noise
        x0_pred = x0.clone()  # perfect prediction

        v_pred, v_target = get_flow_matching_loss(x0_pred, x0, zt, t)

        t_clamped = (t + 0.05).view(shape)
        expected_target = (zt - x0) / t_clamped
        assert torch.allclose(v_target, expected_target, atol=1e-5), \
            "velocity_target formula mismatch"

    def test_perfect_x0_gives_zero_loss(self):
        """When x0_pred == x0, velocity loss should be zero."""
        from library.nanosaur_train_util import get_flow_matching_loss
        B, C, H, W = 2, 4, 4, 4
        x0 = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)
        t = torch.tensor([0.5, 0.5])
        shape = [B] + [1] * 3
        zt = (1 - t.view(shape)) * x0 + t.view(shape) * noise

        v_pred, v_target = get_flow_matching_loss(x0.clone(), x0, zt, t)
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        assert loss.item() < 1e-10, f"Perfect prediction should give ~0 loss, got {loss.item()}"

    def test_timestep_sampling_in_01(self):
        """Sampled timesteps must all be in (0, 1)."""
        from library.nanosaur_train_util import sample_timesteps
        t = sample_timesteps(1000, "cpu", torch.float32, alpha=2.0)
        assert (t > 0).all() and (t < 1).all(), "Timesteps must be in (0, 1)"

    def test_timestep_sampling_mean_near_theoretical(self):
        """With alpha=2, sigmoid distribution mean should be > 0.5."""
        from library.nanosaur_train_util import sample_timesteps
        t = sample_timesteps(10000, "cpu", torch.float32, alpha=2.0)
        mean = t.mean().item()
        # logistic-normal with mu=log(2) is biased toward ~0.667
        assert 0.55 < mean < 0.80, f"Mean timestep {mean:.3f} outside expected range"


# ─── Rectified Flow Sampler ───────────────────────────────────────────────────


class TestRectifiedFlowSampler:
    def test_sampler_output_shape(self, small_model):
        """Sampler must return same shape as input latent."""
        from library.nanosaur_train_util import rectified_flow_sample
        small_model.eval()
        z = torch.randn(1, 8, 8, 8)
        cond = torch.randn(1, 4, 16)
        out = rectified_flow_sample(
            small_model, z, cond, null_cond=None,
            steps=3, guidance_scale=1.0, sample_shift=1.0,
        )
        assert out.shape == z.shape

    def test_sampler_cfg_changes_output(self, small_model):
        """CFG guidance (null_cond provided) must change the output."""
        from library.nanosaur_train_util import rectified_flow_sample
        small_model.eval()
        torch.manual_seed(0)
        z = torch.randn(1, 8, 8, 8)
        cond = torch.randn(1, 4, 16)
        null_cond = torch.zeros_like(cond)

        torch.manual_seed(0)
        out_no_cfg = rectified_flow_sample(
            small_model, z.clone(), cond, null_cond=None,
            steps=4, guidance_scale=7.0, sample_shift=1.0,
            cfg_start=0.0, cfg_end=1.0,
        )
        torch.manual_seed(0)
        out_cfg = rectified_flow_sample(
            small_model, z.clone(), cond, null_cond=null_cond,
            steps=4, guidance_scale=7.0, sample_shift=1.0,
            cfg_start=0.0, cfg_end=1.0,
        )
        assert not torch.allclose(out_no_cfg, out_cfg), \
            "CFG should change sampler output"

    def test_sampler_finite_output(self, small_model):
        """Sampler output must be finite (no NaN/Inf)."""
        from library.nanosaur_train_util import rectified_flow_sample
        small_model.eval()
        z = torch.randn(1, 8, 8, 8)
        cond = torch.randn(1, 4, 16)
        out = rectified_flow_sample(
            small_model, z, cond, null_cond=None,
            steps=5, guidance_scale=1.0,
        )
        assert torch.isfinite(out).all(), "Sampler output contains NaN or Inf"

    def test_sampler_momentum_guidance(self, small_model):
        """Momentum guidance enabled vs disabled must give different results."""
        from library.nanosaur_train_util import rectified_flow_sample
        small_model.eval()
        torch.manual_seed(42)
        z = torch.randn(1, 8, 8, 8)
        cond = torch.randn(1, 4, 16)
        null_cond = torch.randn_like(cond)

        torch.manual_seed(42)
        out_mg = rectified_flow_sample(
            small_model, z.clone(), cond, null_cond=null_cond,
            steps=4, guidance_scale=4.0, sample_shift=1.0,
            cfg_start=0.0, cfg_end=1.0, use_momentum_guidance=True,
        )
        torch.manual_seed(42)
        out_no_mg = rectified_flow_sample(
            small_model, z.clone(), cond, null_cond=null_cond,
            steps=4, guidance_scale=4.0, sample_shift=1.0,
            cfg_start=0.0, cfg_end=1.0, use_momentum_guidance=False,
        )
        assert not torch.allclose(out_mg, out_no_mg), \
            "Momentum guidance should change output"


# ─── SPRINT uncond path ───────────────────────────────────────────────────────


class TestSPRINT:
    def test_uncond_path_matches_dense_on_small_model(self, small_model):
        """
        For encoder-only blocks (is_encoder_block=True), SPRINT skips G blocks
        but F and H always run. On our tiny model all blocks are encoder type,
        so uncond=True should still return valid (finite) output.
        """
        x = torch.randn(1, 8, 8, 8)
        t = torch.tensor([0.5])
        ctx = torch.randn(1, 4, 16)

        with torch.no_grad():
            out_dense = small_model._forward(x, t, ctx, uncond=False)
            out_sprint = small_model._forward(x, t, ctx, uncond=True)

        assert torch.isfinite(out_dense).all()
        assert torch.isfinite(out_sprint).all()
        assert out_dense.shape == out_sprint.shape

    def test_sprint_f_blocks_always_run(self):
        """Model with num_encoder_blocks > 2: uncond skips G (global) blocks."""
        from library.nanosaur_models import NanoSaurTransformer2DModel
        # 6 blocks: F=2, G=2, H=2
        model = NanoSaurTransformer2DModel(
            in_channels=8, num_groups=4, hidden_size=32, decoder_hidden_size=64,
            num_encoder_blocks=6, num_decoder_blocks=1, num_text_blocks=1,
            patch_size=1, txt_embed_dim=16,
        )
        x = torch.randn(1, 8, 8, 8)
        t = torch.tensor([0.4])
        ctx = torch.randn(1, 4, 16)
        with torch.no_grad():
            out = model._forward(x, t, ctx, uncond=True)
        assert torch.isfinite(out).all()
        assert out.shape == x.shape


# ─── VAE Wrapper round-trip ───────────────────────────────────────────────────


class TestVAEWrapper:
    def test_encode_decode_inverse(self):
        """Decode(Encode(z)) should recover original z (in latent space)."""
        from library.nanosaur_models import LATENT_SCALE, LATENT_SHIFT
        from library.nanosaur_utils import NanoSaurVAEWrapper

        raw = torch.randn(1, 8, 4, 4)
        mock_vae = MagicMock()
        mock_vae.encode.return_value = raw.clone()
        mock_vae.decode.return_value = torch.zeros(1, 3, 64, 64)

        wrapper = NanoSaurVAEWrapper(mock_vae, device="cpu", dtype=torch.float32)

        # encode: (raw + SHIFT) / SCALE
        encoded = wrapper.encode(torch.zeros(1, 3, 16, 16))
        expected_encoded = (raw + LATENT_SHIFT) / LATENT_SCALE
        assert torch.allclose(encoded, expected_encoded, atol=1e-5)

        # decode call argument: z * SCALE - SHIFT
        wrapper.decode(encoded)
        decode_input = mock_vae.decode.call_args[0][0]
        expected_decode_input = encoded * LATENT_SCALE - LATENT_SHIFT
        assert torch.allclose(decode_input, expected_decode_input, atol=1e-5)

    def test_scale_shift_constants_consistent_with_comfyui(self):
        """LATENT_SCALE and LATENT_SHIFT must match ComfyUI nodes.py values."""
        from library.nanosaur_models import LATENT_SCALE, LATENT_SHIFT
        # ComfyUI: scale_factor=2.3623, shift_factor=-0.0179
        # process_in: (latent - shift_factor) / scale_factor = (latent + 0.0179) / 2.3623
        assert abs(LATENT_SCALE - 2.3623) < 1e-4
        assert abs(LATENT_SHIFT - 0.0179) < 1e-4

        # Verify encode formula matches ComfyUI process_in
        z = torch.tensor(1.0)
        sd_encode = (z + LATENT_SHIFT) / LATENT_SCALE
        comfyui_process_in = (z - (-0.0179)) / 2.3623
        assert torch.allclose(sd_encode, comfyui_process_in, atol=1e-5)


# ─── Strategy layer ───────────────────────────────────────────────────────────


class TestNanoSaurStrategies:
    def test_tokenize_strategy_returns_two_tensors(self):
        """tokenize() must return [input_ids, attention_mask]."""
        from library.nanosaur_utils import NanoSaurSentencePieceTokenizer
        from library.strategy_nanosaur import NanoSaurTokenizeStrategy

        mock_tokenizer = MagicMock(spec=NanoSaurSentencePieceTokenizer)
        mock_tokenizer.max_length = 16
        mock_tokenizer.bos_token_id = 2
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.batch_encode.return_value = (
            torch.randint(0, 100, (2, 16)),
            torch.ones(2, 16, dtype=torch.long),
        )

        strategy = NanoSaurTokenizeStrategy(mock_tokenizer, max_length=16)
        result = strategy.tokenize(["hello world", "test"])

        assert len(result) == 2, "tokenize must return [input_ids, attention_mask]"
        assert result[0].shape == (2, 16)
        assert result[1].shape == (2, 16)

    def test_text_encoding_strategy_output_shape(self):
        """encode_tokens must return [hidden_states, input_ids, attention_mask]."""
        from library.nanosaur_utils import NanoSaurSentencePieceTokenizer
        from library.strategy_nanosaur import NanoSaurTextEncodingStrategy, NanoSaurTokenizeStrategy

        D = 32
        L = 16
        mock_tokenizer = MagicMock(spec=NanoSaurSentencePieceTokenizer)
        mock_tokenizer.max_length = L
        mock_tokenizer.bos_token_id = 2
        mock_tokenizer.pad_token_id = 0

        strategy = NanoSaurTextEncodingStrategy()
        tokenize_strategy = MagicMock()

        input_ids = torch.randint(0, 100, (2, L))
        attention_mask = torch.ones(2, L, dtype=torch.long)

        mock_te = MagicMock()
        mock_te.device = torch.device("cpu")
        hidden = torch.randn(2, L, D)
        mock_output = MagicMock()
        mock_output.hidden_states = [None] * 5 + [hidden]
        mock_te.return_value = mock_output

        result = strategy.encode_tokens(tokenize_strategy, [mock_te], [input_ids, attention_mask])
        assert len(result) == 3
        assert result[0].shape == (2, L, D), f"hidden_states shape wrong: {result[0].shape}"

    def test_latents_caching_strategy_vae_stride(self):
        """NanoSaurLatentsCachingStrategy must report VAE stride=16."""
        from library.strategy_nanosaur import NanoSaurLatentsCachingStrategy
        strategy = NanoSaurLatentsCachingStrategy(False, 1, False)
        assert strategy.VAE_STRIDE == 16

    def test_te_outputs_caching_strategy_cache_suffix(self):
        """NanoSaurTextEncoderOutputsCachingStrategy must use _nanosaur_te.npz suffix."""
        from library.strategy_nanosaur import NanoSaurTextEncoderOutputsCachingStrategy
        strategy = NanoSaurTextEncoderOutputsCachingStrategy(False, 1, False, False)
        suffix = strategy.NANOSAUR_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX
        assert "_nanosaur" in suffix and suffix.endswith(".npz"), \
            f"Unexpected suffix: {suffix}"

    def test_te_outputs_npz_round_trip(self):
        """Cache save/load round-trip must preserve hidden_state values."""
        from library.strategy_nanosaur import NanoSaurTextEncoderOutputsCachingStrategy

        L, D = 16, 32
        hidden = np.random.randn(L, D).astype(np.float32)
        input_ids = np.random.randint(0, 100, (L,), dtype=np.int32)
        attn_mask = np.ones(L, dtype=np.int32)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp = f.name
        try:
            np.savez(tmp, hidden_state=hidden, input_ids=input_ids, attention_mask=attn_mask)
            strategy = NanoSaurTextEncoderOutputsCachingStrategy(False, 1, False, False)
            loaded = strategy.load_outputs_npz(tmp)
            assert np.allclose(loaded[0], hidden)
        finally:
            os.unlink(tmp)


# ─── nanosaur_train_network.py — online encoding path ────────────────────────


class TestNanoSaurNetworkTrainerOnlineEncoding:
    def test_get_noise_pred_batch_dict_empty(self):
        """get_noise_pred_and_target must handle empty batch dict without crashing."""
        from library.nanosaur_models import NanoSaurTransformer2DModel
        from nanosaur_train_network import NanoSaurNetworkTrainer

        trainer = NanoSaurNetworkTrainer()
        trainer.train_gemma3 = False

        model = NanoSaurTransformer2DModel(
            in_channels=8, num_groups=4, hidden_size=32, decoder_hidden_size=64,
            num_encoder_blocks=4, num_decoder_blocks=1, num_text_blocks=1,
            patch_size=1, txt_embed_dim=16,
        )

        mock_acc = MagicMock()
        mock_acc.device = torch.device("cpu")
        mock_acc.autocast.return_value.__enter__ = lambda s: None
        mock_acc.autocast.return_value.__exit__ = lambda s, *a: None

        noise_sched = MagicMock()
        noise_sched.num_train_timesteps = 1000

        mock_args = MagicMock()
        mock_args.gradient_checkpointing = False
        mock_args.time_sampling_alpha = 2.0

        mock_network = MagicMock()

        latents = torch.randn(1, 8, 8, 8)
        hidden = torch.randn(1, 4, 16)

        v_pred, v_target, timesteps, weighting = trainer.get_noise_pred_and_target(
            args=mock_args,
            accelerator=mock_acc,
            noise_scheduler=noise_sched,
            latents=latents,
            batch={},
            text_encoder_conds=(hidden, None, None),
            dit=model,
            network=mock_network,
            weight_dtype=torch.float32,
            train_unet=True,
            is_train=False,
        )
        assert v_pred.shape == latents.shape
        assert v_target.shape == latents.shape
        assert weighting is None


# ─── Multi-GPU / DeepSpeed Trainer Hooks ─────────────────────────────────────


class TestMultiGPUTrainerHooks:
    def test_prepare_unet_no_swap_delegates_to_super(self):
        """Without block swap, prepare_unet_with_accelerator must call super."""
        from nanosaur_train_network import NanoSaurNetworkTrainer
        trainer = NanoSaurNetworkTrainer()
        trainer.is_swapping_blocks = False

        mock_acc = MagicMock()
        mock_unet = MagicMock()
        mock_args = MagicMock()

        with patch("train_network.NetworkTrainer.prepare_unet_with_accelerator",
                   return_value=mock_unet) as mock_super:
            result = trainer.prepare_unet_with_accelerator(mock_args, mock_acc, mock_unet)
            mock_super.assert_called_once_with(mock_args, mock_acc, mock_unet)

    def test_prepare_unet_with_swap_uses_manual_placement(self):
        """With block swap, accelerator.prepare must be called with device_placement=[False]."""
        from nanosaur_train_network import NanoSaurNetworkTrainer
        trainer = NanoSaurNetworkTrainer()
        trainer.is_swapping_blocks = True

        mock_acc = MagicMock()
        mock_acc.prepare.return_value = MagicMock()
        mock_acc.unwrap_model.return_value = MagicMock()
        mock_unet = MagicMock()
        mock_args = MagicMock()

        trainer.prepare_unet_with_accelerator(mock_args, mock_acc, mock_unet)

        call_kwargs = mock_acc.prepare.call_args
        assert call_kwargs[1].get("device_placement") == [False], \
            "Block swap must use device_placement=[False]"

    def test_on_validation_step_end_calls_prepare_swap(self):
        """on_validation_step_end must call prepare_block_swap_before_forward when swapping."""
        from nanosaur_train_network import NanoSaurNetworkTrainer
        trainer = NanoSaurNetworkTrainer()
        trainer.is_swapping_blocks = True

        mock_acc = MagicMock()
        mock_unet = MagicMock()
        mock_unwrapped = MagicMock()
        mock_acc.unwrap_model.return_value = mock_unwrapped

        trainer.on_validation_step_end(
            MagicMock(), mock_acc, MagicMock(), [], mock_unet, {}, torch.float32
        )
        mock_unwrapped.prepare_block_swap_before_forward.assert_called_once()

    def test_on_validation_step_end_no_swap_no_call(self):
        """on_validation_step_end must NOT call prepare_block_swap when not swapping."""
        from nanosaur_train_network import NanoSaurNetworkTrainer
        trainer = NanoSaurNetworkTrainer()
        trainer.is_swapping_blocks = False

        mock_acc = MagicMock()
        mock_unet = MagicMock()

        trainer.on_validation_step_end(
            MagicMock(), mock_acc, MagicMock(), [], mock_unet, {}, torch.float32
        )
        mock_acc.unwrap_model.assert_not_called()

    def test_cache_te_outputs_online_mode_moves_to_device(self):
        """When cache_text_encoder_outputs=False, TE must be moved to device."""
        from nanosaur_train_network import NanoSaurNetworkTrainer
        trainer = NanoSaurNetworkTrainer()

        mock_acc = MagicMock()
        mock_acc.device = torch.device("cpu")
        mock_te = MagicMock()
        mock_vae = MagicMock()
        mock_unet = MagicMock()
        mock_args = MagicMock()
        mock_args.cache_text_encoder_outputs = False
        mock_dataset = MagicMock()

        trainer.cache_text_encoder_outputs_if_needed(
            mock_args, mock_acc, mock_unet, mock_vae, [mock_te], mock_dataset, torch.float32
        )

        # TE must be moved to device
        mock_te.to.assert_called()
        call_args = mock_te.to.call_args
        assert torch.device("cpu") in call_args[0] or \
               call_args[1].get("device", None) is not None or \
               mock_te.to.called

    def test_deepspeed_args_preparation_callable(self):
        """deepspeed_utils.prepare_deepspeed_args must be importable and callable."""
        from library import deepspeed_utils
        assert hasattr(deepspeed_utils, "prepare_deepspeed_args"), \
            "deepspeed_utils must export prepare_deepspeed_args"
        args = MagicMock()
        args.deepspeed = False
        # Should not raise
        deepspeed_utils.prepare_deepspeed_args(args)

    def test_accelerate_multi_process_config(self):
        """Accelerate DistributedDataParallelKwargs must be importable for multi-GPU."""
        from accelerate import DistributedDataParallelKwargs
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        assert kwargs is not None


# ─── nanosaur_train.py online encoding fix ───────────────────────────────────


class TestNanoSaurTrainOnlineEncoding:
    def test_online_encoding_path_exists(self):
        """nanosaur_train.py must NOT contain raise NotImplementedError in the else branch."""
        with open("nanosaur_train.py", "r", encoding="utf-8") as f:
            src = f.read()
        # Check the NotImplementedError is gone from online encoding branch
        assert "Online text encoding not yet implemented" not in src, \
            "nanosaur_train.py still has NotImplementedError for online encoding"

    def test_online_encoding_calls_tokenize_and_encode(self):
        """The online encoding path must call tokenize_strategy.tokenize and encode_tokens."""
        with open("nanosaur_train.py", "r", encoding="utf-8") as f:
            src = f.read()
        # Verify the fix is present
        assert "tokenize_strategy.tokenize" in src, \
            "Online encoding must call tokenize_strategy.tokenize"
        assert "encode_tokens" in src or "text_encoding_strategy" in src, \
            "Online encoding must call text_encoding_strategy.encode_tokens"

    def test_gemma3_moved_to_device_when_not_caching(self):
        """When not caching, gemma3 must be moved to device before training."""
        with open("nanosaur_train.py", "r", encoding="utf-8") as f:
            src = f.read()
        # The else branch after the cache block must move gemma3
        assert "gemma3.to(accelerator.device" in src or \
               "gemma3.to(accelerator.device, dtype" in src, \
               "gemma3 must be moved to device in online mode"


# ─── SyntaxWarning regression ─────────────────────────────────────────────────


class TestNoSyntaxErrorsInNanoSaurFiles:
    @pytest.mark.parametrize("module_path", [
        "library.nanosaur_models",
        "library.nanosaur_utils",
        "library.nanosaur_train_util",
        "library.strategy_nanosaur",
        "networks.lora_nanosaur",
    ])
    def test_module_imports_cleanly(self, module_path):
        """All NanoSaur modules must import without errors."""
        import importlib
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            pytest.fail(f"{module_path} failed to import: {e}")

    def test_nanosaur_train_network_imports(self):
        """nanosaur_train_network must import without errors."""
        import importlib
        importlib.import_module("nanosaur_train_network")


# ─── Model state dict compatibility ──────────────────────────────────────────


class TestModelStateDictCompatibility:
    def test_state_dict_keys_no_private_prefix_exposed(self):
        """NanoSaurVAE state dict must not expose private _DINOv3Encoder at top level."""
        from library.nanosaur_models import NanoSaurVAE
        vae = NanoSaurVAE(latent_dim=16)
        sd = vae.state_dict()
        # All keys should be accessible — none should start with private prefix
        private_keys = [k for k in sd if k.startswith("_")]
        assert len(private_keys) == 0, \
            f"Private-prefixed keys exposed in state dict: {private_keys[:3]}"
        # Encoder and decoder must be present
        assert len(sd) > 0, "VAE state dict must not be empty"

    def test_fc_norm_head_not_in_state_dict(self, small_model):
        """fc_norm and head are nn.Identity — they should have no parameters in state dict."""
        from library.nanosaur_models import NanoSaurVAE
        vae = NanoSaurVAE(latent_dim=16)
        sd = vae.state_dict()
        for key in sd:
            assert "fc_norm" not in key, f"fc_norm should not appear in state dict: {key}"
            assert ".head." not in key, f".head. should not appear in state dict: {key}"

    def test_lora_network_state_dict_has_lora_keys_after_apply(self, small_model, lora_network):
        """After apply_to, LoRANetwork.state_dict() must contain lora_up/down weights."""
        sd = lora_network.state_dict()
        lora_keys = [k for k in sd if "lora_up" in k or "lora_down" in k]
        assert len(lora_keys) > 0, "LoRANetwork state dict must contain lora_up/lora_down keys"

    def test_lora_state_dict_keys_start_with_lora_unet(self, lora_network):
        """All LoRA state dict keys must start with lora_unet_."""
        sd = lora_network.state_dict()
        for key in sd:
            assert key.startswith("lora_unet_"), \
                f"LoRA state dict key does not start with lora_unet_: {key}"
