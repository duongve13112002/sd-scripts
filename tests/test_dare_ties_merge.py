"""Tests for DARE-TIES LoRA merge functionality."""

import math
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file, load_file


# We'll import from the module once it exists
# For now, define the expected interface
def make_lora_state_dict(module_names, rank=4, in_dim=64, out_dim=64, alpha=None, seed=None):
    """Create a synthetic LoRA state dict for testing."""
    if seed is not None:
        torch.manual_seed(seed)
    if alpha is None:
        alpha = float(rank)
    sd = {}
    for name in module_names:
        sd[f"{name}.lora_down.weight"] = torch.randn(rank, in_dim)
        sd[f"{name}.lora_up.weight"] = torch.randn(out_dim, rank)
        sd[f"{name}.alpha"] = torch.tensor(alpha)
    return sd


def save_lora_to_temp(sd, metadata=None):
    """Save a LoRA state dict to a temporary safetensors file."""
    f = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    f.close()
    if metadata is None:
        metadata = {}
    # safetensors metadata must be strings
    str_metadata = {k: str(v) for k, v in metadata.items()}
    save_file(sd, f.name, metadata=str_metadata)
    return f.name


def expand_lora_delta(down, up, alpha):
    """Compute full delta from LoRA matrices: up @ down * (alpha / rank)."""
    rank = down.shape[0]
    scale = alpha / rank
    return (up @ down) * scale


class TestExpandLoraDelta:
    """Test LoRA expansion to full delta."""

    def test_basic_expansion(self):
        rank, in_dim, out_dim = 4, 16, 16
        down = torch.randn(rank, in_dim)
        up = torch.randn(out_dim, rank)
        alpha = 4.0

        delta = expand_lora_delta(down, up, alpha)
        assert delta.shape == (out_dim, in_dim)

    def test_scale_factor(self):
        rank, in_dim, out_dim = 4, 16, 16
        down = torch.eye(rank, in_dim)
        up = torch.eye(out_dim, rank)
        alpha = 2.0

        delta = expand_lora_delta(down, up, alpha)
        expected_scale = alpha / rank  # 0.5
        # The diagonal elements should be 0.5
        for i in range(rank):
            assert abs(delta[i, i].item() - expected_scale) < 1e-6

    def test_different_rank_and_alpha(self):
        rank, in_dim, out_dim = 8, 32, 64
        down = torch.randn(rank, in_dim)
        up = torch.randn(out_dim, rank)
        alpha = 16.0

        delta = expand_lora_delta(down, up, alpha)
        assert delta.shape == (out_dim, in_dim)
        # Scale should be 16/8 = 2.0
        manual = (up @ down) * 2.0
        assert torch.allclose(delta, manual)


class TestDarePruning:
    """Test DARE random pruning and rescaling."""

    def test_density_1_preserves_all(self):
        """density=1.0 should keep all weights."""
        from networks.dare_ties_merge_lora import dare_prune_rescale

        tensor = torch.randn(64, 64)
        result = dare_prune_rescale(tensor, density=1.0, seed=42)
        assert torch.allclose(tensor, result)

    def test_density_0_zeros_all(self):
        """density=0.0 should zero all weights."""
        from networks.dare_ties_merge_lora import dare_prune_rescale

        tensor = torch.randn(64, 64)
        result = dare_prune_rescale(tensor, density=0.0, seed=42)
        assert torch.all(result == 0)

    def test_sparsity_matches_density(self):
        """Fraction of non-zero elements should approximately match density."""
        from networks.dare_ties_merge_lora import dare_prune_rescale

        torch.manual_seed(0)
        tensor = torch.randn(1000, 1000)
        density = 0.5
        result = dare_prune_rescale(tensor, density=density, seed=42)

        nonzero_frac = (result != 0).float().mean().item()
        # Allow 5% tolerance for randomness
        assert abs(nonzero_frac - density) < 0.05, f"Expected ~{density}, got {nonzero_frac}"

    def test_rescaling_preserves_l1_norm(self):
        """After DARE pruning+rescaling, L1 norm should be approximately preserved."""
        from networks.dare_ties_merge_lora import dare_prune_rescale

        torch.manual_seed(0)
        tensor = torch.randn(500, 500)
        original_l1 = tensor.abs().sum().item()

        result = dare_prune_rescale(tensor, density=0.5, seed=42)
        result_l1 = result.abs().sum().item()

        # L1 norm should be approximately preserved (within 10% for large tensors)
        ratio = result_l1 / original_l1
        assert 0.8 < ratio < 1.2, f"L1 norm ratio {ratio} is too far from 1.0"

    def test_different_seeds_produce_different_masks(self):
        """Different seeds should produce different pruning masks."""
        from networks.dare_ties_merge_lora import dare_prune_rescale

        tensor = torch.randn(100, 100)
        result1 = dare_prune_rescale(tensor, density=0.5, seed=42)
        result2 = dare_prune_rescale(tensor, density=0.5, seed=123)
        assert not torch.allclose(result1, result2)


class TestTiesSignConsensus:
    """Test TIES sign consensus voting."""

    def test_unanimous_positive(self):
        """When all models agree on positive sign, all should pass."""
        from networks.dare_ties_merge_lora import ties_sign_consensus

        deltas = [torch.ones(4, 4), torch.ones(4, 4) * 2, torch.ones(4, 4) * 0.5]
        weights = [1.0, 1.0, 1.0]
        result = ties_sign_consensus(deltas, weights)

        # All values are positive, all should pass consensus
        for r in result:
            assert torch.all(r >= 0)

    def test_unanimous_negative(self):
        """When all models agree on negative sign, all should pass."""
        from networks.dare_ties_merge_lora import ties_sign_consensus

        deltas = [-torch.ones(4, 4), -torch.ones(4, 4) * 2, -torch.ones(4, 4) * 0.5]
        weights = [1.0, 1.0, 1.0]
        result = ties_sign_consensus(deltas, weights)

        for r in result:
            assert torch.all(r <= 0)

    def test_disagreement_masks_minority(self):
        """When 2/3 models are positive and 1 is negative, negative should be masked."""
        from networks.dare_ties_merge_lora import ties_sign_consensus

        deltas = [
            torch.ones(4, 4) * 3.0,   # positive, large
            torch.ones(4, 4) * 2.0,   # positive
            -torch.ones(4, 4) * 1.0,  # negative, minority
        ]
        weights = [1.0, 1.0, 1.0]
        result = ties_sign_consensus(deltas, weights)

        # Majority sign is positive (sum of weighted deltas > 0)
        # So the third model's values should be masked to 0
        assert torch.all(result[0] > 0)
        assert torch.all(result[1] > 0)
        assert torch.all(result[2] == 0)

    def test_weights_affect_consensus(self):
        """Weights should influence which sign wins."""
        from networks.dare_ties_merge_lora import ties_sign_consensus

        deltas = [
            torch.ones(4, 4) * 1.0,   # positive, small
            -torch.ones(4, 4) * 1.0,  # negative, but heavily weighted
        ]
        # With equal weights, positive wins (or tie goes to positive)
        result_equal = ties_sign_consensus(deltas, [1.0, 1.0])
        # With heavy weight on negative, negative should win
        result_heavy_neg = ties_sign_consensus(deltas, [1.0, 10.0])

        # In the heavily weighted case, majority sign should be negative
        assert torch.all(result_heavy_neg[0] == 0)  # positive masked out
        assert torch.all(result_heavy_neg[1] < 0)    # negative survives

    def test_zero_values_stay_zero(self):
        """Zero values should remain zero regardless of consensus."""
        from networks.dare_ties_merge_lora import ties_sign_consensus

        deltas = [
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        ]
        weights = [1.0, 1.0]
        result = ties_sign_consensus(deltas, weights)

        for r in result:
            assert r[0, 0] == 0
            assert r[1, 1] == 0


class TestSVDRecompose:
    """Test SVD decomposition back to LoRA format."""

    def test_svd_approximation_quality(self):
        """SVD recomposition should approximate the original well for low-rank matrices."""
        from networks.dare_ties_merge_lora import svd_decompose_delta

        # Create a low-rank matrix (rank 4)
        rank = 4
        A = torch.randn(64, rank)
        B = torch.randn(rank, 64)
        original = A @ B

        down, up = svd_decompose_delta(original, rank=rank)
        reconstructed = up @ down

        # Should be very close for same-rank decomposition
        rel_error = (original - reconstructed).norm() / original.norm()
        assert rel_error < 0.01, f"Relative error {rel_error} too high for same-rank SVD"

    def test_svd_lower_rank_compression(self):
        """SVD with lower rank should still produce reasonable approximation."""
        from networks.dare_ties_merge_lora import svd_decompose_delta

        original = torch.randn(64, 64)
        rank = 8
        down, up = svd_decompose_delta(original, rank=rank)

        assert down.shape == (rank, 64)
        assert up.shape == (64, rank)

    def test_svd_output_shapes(self):
        """Output shapes should match expected LoRA format."""
        from networks.dare_ties_merge_lora import svd_decompose_delta

        out_dim, in_dim, rank = 128, 64, 8
        delta = torch.randn(out_dim, in_dim)
        down, up = svd_decompose_delta(delta, rank=rank)

        assert down.shape == (rank, in_dim), f"down shape {down.shape} != ({rank}, {in_dim})"
        assert up.shape == (out_dim, rank), f"up shape {up.shape} != ({out_dim}, {rank})"


class TestFullMergePipeline:
    """Test the complete DARE-TIES merge pipeline."""

    def test_single_model_passthrough(self):
        """Merging a single model should approximately reproduce it."""
        from networks.dare_ties_merge_lora import dare_ties_merge

        modules = ["lora_unet_blocks_0_attn_qkv"]
        sd1 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=42)

        f1 = save_lora_to_temp(sd1)
        try:
            merged_sd = dare_ties_merge(
                model_paths=[f1],
                weights=[1.0],
                density=1.0,  # keep everything
                output_rank=4,
                merge_dtype=torch.float32,
            )

            # Check that keys exist
            for name in modules:
                assert f"{name}.lora_down.weight" in merged_sd
                assert f"{name}.lora_up.weight" in merged_sd
                assert f"{name}.alpha" in merged_sd

            # Reconstruct deltas and compare
            orig_delta = expand_lora_delta(
                sd1[f"{modules[0]}.lora_down.weight"],
                sd1[f"{modules[0]}.lora_up.weight"],
                sd1[f"{modules[0]}.alpha"].item(),
            )
            merged_delta = expand_lora_delta(
                merged_sd[f"{modules[0]}.lora_down.weight"],
                merged_sd[f"{modules[0]}.lora_up.weight"],
                merged_sd[f"{modules[0]}.alpha"].item(),
            )

            rel_error = (orig_delta - merged_delta).norm() / orig_delta.norm()
            assert rel_error < 0.05, f"Single model passthrough error {rel_error} too high"
        finally:
            os.unlink(f1)

    def test_two_model_merge(self):
        """Merging two models should produce valid output."""
        from networks.dare_ties_merge_lora import dare_ties_merge

        modules = ["lora_unet_blocks_0_attn_qkv"]
        sd1 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=42)
        sd2 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=123)

        f1 = save_lora_to_temp(sd1)
        f2 = save_lora_to_temp(sd2)
        try:
            merged_sd = dare_ties_merge(
                model_paths=[f1, f2],
                weights=[1.0, 1.0],
                density=0.7,
                output_rank=4,
                merge_dtype=torch.float32,
                seed=42,
            )

            for name in modules:
                assert f"{name}.lora_down.weight" in merged_sd
                assert f"{name}.lora_up.weight" in merged_sd
                assert f"{name}.alpha" in merged_sd
        finally:
            os.unlink(f1)
            os.unlink(f2)

    def test_different_ranks(self):
        """Should handle LoRAs with different ranks."""
        from networks.dare_ties_merge_lora import dare_ties_merge

        modules = ["lora_unet_blocks_0_attn_qkv"]
        sd1 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=42)
        sd2 = make_lora_state_dict(modules, rank=8, in_dim=32, out_dim=32, seed=123)

        f1 = save_lora_to_temp(sd1)
        f2 = save_lora_to_temp(sd2)
        try:
            merged_sd = dare_ties_merge(
                model_paths=[f1, f2],
                weights=[1.0, 1.0],
                density=0.7,
                output_rank=4,
                merge_dtype=torch.float32,
                seed=42,
            )

            # Output rank should be the requested rank
            down = merged_sd[f"{modules[0]}.lora_down.weight"]
            assert down.shape[0] == 4
        finally:
            os.unlink(f1)
            os.unlink(f2)

    def test_missing_module_in_some_models(self):
        """Should handle modules present in some models but not others."""
        from networks.dare_ties_merge_lora import dare_ties_merge

        sd1 = make_lora_state_dict(["lora_unet_blocks_0_attn_qkv", "lora_unet_blocks_1_attn_qkv"],
                                    rank=4, in_dim=32, out_dim=32, seed=42)
        sd2 = make_lora_state_dict(["lora_unet_blocks_0_attn_qkv"],
                                    rank=4, in_dim=32, out_dim=32, seed=123)

        f1 = save_lora_to_temp(sd1)
        f2 = save_lora_to_temp(sd2)
        try:
            merged_sd = dare_ties_merge(
                model_paths=[f1, f2],
                weights=[1.0, 1.0],
                density=0.7,
                output_rank=4,
                merge_dtype=torch.float32,
                seed=42,
            )

            # Both modules should exist in output
            assert "lora_unet_blocks_0_attn_qkv.lora_down.weight" in merged_sd
            assert "lora_unet_blocks_1_attn_qkv.lora_down.weight" in merged_sd
        finally:
            os.unlink(f1)
            os.unlink(f2)

    def test_three_model_merge(self):
        """Merging three models should work correctly."""
        from networks.dare_ties_merge_lora import dare_ties_merge

        modules = ["lora_unet_blocks_0_attn_qkv"]
        sd1 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=1)
        sd2 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=2)
        sd3 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=3)

        f1 = save_lora_to_temp(sd1)
        f2 = save_lora_to_temp(sd2)
        f3 = save_lora_to_temp(sd3)
        try:
            merged_sd = dare_ties_merge(
                model_paths=[f1, f2, f3],
                weights=[1.0, 0.8, 0.6],
                density=0.5,
                output_rank=4,
                merge_dtype=torch.float32,
                seed=42,
            )

            for name in modules:
                assert f"{name}.lora_down.weight" in merged_sd
                assert f"{name}.lora_up.weight" in merged_sd
        finally:
            os.unlink(f1)
            os.unlink(f2)
            os.unlink(f3)

    def test_output_file_save_and_load(self):
        """Test saving and loading the merged result."""
        from networks.dare_ties_merge_lora import dare_ties_merge

        modules = ["lora_unet_blocks_0_attn_qkv"]
        sd1 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=42)
        sd2 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=123)

        f1 = save_lora_to_temp(sd1)
        f2 = save_lora_to_temp(sd2)
        out_f = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        out_f.close()
        try:
            merged_sd = dare_ties_merge(
                model_paths=[f1, f2],
                weights=[1.0, 1.0],
                density=0.7,
                output_rank=4,
                merge_dtype=torch.float32,
                seed=42,
            )

            # Save
            save_file(merged_sd, out_f.name)

            # Load and verify
            loaded = load_file(out_f.name)
            for key in merged_sd:
                assert key in loaded
                assert torch.allclose(merged_sd[key], loaded[key])
        finally:
            os.unlink(f1)
            os.unlink(f2)
            os.unlink(out_f.name)

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical results."""
        from networks.dare_ties_merge_lora import dare_ties_merge

        modules = ["lora_unet_blocks_0_attn_qkv"]
        sd1 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=42)
        sd2 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=123)

        f1 = save_lora_to_temp(sd1)
        f2 = save_lora_to_temp(sd2)
        try:
            result1 = dare_ties_merge(
                model_paths=[f1, f2], weights=[1.0, 1.0],
                density=0.5, output_rank=4, merge_dtype=torch.float32, seed=42,
            )
            result2 = dare_ties_merge(
                model_paths=[f1, f2], weights=[1.0, 1.0],
                density=0.5, output_rank=4, merge_dtype=torch.float32, seed=42,
            )

            for key in result1:
                assert torch.allclose(result1[key], result2[key]), f"Mismatch at {key}"
        finally:
            os.unlink(f1)
            os.unlink(f2)


class TestDareTiesMethodVariant:
    """Test DARE-only and TIES-only variants."""

    def test_dare_linear_no_sign_consensus(self):
        """DARE-linear (no TIES sign consensus) should just prune+rescale+sum."""
        from networks.dare_ties_merge_lora import dare_ties_merge

        modules = ["lora_unet_blocks_0_attn_qkv"]
        sd1 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=42)
        sd2 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=123)

        f1 = save_lora_to_temp(sd1)
        f2 = save_lora_to_temp(sd2)
        try:
            merged_sd = dare_ties_merge(
                model_paths=[f1, f2],
                weights=[1.0, 1.0],
                density=0.7,
                output_rank=4,
                merge_dtype=torch.float32,
                seed=42,
                method="dare_linear",
            )

            for name in modules:
                assert f"{name}.lora_down.weight" in merged_sd
        finally:
            os.unlink(f1)
            os.unlink(f2)

    def test_ties_only(self):
        """TIES-only (magnitude pruning instead of random) should work."""
        from networks.dare_ties_merge_lora import dare_ties_merge

        modules = ["lora_unet_blocks_0_attn_qkv"]
        sd1 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=42)
        sd2 = make_lora_state_dict(modules, rank=4, in_dim=32, out_dim=32, seed=123)

        f1 = save_lora_to_temp(sd1)
        f2 = save_lora_to_temp(sd2)
        try:
            merged_sd = dare_ties_merge(
                model_paths=[f1, f2],
                weights=[1.0, 1.0],
                density=0.7,
                output_rank=4,
                merge_dtype=torch.float32,
                seed=42,
                method="ties",
            )

            for name in modules:
                assert f"{name}.lora_down.weight" in merged_sd
        finally:
            os.unlink(f1)
            os.unlink(f2)
