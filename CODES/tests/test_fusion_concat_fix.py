"""
Unit tests for fusion_concat bug fix in DMLPFFNAblation.

These tests verify that:
1. fusion_concat mode produces 192 channels (from concat fusion)
2. additive fusion mode produces 64 channels
3. Parameter counts match expected values
4. Forward pass executes correctly
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CODES.baselines.factory import DMLPFFNAblation, DMLPFFN_ABLATION_CONFIGS


class TestFusionConcatFix:
    """Test suite for verifying correct concat fusion behavior."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 128, 9, 9)

    def test_fusion_concat_parameter_count(self):
        """fusion_concat should have 448,292 parameters (192-channel classifier)."""
        config = DMLPFFN_ABLATION_CONFIGS['fusion_concat'].copy()
        config['in_channels'] = 128
        config['num_classes'] = 4

        model = DMLPFFNAblation(**config)
        total_params = sum(p.numel() for p in model.parameters())

        assert total_params == 448292, \
            f"fusion_concat should have 448,292 params but got {total_params:,}"

    def test_fusion_concat_classifier_dimensions(self):
        """fusion_concat classifier should accept 192 channels (64*3 from concat)."""
        config = DMLPFFN_ABLATION_CONFIGS['fusion_concat'].copy()
        config['in_channels'] = 128
        config['num_classes'] = 4

        model = DMLPFFNAblation(**config)

        assert model.fused_channels == 192, \
            f"fusion_concat fused_channels should be 192 but got {model.fused_channels}"
        assert model.classifier.in_features == 192, \
            f"classifier should have in_features=192 but got {model.classifier.in_features}"

    def test_additive_fusion_parameter_count(self):
        """additive fusion ('full' config) should have 447,780 parameters (64-channel classifier)."""
        config = DMLPFFN_ABLATION_CONFIGS['full'].copy()
        config['in_channels'] = 128
        config['num_classes'] = 4

        model = DMLPFFNAblation(**config)
        total_params = sum(p.numel() for p in model.parameters())

        assert total_params == 447780, \
            f"additive fusion should have 447,780 params but got {total_params:,}"

    def test_additive_fusion_classifier_dimensions(self):
        """additive fusion classifier should accept 64 channels."""
        config = DMLPFFN_ABLATION_CONFIGS['full'].copy()
        config['in_channels'] = 128
        config['num_classes'] = 4

        model = DMLPFFNAblation(**config)

        assert model.fused_channels == 64, \
            f"additive fusion fused_channels should be 64 but got {model.fused_channels}"
        assert model.classifier.in_features == 64, \
            f"classifier should have in_features=64 but got {model.classifier.in_features}"

    def test_fusion_concat_forward_pass(self, sample_input):
        """fusion_concat forward pass should work without shape errors."""
        config = DMLPFFN_ABLATION_CONFIGS['fusion_concat'].copy()
        config['in_channels'] = 128
        config['num_classes'] = 4

        model = DMLPFFNAblation(**config)
        model.eval()

        with torch.no_grad():
            output = model(sample_input)

        assert output.shape == (2, 4), \
            f"Expected output shape (2, 4), got {tuple(output.shape)}"

    def test_additive_fusion_forward_pass(self, sample_input):
        """additive fusion forward pass should work correctly."""
        config = DMLPFFN_ABLATION_CONFIGS['full'].copy()
        config['in_channels'] = 128
        config['num_classes'] = 4

        model = DMLPFFNAblation(**config)
        model.eval()

        with torch.no_grad():
            output = model(sample_input)

        assert output.shape == (2, 4), \
            f"Expected output shape (2, 4), got {tuple(output.shape)}"

    def test_concat_vs_additive_output_divergence(self, sample_input):
        """Concat and additive fusion should produce different outputs."""
        concat_config = DMLPFFN_ABLATION_CONFIGS['fusion_concat'].copy()
        concat_config['in_channels'] = 128
        concat_config['num_classes'] = 4

        add_config = DMLPFFN_ABLATION_CONFIGS['full'].copy()
        add_config['in_channels'] = 128
        add_config['num_classes'] = 4

        concat_model = DMLPFFNAblation(**concat_config)
        add_model = DMLPFFNAblation(**add_config)

        concat_model.eval()
        add_model.eval()

        with torch.no_grad():
            concat_out = concat_model(sample_input)
            add_out = add_model(sample_input)

        # Outputs should be different (different fusion modes)
        assert not torch.allclose(concat_out, add_out), \
            "Concat and additive fusion should produce different outputs"

    def test_branch_output_channels_consistency(self):
        """All branches should output channel_output_channels (64) before fusion."""
        config = DMLPFFN_ABLATION_CONFIGS['fusion_concat'].copy()
        config['in_channels'] = 128
        config['num_classes'] = 4

        model = DMLPFFNAblation(**config)
        model.eval()

        x = torch.randn(1, 128, 9, 9)

        # Get branch outputs
        low_out = model.low_branch(x)
        mid_out = model.mid_branch(x)
        high_out = model.high_branch(x)

        # Each branch should output 64 channels
        assert low_out.shape[1] == 64, f"low_branch should output 64 channels, got {low_out.shape[1]}"
        assert mid_out.shape[1] == 64, f"mid_branch should output 64 channels, got {mid_out.shape[1]}"
        assert high_out.shape[1] == 64, f"high_branch should output 64 channels, got {high_out.shape[1]}"

    def test_all_branches_present_in_concat(self, sample_input):
        """Concat fusion should include all 3 branches."""
        config = DMLPFFN_ABLATION_CONFIGS['fusion_concat'].copy()
        config['in_channels'] = 128
        config['num_classes'] = 4

        model = DMLPFFNAblation(**config)
        model.eval()

        with torch.no_grad():
            x = sample_input.clone()

            # Manually compute expected concat result
            low = model.low_branch(x)
            mid = model.mid_branch(x)
            high = model.high_branch(x)
            expected_concat = torch.cat([low, mid, high], dim=1)

            # Verify dimensions
            assert expected_concat.shape[1] == 192, \
                f"Expected concat channels 192, got {expected_concat.shape[1]}"

    def test_fusion_concat_with_different_dilation_rates(self):
        """fusion_concat should work with custom dilation rates."""
        config = DMLPFFN_ABLATION_CONFIGS['fusion_concat'].copy()
        config['in_channels'] = 128
        config['num_classes'] = 4
        config['dilation_rates'] = [1, 3, 7]

        model = DMLPFFNAblation(**config)
        total_params = sum(p.numel() for p in model.parameters())

        assert model.fused_channels == 192, \
            f"fusion_concat with custom dilation should still have fused_channels=192"
        assert total_params == 448292, \
            f"fusion_concat with custom dilation should have 448,292 params"

        # Test forward pass
        x = torch.randn(2, 128, 9, 9)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4)


class TestHistoricalBugReplication:
    """Tests that verify the historical bug pattern is now fixed."""

    def test_concat_not_overwritten_by_add(self):
        """Verify that concat output is NOT silently converted to add output."""
        config = DMLPFFN_ABLATION_CONFIGS['fusion_concat'].copy()
        config['in_channels'] = 128
        config['num_classes'] = 4

        model = DMLPFFNAblation(**config)

        # The classifier should accept 192 channels (from concat), not 64 (from add)
        # This is what distinguishes proper concat from the buggy historical version
        assert model.classifier.in_features == 192, \
            "Classifier should accept 192 channels for proper concat fusion"

    def test_parameter_count_differs_between_fusion_modes(self):
        """Concat and additive should have different parameter counts (as expected in correct version)."""
        concat_config = DMLPFFN_ABLATION_CONFIGS['fusion_concat'].copy()
        concat_config['in_channels'] = 128
        concat_config['num_classes'] = 4

        add_config = DMLPFFN_ABLATION_CONFIGS['full'].copy()
        add_config['in_channels'] = 128
        add_config['num_classes'] = 4

        concat_model = DMLPFFNAblation(**concat_config)
        add_model = DMLPFFNAblation(**add_config)

        concat_params = sum(p.numel() for p in concat_model.parameters())
        add_params = sum(p.numel() for p in add_model.parameters())

        # Diff is exactly 512 params: (192-64)*4 = 512
        expected_diff = 512
        actual_diff = concat_params - add_params

        assert actual_diff == expected_diff, \
            f"Expected param diff of {expected_diff} between concat and add modes, got {actual_diff}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])