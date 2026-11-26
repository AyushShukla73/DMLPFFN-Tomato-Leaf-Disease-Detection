"""
Historical DMLPFFN Ablation Implementation (Legacy)

This module preserves the historical DMLPFFN ablation implementation from
_archive/CODES/baseline_models.py for reproducibility and reference purposes.

Key differences from modern modular architecture (CODES/baselines/factory.py):
- Model naming: simplecnn (historical) vs simple_cnn (modern)
- 11 ablation configurations (historical) vs 23 configurations (modern)
- Monolithic implementation vs modular factory pattern
- No physical checkpoint file exists - source code only

Historical Ablation Configurations (11):
- full: All branches, all perceptrons, default dilation [1,2,5]
- no_global: Disable global perceptron
- no_partition: Disable partition perceptron
- no_local: Disable local perceptron
- only_global: Only global perceptron enabled
- only_partition: Only partition perceptron enabled
- only_local: Only local perceptron enabled
- fusion_concat: Concat fusion instead of additive
- dilation_137: Dilation rates [1,3,7]
- dilation_1_only: Single dilation rate [1]
- dilation_15: Dilation rates [1,5]

The 4 additional full-model variants (full_dmlpffn, full_resnet_style,
full_with_attention, full_enhanced) are Phase 5 modern additions (commit 5e9c0a6)
and are NOT part of this historical implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


# =============================================================================
# Perceptron Modules (Required by DMLPFFNAblation)
# =============================================================================

# Global Perceptron Module
class GlobalPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Branch 1: split & BN + FC + ReLU + FC
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        # Branch 2: avgpool → BN + FC + ReLU + FC
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.fc3 = nn.Linear(in_channels, out_channels)
        self.fc4 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        # Branch 1: spatial mean + BN + FC
        x1 = self.bn1(x).mean(dim=[2, 3])  # (b, c, 1, 1) -> mean -> (b, c)
        x1 = self.fc2(torch.relu(self.fc1(x1.squeeze(-1).squeeze(-1))))
        # Branch 2: avgpool + BN + FC
        x2 = self.pool(x).view(b, c)
        x2 = self.fc4(torch.relu(self.fc3(self.bn2(x2))))
        # broadcast and combine branches
        out = x1 + x2
        return out


# Partition Perceptron Module
class PartitionPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


# =============================================================================
# Ablation Model Variants (Historical Implementation)
# =============================================================================

class DMLPFFNAblation(nn.Module):
    """DMLPFFN with configurable ablation options (Historical Version)."""

    def __init__(self, in_channels=128, num_classes=4, dropout=0.7,
                 use_global=True, use_partition=True, use_local=True,
                 fusion_type='add', dilation_rates=[1,2,5]):
        super().__init__()
        self.use_global = use_global
        self.use_partition = use_partition
        self.use_local = use_local
        self.fusion_type = fusion_type
        self.dilation_rates = dilation_rates

        # Calculate output channels from DMLPBlockAblation (accounts for concat fusion)
        # When fusion_type is 'concat' and multiple perceptrons enabled, output is summed
        # Otherwise, output equals out_channels (additive fusion)
        low_out = 16 if fusion_type != 'concat' else (sum([16 if use_global else 0, 16 if use_partition else 0, 16 if use_local else 0]))
        mid_out = 32 if fusion_type != 'concat' else (sum([32 if use_global else 0, 32 if use_partition else 0, 32 if use_local else 0]))
        high_out = 64 if fusion_type != 'concat' else (sum([64 if use_global else 0, 64 if use_partition else 0, 64 if use_local else 0]))

        # Low-level branch
        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DMLPBlockAblation(16, 16, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(low_out, 64, 1)
        )
        # Middle-level branch
        self.mid_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DMLPBlockAblation(32, 32, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(mid_out, 64, 1)
        )
        # High-level branch
        self.high_branch = nn.Sequential(
            DMLPBlockAblation(in_channels, 64, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(high_out, 64, 1)
        )

        # Fusion channel size depends on fusion type
        if fusion_type == 'concat':
            fused_channels = 192  # 64 * 3
        else:
            fused_channels = 64  # 64 * 1 (additive)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused_channels, num_classes)

    def forward(self, x):
        low = self.low_branch(x)
        mid = self.mid_branch(x)
        high = self.high_branch(x)

        if self.fusion_type == 'concat':
            fused = torch.cat([low, mid, high], dim=1)
        else:
            fused = low + mid + high

        pooled = self.gap(fused).flatten(1)
        pooled = self.dropout(pooled)
        out = self.classifier(pooled)
        return out


class DMLPBlockAblation(nn.Module):
    """DMLP Block with ablation options (Historical Version)."""

    def __init__(self, in_channels, out_channels, use_global=True, use_partition=True,
                 use_local=True, fusion_type='add', dilation_rates=[1,2,5]):
        super().__init__()
        self.use_global = use_global
        self.use_partition = use_partition
        self.use_local = use_local
        self.fusion_type = fusion_type

        if use_global:
            self.global_perc = GlobalPerceptron(in_channels, out_channels)
        if use_partition:
            self.partition_perc = PartitionPerceptron(in_channels, out_channels)
        if use_local:
            self.local_perc = LocalPerceptronAblation(in_channels, out_channels, dilation_rates)

        # Calculate fused channels for BN and shortcut
        if fusion_type == 'concat':
            self.fused_channels = sum([
                out_channels if use_global else 0,
                out_channels if use_partition else 0,
                out_channels if use_local else 0
            ])
        else:
            self.fused_channels = out_channels

        self.shortcut = nn.Conv2d(in_channels, self.fused_channels, 1)
        self.bn = nn.BatchNorm2d(self.fused_channels)

    def forward(self, x):
        outputs = []

        if self.use_global:
            g = self.global_perc(x)
            outputs.append(('global', g))
        if self.use_partition:
            p = self.partition_perc(x)
            outputs.append(('partition', p))
        if self.use_local:
            l = self.local_perc(x)
            outputs.append(('local', l))

        if not outputs:
            return self.shortcut(x)

        # Get spatial dimensions from input
        b, c, h, w = x.shape

        # Gather all outputs and convert to 4D if needed
        processed_outputs = []
        for name, out in outputs:
            if out.dim() == 2:
                # Global perceptron output: broadcast to spatial dimensions
                out = out.unsqueeze(-1).unsqueeze(-1).expand(b, -1, h, w).contiguous()
            processed_outputs.append(out)

        # Fusion
        if self.fusion_type == 'concat':
            fusion = torch.cat(processed_outputs, dim=1)
        else:
            fusion = sum(processed_outputs)

        fusion = self.bn(fusion)
        fusion = torch.relu(fusion)

        # Add shortcut
        shortcut_out = self.shortcut(x)
        fusion = fusion + shortcut_out
        fusion = torch.relu(fusion)
        return fusion


class LocalPerceptronAblation(nn.Module):
    """Local Perceptron with configurable dilation rates (Historical Version)."""

    def __init__(self, in_channels, out_channels, dilation_rates=[1,2,5]):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d),
                nn.BatchNorm2d(out_channels)
            ) for d in dilation_rates
        ])

    def forward(self, x):
        return sum([conv(x) for conv in self.dilated_convs])


def get_ablation_model(config: Dict[str, Any], in_channels=128, num_classes=4, dropout=0.7):
    """Create DMLPFFN ablation variant from config (Historical Version)."""
    return DMLPFFNAblation(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
        use_global=config.get('use_global', True),
        use_partition=config.get('use_partition', True),
        use_local=config.get('use_local', True),
        fusion_type=config.get('fusion_type', 'add'),
        dilation_rates=config.get('dilation_rates', [1,2,5])
    )


# =============================================================================
# Historical Ablation Configurations (11 configs)
# =============================================================================

# Ablation configurations - 11 historical configurations
ABLATION_CONFIGS = {
    'full': {'use_global': True, 'use_partition': True, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'no_global': {'use_global': False, 'use_partition': True, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'no_partition': {'use_global': True, 'use_partition': False, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'no_local': {'use_global': True, 'use_partition': True, 'use_local': False, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'only_global': {'use_global': True, 'use_partition': False, 'use_local': False, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'only_partition': {'use_global': False, 'use_partition': True, 'use_local': False, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'only_local': {'use_global': False, 'use_partition': False, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,2,5]},
    'fusion_concat': {'use_global': True, 'use_partition': True, 'use_local': True, 'fusion_type': 'concat', 'dilation_rates': [1,2,5]},
    'dilation_137': {'use_global': True, 'use_partition': True, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,3,7]},
    'dilation_1_only': {'use_global': True, 'use_partition': True, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1]},
    'dilation_15': {'use_global': True, 'use_partition': True, 'use_local': True, 'fusion_type': 'add', 'dilation_rates': [1,5]},
}


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Validation: Test all 11 historical ablation configurations
    print("Testing Historical DMLPFFN Ablation Configurations (11 configs)")
    print("=" * 60)

    for config_name, config in ABLATION_CONFIGS.items():
        try:
            model = get_ablation_model(config, in_channels=128, num_classes=4, dropout=0.7)
            total_params, trainable_params = count_parameters(model)

            # Test forward pass
            x = torch.randn(2, 128, 9, 9)
            with torch.no_grad():
                out = model(x)

            print(f"OK {config_name:20s} - Params: {total_params:>10,} - Output: {tuple(out.shape)}")
        except Exception as e:
            print(f"FAIL {config_name:20s} - ERROR: {e}")

    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)

    # Print config count
    print(f"\nTotal Historical Ablation Configurations: {len(ABLATION_CONFIGS)}")
    print("Config names:", list(ABLATION_CONFIGS.keys()))