"""
Corrected DMLPFFN Ablation Implementation (Fixed Version)

This module provides the corrected DMLPFFN ablation implementation with proper
concat fusion handling. It is derived from the historical codebase but fixes
the fusion_concat bug where the concat result was incorrectly overwritten.

BUG FIX: In the historical implementation (historical_baseline_models.py), lines 126-128
erroneously overwrote the torch.cat result with `low + mid + high`. This version
properly uses the concatenated output for concat fusion.

Source:
- Historical (buggy): codes/legacy/historical_baseline_models.py (bug preserved at lines 126-128)
- Archive: _archive/CODES/baseline_models.py
- This version fixes ONLY the concat fusion bug, preserving all other behavior

Parameter counts (corrected version):
- full, dilation_137: 447,780 (64-channel classifier, additive fusion)
- fusion_concat: 448,292 (192-channel classifier, proper concat fusion)
- no_global: 416,932
- no_partition: 426,004
- no_local: 191,028
- only_global: 169,252
- only_partition: 160,180
- only_local: 395,156
- dilation_1_only: 276,612
- dilation_15: 362,196

MATHEMATICAL NOTE: The historical fusion_concat achieves 447,780 params due to the bug
where concat is immediately overwritten with add. The corrected version uses proper
concat fusion, requiring a 64×3=192 channel classifier input (vs 64 for additive).
This adds 512 params: (192-64)×4 = 512. The signature file (baseline_models_signatures.json)
only specifies 447,780 for the 'full' config - it does NOT guarantee this param count
for fusion_concat, which uses proper concat fusion and has different dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List


# =============================================================================
# Perceptron Modules (Required by DMLPFFNAblation)
# =============================================================================

class GlobalPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.fc3 = nn.Linear(in_channels, out_channels)
        self.fc4 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        # Branch 1: spatial mean + BN + FC
        x1 = x.permute(0,2,3,1).reshape(-1, c)
        x1 = self.fc2(torch.relu(self.fc1(self.bn1(x).mean(dim=[2,3]))))
        # Branch 2: avgpool + BN + FC + FC
        x2 = self.pool(x).view(b, c)
        x2 = self.fc4(torch.relu(self.fc3(self.bn2(x2))))
        # broadcast and combine branches
        x1 = x1.view(b, -1)
        out = x1 + x2
        return out


class PartitionPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class LocalPerceptronAblation(nn.Module):
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


# =============================================================================
# Ablation Model Variants
# =============================================================================

class DMLPFFNAblation(nn.Module):
    """DMLPFFN with configurable ablation options (Correct Version with fixed concat)."""

    def __init__(self, in_channels=128, num_classes=4, dropout=0.7,
                 use_global=True, use_partition=True, use_local=True,
                 fusion_type='add', dilation_rates=[1,2,5]):
        super().__init__()
        self.use_global = use_global
        self.use_partition = use_partition
        self.use_local = use_local
        self.fusion_type = fusion_type
        self.dilation_rates = dilation_rates

        # Low-level branch
        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DMLPBlockAblation(16, 16, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(16, 64, 1)
        )
        # Middle-level branch
        self.mid_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DMLPBlockAblation(32, 32, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(32, 64, 1)
        )
        # High-level branch
        self.high_branch = nn.Sequential(
            DMLPBlockAblation(in_channels, 64, use_global, use_partition, use_local, fusion_type, dilation_rates),
            nn.Conv2d(64, 64, 1)
        )

        # For concat fusion: classifier accepts 192 channels from concat, 64 from add
        fused_channels = 192 if fusion_type == 'concat' else 64
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused_channels, num_classes)

    def forward(self, x):
        low = self.low_branch(x)
        mid = self.mid_branch(x)
        high = self.high_branch(x)

        if self.fusion_type == 'add':
            fused = low + mid + high
        elif self.fusion_type == 'concat':
            # CORRECT: Use concatenation, do NOT overwrite with add
            fused = torch.cat([low, mid, high], dim=1)
        else:
            fused = low + mid + high

        pooled = self.gap(fused).flatten(1)
        pooled = self.dropout(pooled)
        out = self.classifier(pooled)
        return out


class DMLPBlockAblation(nn.Module):
    """DMLP Block with ablation options (Corrected version)."""

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

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

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

        # Fusion
        if self.fusion_type == 'add':
            spatial_outputs = [o[1] for o in outputs if o[0] != 'global']
            global_outputs = [o[1] for o in outputs if o[0] == 'global']

            if spatial_outputs:
                fusion = sum(spatial_outputs)
            else:
                fusion = torch.zeros_like(self.shortcut(x))

            fusion = self.bn(fusion)
            fusion = fusion + self.shortcut(x)

            if global_outputs:
                fusion = fusion + global_outputs[0].unsqueeze(-1).unsqueeze(-1)

            fusion = torch.relu(fusion)
            return fusion

        return self.shortcut(x)


def get_ablation_model(config: dict, in_channels=128, num_classes=4, dropout=0.7):
    """Create DMLPFFN ablation variant from config."""
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


# Ablation configurations
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
    # Validation: Test all 11 ablation configurations
    print("Testing Corrected DMLPFFN Ablation Configurations (11 configs)")
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