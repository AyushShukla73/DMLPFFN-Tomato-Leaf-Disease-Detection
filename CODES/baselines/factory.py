"""
Unified factory for baseline models and ablations.
Replaces the old baseline_models.py functionality with the new modular architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

# Import baseline models
from CODES.baselines.simple_cnn import SimpleCNN, count_parameters as count_parameters_simple_cnn
from CODES.baselines.resnet18_hsi import ResNet18_HSI, count_parameters as count_parameters_resnet18
from CODES.baselines.mlp_mixer_hsi import MLPMixer_HSI, count_parameters as count_parameters_mlp_mixer
from CODES.baselines.classical_ml import ClassicalMLBaselines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from CODES.config_baselines import EXPERIMENTS, DMLPFFN_REFERENCE, TrainingConfig, get_all_experiment_names


# Wrapper classes for classical ML to provide consistent interface
class SVMPipeline(nn.Module):
    """Wrapper for SVM pipeline to provide nn.Module-like interface."""
    def __init__(self):
        super().__init__()
        self.pipeline = ClassicalMLBaselines.get_svm_pipeline()

    def forward(self, x):
        # x is expected to be [B, C, H, W] - flatten spatial dims
        B, C, H, W = x.shape
        x_flat = x.reshape(B, -1).cpu().numpy()
        probs = self.pipeline.predict_proba(x_flat)
        return torch.from_numpy(probs).float()


class RandomForestPipeline(nn.Module):
    """Wrapper for Random Forest pipeline to provide nn.Module-like interface."""
    def __init__(self):
        super().__init__()
        self.pipeline = ClassicalMLBaselines.get_rf_pipeline()

    def forward(self, x):
        # x is expected to be [B, C, H, W] - flatten spatial dims
        B, C, H, W = x.shape
        x_flat = x.reshape(B, -1).cpu().numpy()
        probs = self.pipeline.predict_proba(x_flat)
        return torch.from_numpy(probs).float()


def count_parameters(model: nn.Module):
    """Unified parameter counting function.
    Returns (total_params, trainable_params) tuple for backward compatibility.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_parameters_detailed(model: nn.Module) -> Dict[str, int]:
    """Detailed parameter count breakdown."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def get_baseline_model(name: str, in_channels: int = 128, num_classes: int = 4, **kwargs) -> nn.Module:
    """
    Factory function to create baseline models.

    Args:
        name: Model name from EXPERIMENTS config (simple_cnn, resnet18_hsi, mlp_mixer_hsi)
        in_channels: Input channels (PCA components)
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    # Handle classical ML as special cases (not in EXPERIMENTS)
    if name == 'svm':
        return SVMPipeline()
    elif name == 'rf' or name == 'random_forest':
        return RandomForestPipeline()
    elif name not in EXPERIMENTS:
        raise ValueError(f"Unknown model: {name}. Available: {get_all_experiment_names()}")

    exp = EXPERIMENTS[name]
    model_class_name = exp['model_class']
    module_name = exp['module']
    config = exp['config']

    # Handle deep learning models - dynamically import the class
    module = __import__(module_name, fromlist=[model_class_name])
    model_class = getattr(module, model_class_name)

    model_kwargs = {
        'in_channels': in_channels,
        'num_classes': num_classes,
    }

    # Add config-specific parameters
    if hasattr(config, '__dict__'):
        for key, value in config.__dict__.items():
            # Skip metadata fields that aren't model constructor args
            if key not in [
                'expected_params', 'name', 'expected_vram_gb',
                'input_format', 'num_classes', 'in_channels',
                'use_pretrained', 'conv1_kernel_size', 'conv1_stride',
                'conv1_padding', 'remove_maxpool', 'fallback_batch_size',
                'fallback_grad_accum', 'pca_components', 'kernel', 'C', 'gamma',
                'class_weight', 'probability', 'random_state', 'n_estimators',
                'max_depth', 'n_jobs', 'dropout', 'token_mixing_dim',
                'channel_mixing_dim', 'num_blocks', 'embedding_dim', 'patch_size'
            ]:
                model_kwargs[key] = value

    # Override with any explicit kwargs
    model_kwargs.update(kwargs)

    # Instantiate model
    model = model_class(**model_kwargs)
    return model


def get_all_baseline_models(in_channels: int = 128, num_classes: int = 4) -> Dict[str, nn.Module]:
    """Get all baseline models as a dictionary."""
    models = {}
    for name in get_all_experiment_names():
        if name in ['svm', 'rf']:
            # Classical ML - store pipeline
            models[name] = get_baseline_model(name, in_channels, num_classes)
        else:
            models[name] = get_baseline_model(name, in_channels, num_classes)
    return models


# =====================================================================
# DMLPFFN Ablation Study - for the ACTUAL DMLPFFN architecture
# =====================================================================

from CODES.dmlpffn_disease import DMLPFFN, DMLPBlock, GlobalPerceptron, PartitionPerceptron, LocalPerceptron


class DMLPFFNAblation(nn.Module):
    """
    Ablation version of the ACTUAL DMLPFFN architecture.
    The actual DMLPFFN has 3 parallel branches (low, mid, high), each with DMLPBlock.
    DMLPBlock combines GlobalPerceptron, PartitionPerceptron, LocalPerceptron.

    This ablation allows:
    - Removing low/mid/high branches
    - Removing individual perceptrons within DMLPBlock
    - Changing fusion mode
    """
    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 4,
        dropout: float = 0.7,
        use_low: bool = True,
        use_mid: bool = True,
        use_high: bool = True,
        # Within DMLPBlock: enable/disable individual perceptrons
        use_global: bool = True,
        use_partition: bool = True,
        use_local: bool = True,
        # Fusion mode: 'add' or 'concat'
        fusion_mode: str = 'add',
        # Dilation rates for LocalPerceptron
        dilation_rates: List[int] = None,
        # Channel dimensions for each branch
        low_channels: int = 16,
        mid_channels: int = 32,
        high_channels: int = 64,
        branch_output_channels: int = 64,  # Output channels per branch (before fusion)
        fused_channels: int = None,  # Auto-computed if None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_low = use_low
        self.use_mid = use_mid
        self.use_high = use_high
        self.use_global = use_global
        self.use_partition = use_partition
        self.use_local = use_local
        self.fusion_mode = fusion_mode
        self.dilation_rates = dilation_rates if dilation_rates is not None else [1, 2, 5]

        # Auto-compute fused_channels based on fusion_mode if not specified
        # Each branch outputs 'branch_output_channels' channels
        # For 'add' fusion: final channels = branch_output_channels (64)
        # For 'concat' fusion: final channels = 3 * branch_output_channels (192)
        if fused_channels is None:
            if fusion_mode == 'concat':
                fused_channels = 3 * branch_output_channels
            else:
                fused_channels = branch_output_channels

        # Low-level branch
        if use_low:
            self.low_branch = nn.Sequential(
                nn.Conv2d(in_channels, low_channels, 3, padding=1),
                nn.BatchNorm2d(low_channels),
                nn.ReLU(inplace=True),
                AblationDMLPBlock(
                    in_channels=low_channels,
                    out_channels=low_channels,
                    use_global=use_global,
                    use_partition=use_partition,
                    use_local=use_local,
                    dilation_rates=self.dilation_rates,
                ),
                nn.Conv2d(low_channels, branch_output_channels, 1)
            )

        # Middle-level branch
        if use_mid:
            self.mid_branch = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 5, padding=2),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                AblationDMLPBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    use_global=use_global,
                    use_partition=use_partition,
                    use_local=use_local,
                    dilation_rates=self.dilation_rates,
                ),
                nn.Conv2d(mid_channels, branch_output_channels, 1)
            )

        # High-level branch
        if use_high:
            self.high_branch = nn.Sequential(
                AblationDMLPBlock(
                    in_channels=in_channels,
                    out_channels=high_channels,
                    use_global=use_global,
                    use_partition=use_partition,
                    use_local=use_local,
                    dilation_rates=self.dilation_rates,
                ),
                nn.Conv2d(high_channels, branch_output_channels, 1)
            )

        self.fused_channels = fused_channels

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.fused_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        branch_outputs = []

        if self.use_low:
            branch_outputs.append(self.low_branch(x))
        if self.use_mid:
            branch_outputs.append(self.mid_branch(x))
        if self.use_high:
            branch_outputs.append(self.high_branch(x))

        # Fuse branches by fusion_mode
        if self.fusion_mode == 'concat':
            fused = torch.cat(branch_outputs, dim=1)
        else:
            fused = branch_outputs[0]
            for feat in branch_outputs[1:]:
                fused = fused + feat

        pooled = self.gap(fused).flatten(1)
        pooled = self.dropout(pooled)
        out = self.classifier(pooled)
        return out


class LocalPerceptronAblation(nn.Module):
    """Local Perceptron with configurable dilation rates."""

    def __init__(self, in_channels: int, out_channels: int, dilation_rates: List[int] = None):
        super().__init__()
        self.dilation_rates = dilation_rates if dilation_rates is not None else [1, 2, 5]
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d),
                nn.BatchNorm2d(out_channels)
            ) for d in self.dilation_rates
        ])

    def forward(self, x):
        return sum([conv(x) for conv in self.dilated_convs])


class AblationDMLPBlock(nn.Module):
    """
    Ablation version of DMLPBlock that can selectively disable
    Global, Partition, and Local perceptrons.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_global: bool = True,
        use_partition: bool = True,
        use_local: bool = True,
        dilation_rates: List[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_global = use_global
        self.use_partition = use_partition
        self.use_local = use_local
        self.dilation_rates = dilation_rates if dilation_rates is not None else [1, 2, 5]

        # Perceptrons (only create if enabled)
        if use_global:
            self.global_perc = GlobalPerceptron(in_channels, out_channels)
        if use_partition:
            self.partition_perc = PartitionPerceptron(in_channels, out_channels)
        if use_local:
            self.local_perc = LocalPerceptronAblation(in_channels, out_channels, self.dilation_rates)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch_outputs = []

        if self.use_global:
            g = self.global_perc(x)
            branch_outputs.append(g.unsqueeze(-1).unsqueeze(-1))  # Expand to spatial

        if self.use_partition:
            p = self.partition_perc(x)
            branch_outputs.append(p)

        if self.use_local:
            l = self.local_perc(x)
            branch_outputs.append(l)

        # Fuse available branches
        if not branch_outputs:
            # Fallback: just shortcut
            fused = self.shortcut(x)
            fused = self.bn(fused)
            return self.relu(fused)

        # Apply fusion mode
        if len(branch_outputs) == 1:
            fused = branch_outputs[0]
        else:
            # Check if we should use concat (only valid when multiple spatial outputs exist)
            # Global perceptron produces 2D output that needs broadcasting
            spatial_outputs = []
            global_outputs = []
            for name, out in branch_outputs if isinstance(branch_outputs[0], tuple) else [(None, branch_outputs[0])]:
                pass

            # For simplicity in DMLPBlock context, always use addition
            # Concat fusion is handled at the DMLPFFNAblation level
            fused = branch_outputs[0]
            for feat in branch_outputs[1:]:
                fused = fused + feat

        # Add residual
        fused = fused + self.shortcut(x)

        # BatchNorm and ReLU
        fused = self.bn(fused)
        fused = self.relu(fused)

        return fused


# Ablation configurations for the ACTUAL DMLPFFN architecture
# The new architecture has:
# - 3 parallel branches: low, mid, high
# - Each branch has AblationDMLPBlock with Global/Partition/Local perceptrons
# - DMLPFFNAblation enables/disables branches and perceptrons within blocks
DMLPFFN_ABLATION_CONFIGS = {
    # Full model (reference) - all branches and all perceptrons
    'full': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    # Single branch ablations (keep all perceptrons in that branch)
    'only_low': {
        'use_low': True,
        'use_mid': False,
        'use_high': False,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    'only_mid': {
        'use_low': False,
        'use_mid': True,
        'use_high': False,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    'only_high': {
        'use_low': False,
        'use_mid': False,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    # Two branch ablations
    'no_low': {
        'use_low': False,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    'no_mid': {
        'use_low': True,
        'use_mid': False,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    'no_high': {
        'use_low': True,
        'use_mid': True,
        'use_high': False,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    # Perceptron ablations within blocks (all branches enabled, disable specific perceptrons)
    'no_global_perc': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': False,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    'no_partition_perc': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': False,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    'no_local_perc': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': False,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    # Single perceptron ablations
    'only_global_perc': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': False,
        'use_local': False,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    'only_partition_perc': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': False,
        'use_partition': True,
        'use_local': False,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    'only_local_perc': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': False,
        'use_partition': False,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    # Width ablations
    'narrow_32ch': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 8,
        'mid_channels': 16,
        'high_channels': 32,
        'fused_channels': 32,
        'dropout': 0.7,
    },
    'wide_64ch': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 32,
        'mid_channels': 64,
        'high_channels': 128,
        'fused_channels': 128,
        'dropout': 0.7,
    },
    # Fusion mode ablations
    'fusion_concat': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'dropout': 0.7,
        'fusion_mode': 'concat',
        'branch_output_channels': 64,  # Each branch outputs 64 channels
    },
    # Dilation rate ablations for high branch
    'dilation_137': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
        'dilation_rates': [1, 3, 7],
    },
    'dilation_15': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
        'dilation_rates': [1, 5],
    },
    'dilation_1_only': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
        'dilation_rates': [1],
    },
    # Full model variants for completeness
    'full_dmlpffn': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
    },
    'full_resnet_style': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
        'resnet_style_downsample': True,
    },
    'full_with_attention': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
        'use_attention': True,
    },
    'full_enhanced': {
        'use_low': True,
        'use_mid': True,
        'use_high': True,
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'low_channels': 16,
        'mid_channels': 32,
        'high_channels': 64,
        'fused_channels': 64,
        'dropout': 0.7,
        'enhanced_activation': True,
    },
}


def get_ablation_model(config_name: str, in_channels: int = 128, num_classes: int = 4, dropout: float = 0.7) -> nn.Module:
    """
    Factory function to create DMLPFFN ablation models.

    Args:
        config_name: Name of ablation config from DMLPFFN_ABLATION_CONFIGS
        in_channels: Input channels
        num_classes: Number of output classes
        dropout: Dropout rate

    Returns:
        Initialized DMLPFFNAblation model
    """
    if config_name not in DMLPFFN_ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation config: {config_name}. Available: {list(DMLPFFN_ABLATION_CONFIGS.keys())}")

    config = DMLPFFN_ABLATION_CONFIGS[config_name].copy()
    config['in_channels'] = in_channels
    config['num_classes'] = num_classes
    config['dropout'] = dropout

    return DMLPFFNAblation(**config)


# For backward compatibility with old code that expects ABLATION_CONFIGS
ABLATION_CONFIGS = DMLPFFN_ABLATION_CONFIGS


def get_dmlpffn_reference(in_channels: int = 128, num_classes: int = 4, dropout: float = 0.7) -> nn.Module:
    """Get the reference DMLPFFN model (full architecture)."""
    from CODES.dmlpffn_disease import DMLPFFN
    return DMLPFFN(in_channels=in_channels, num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    # Test factory
    print("Testing baseline factory...")

    # Test baseline models
    for name in ['simple_cnn', 'resnet18_hsi', 'mlp_mixer_hsi', 'svm', 'random_forest']:
        model = get_baseline_model(name, in_channels=128, num_classes=4)
        total_params, trainable_params = count_parameters(model)
        print(f"  {name}: {total_params:,} params ({trainable_params:,} trainable)")

    # Test ablation models
    print("\nTesting DMLPFFN ablation models...")
    for config_name in ['full', 'no_global_perc', 'no_partition_perc', 'no_local_perc', 'only_global_perc', 'narrow_32ch', 'wide_64ch']:
        model = get_ablation_model(config_name, in_channels=128, num_classes=4, dropout=0.7)
        total_params, trainable_params = count_parameters(model)
        print(f"  {config_name}: {total_params:,} params ({trainable_params:,} trainable)")

    print("\nAll factory tests passed!")