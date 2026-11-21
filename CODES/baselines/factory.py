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
from CODES.config_baselines import EXPERIMENTS, DMLPFFN_REFERENCE, TrainingConfig, get_all_experiment_names


def count_parameters(model: nn.Module) -> int:
    """Unified parameter counting function."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        name: Model name from EXPERIMENTS config (simplecnn, resnet18, mlp_mixer, svm, rf)
        in_channels: Input channels (PCA components)
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown model: {name}. Available: {get_all_experiment_names()}")

    exp = EXPERIMENTS[name]
    model_class_name = exp['model_class']
    module_name = exp['module']
    config = exp['config']

    # Handle classical ML models
    if name in ['svm', 'rf']:
        if name == 'svm':
            return ClassicalMLBaselines.get_svm_pipeline()
        else:
            return ClassicalMLBaselines.get_rf_pipeline()

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
            if key not in ['expected_params', 'name', 'expected_vram_gb', 'input_format', 'num_classes', 'in_channels']:  # Skip metadata
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
    Allows selective removal of Global, Partition, and Local branches.
    """
    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 4,
        base_channels: int = 48,
        num_blocks: int = 4,
        dropout: float = 0.7,
        use_global: bool = True,
        use_partition: bool = True,
        use_local: bool = True,
        global_dilations: List[int] = [1, 2, 5],
        partition_dilations: List[int] = [1, 2, 5],
        local_dilations: List[int] = [1, 2, 5],
        fusion_mode: str = 'add',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.use_global = use_global
        self.use_partition = use_partition
        self.use_local = use_local
        self.fusion_mode = fusion_mode

        # Calculate output channels per branch
        # Global: base_channels * 4 (expansion factor 4)
        # Partition: base_channels * 2 (expansion factor 2)
        # Local: base_channels (expansion factor 1)

        self.global_out = base_channels * 4 if use_global else 0
        self.partition_out = base_channels * 2 if use_partition else 0
        self.local_out = base_channels if use_local else 0

        fused_channels = self.global_out + self.partition_out + self.local_out
        if fused_channels == 0:
            raise ValueError("At least one branch must be enabled")

        # Branch modules
        if use_global:
            self.global_branch = GlobalPerceptron(
                in_channels=in_channels,
                out_channels=self.global_out,
                expansion=4,
                dropout=dropout,
            )

        if use_partition:
            self.partition_branch = PartitionPerceptron(
                in_channels=in_channels,
                out_channels=self.partition_out,
                expansion=2,
                dropout=dropout,
            )

        if use_local:
            self.local_branch = LocalPerceptron(
                in_channels=in_channels,
                out_channels=self.local_out,
                dropout=dropout,
            )

        # DMLP blocks for processing fused features
        self.blocks = nn.ModuleList([
            DMLPBlock(
                in_channels=fused_channels,
                out_channels=fused_channels,
                dilations=global_dilations,  # Use global dilations for fused processing
                dropout=dropout,
                expansion=2,
            )
            for _ in range(num_blocks)
        ])

        # Final classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_channels, num_classes)
        )

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
        """
        x: [B, C, H, W] - NCHW format
        """
        branch_outputs = []

        if self.use_global:
            global_feat = self.global_branch(x)
            branch_outputs.append(global_feat)

        if self.use_partition:
            partition_feat = self.partition_branch(x)
            branch_outputs.append(partition_feat)

        if self.use_local:
            local_feat = self.local_branch(x)
            branch_outputs.append(local_feat)

        # Fusion
        if self.fusion_mode == 'add':
            if len(branch_outputs) == 1:
                fused = branch_outputs[0]
            else:
                fused = branch_outputs[0]
                for feat in branch_outputs[1:]:
                    fused = fused + feat
        elif self.fusion_mode == 'concat':
            fused = torch.cat(branch_outputs, dim=1)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        # Process through DMLP blocks
        for block in self.blocks:
            fused = block(fused)

        # Global average pooling and classification
        pooled = self.global_pool(fused).flatten(1)
        out = self.classifier(pooled)
        return out


# Ablation configurations for the ACTUAL DMLPFFN architecture
DMLPFFN_ABLATION_CONFIGS = {
    # Full model (reference)
    'full': {
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    # Single branch ablations
    'only_global': {
        'use_global': True,
        'use_partition': False,
        'use_local': False,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    'only_partition': {
        'use_global': False,
        'use_partition': True,
        'use_local': False,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    'only_local': {
        'use_global': False,
        'use_partition': False,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    # Two branch ablations
    'no_global': {
        'use_global': False,
        'use_partition': True,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    'no_partition': {
        'use_global': True,
        'use_partition': False,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    'no_local': {
        'use_global': True,
        'use_partition': True,
        'use_local': False,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    # Fusion mode ablation
    'fusion_concat': {
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'fusion_mode': 'concat',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    # Dilation ablations
    'dilation_1_only': {
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1],
        'partition_dilations': [1],
        'local_dilations': [1],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    'dilation_137': {
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1, 3, 7],
        'partition_dilations': [1, 3, 7],
        'local_dilations': [1, 3, 7],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    'dilation_15': {
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1, 5],
        'partition_dilations': [1, 5],
        'local_dilations': [1, 5],
        'num_blocks': 4,
        'base_channels': 48,
        'dropout': 0.7,
    },
    # Depth ablation
    'shallow_2blocks': {
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 2,
        'base_channels': 48,
        'dropout': 0.7,
    },
    'deep_6blocks': {
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 6,
        'base_channels': 48,
        'dropout': 0.7,
    },
    # Width ablation
    'narrow_32ch': {
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 4,
        'base_channels': 32,
        'dropout': 0.7,
    },
    'wide_64ch': {
        'use_global': True,
        'use_partition': True,
        'use_local': True,
        'fusion_mode': 'add',
        'global_dilations': [1, 2, 5],
        'partition_dilations': [1, 2, 5],
        'local_dilations': [1, 2, 5],
        'num_blocks': 4,
        'base_channels': 64,
        'dropout': 0.7,
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
    for name in ['simplecnn', 'resnet18', 'mlp_mixer']:
        model = get_baseline_model(name, in_channels=128, num_classes=4)
        n_params = count_parameters(model)
        print(f"  {name}: {n_params:,} params")

    # Test classical ML
    svm = get_baseline_model('svm')
    rf = get_baseline_model('rf')
    print(f"  svm: {type(svm).__name__}")
    print(f"  rf: {type(rf).__name__}")

    # Test ablation models
    print("\nTesting DMLPFFN ablation models...")
    for config_name in ['full', 'no_global', 'no_partition', 'no_local', 'only_global', 'fusion_concat', 'dilation_1_only']:
        model = get_ablation_model(config_name, in_channels=128, num_classes=4, dropout=0.7)
        n_params = count_parameters(model)
        print(f"  {config_name}: {n_params:,} params")

    print("\nAll factory tests passed!")