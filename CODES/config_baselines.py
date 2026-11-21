"""
Baseline Model Configurations for Phase 5 Experiments
Centralized configuration for all 5 baselines + DMLPFFN reference
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import torch


# =============================================================================
# Hardware & Device Configuration
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True  # Mixed precision for all DL models

# RTX 3050 4GB VRAM constraints
VRAM_LIMIT_GB = 4.0
BATCH_SIZE = 16
BATCH_SIZE_RESNET18 = 16  # May need to reduce to 8 if OOM
GRAD_ACCUM_STEPS = 1  # Increase to 2 for ResNet-18 if batch reduced


# =============================================================================
# Frozen Data Split (from artifacts/split_manifest.json, seed=42)
# =============================================================================

SPLIT_MANIFEST_PATH = "artifacts/split_manifest_frozen.json"

# Expected class distribution (validated in technical validation)
TRAIN_CLASSES = {0: 1316, 1: 756, 2: 1512, 3: 5670}
VAL_CLASSES = {0: 196, 2: 182, 3: 1316}
TEST_CLASSES = {0: 574, 3: 756}  # Only Healthy (0) and Late (3) in test

# Binary evaluation mapping
CLASS_HEALTHY = 0
CLASS_LATE = 3
BINARY_CLASS_MAP = {CLASS_HEALTHY: 0, CLASS_LATE: 1}  # 4-class -> binary


# =============================================================================
# Unified Training Configuration (Phase 3 Regularization - Immutable)
# =============================================================================

@dataclass
class TrainingConfig:
    """Phase 3 training configuration - IDENTICAL for all DL baselines"""
    # Optimization
    batch_size: int = BATCH_SIZE
    learning_rate: float = 2e-4
    weight_decay: float = 5e-4
    epochs: int = 250
    early_stop_patience: int = 25

    # Scheduler
    scheduler_type: str = "CosineAnnealingWarmRestarts"
    scheduler_T_0: int = 10
    scheduler_T_mult: int = 1

    # Regularization (Phase 3)
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    cutmix_prob: float = 0.5

    # Class balancing
    use_weighted_sampler: bool = True

    # Loss
    loss_type: str = "CrossEntropy"  # or "FocalLoss"
    focal_gamma: float = 2.0

    # Hardware
    device: str = "cuda"
    use_amp: bool = USE_AMP
    num_workers: int = 0  # Windows compatibility
    pin_memory: bool = True

    # Reproducibility
    seed: int = 42

    # Logging
    log_interval: int = 10
    save_best_only: bool = True


# =============================================================================
# Model-Specific Configurations
# =============================================================================

@dataclass
class ModelConfig:
    """Base model configuration"""
    name: str
    num_classes: int = 4  # 4-class head (binary evaluation extracts 2 classes)
    in_channels: int = 128
    input_format: str = "NHWC"  # All baselines receive [B, 9, 9, 128]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class SimpleCNNConfig(ModelConfig):
    """SimpleCNN HSI-Adapted"""
    base_channels: int = 44  # Hits ~86K params
    expected_params: int = 86156
    expected_vram_gb: float = 1.2


@dataclass
class ResNet18Config(ModelConfig):
    """ResNet-18 128-Channel Adapted"""
    use_pretrained: bool = False
    conv1_kernel_size: int = 3
    conv1_stride: int = 1
    conv1_padding: int = 1
    remove_maxpool: bool = True
    expected_params: int = 11242884
    expected_vram_gb: float = 3.5
    # VRAM fallback
    fallback_batch_size: int = 8
    fallback_grad_accum: int = 2


@dataclass
class MLPMixerConfig(ModelConfig):
    """MLP-Mixer Patch-Based (Corrected Architecture)"""
    patch_size: int = 9
    embedding_dim: int = 256
    num_blocks: int = 8
    token_mixing_dim: int = 128  # Across spatial tokens (81)
    channel_mixing_dim: int = 512  # Across embedding dim (256)
    dropout: float = 0.1
    expected_params: int = 2310812
    expected_vram_gb: float = 2.8


@dataclass
class SVMConfig(ModelConfig):
    """SVM RBF Kernel"""
    # Pipeline: PCA(256) -> StandardScaler -> SVC(RBF, C=10)
    pca_components: int = 256
    kernel: str = "rbf"
    C: float = 10.0
    gamma: str = "scale"
    class_weight: str = "balanced"
    probability: bool = True
    random_state: int = 42
    expected_vram_gb: str = "CPU"


@dataclass
class RandomForestConfig(ModelConfig):
    """Random Forest"""
    # Pipeline: PCA(256) -> StandardScaler -> RF(500, max_depth=20)
    pca_components: int = 256
    n_estimators: int = 500
    max_depth: int = 20
    class_weight: str = "balanced"
    n_jobs: int = -1
    random_state: int = 42
    expected_vram_gb: str = "CPU"


@dataclass
class DMLPFFNConfig(ModelConfig):
    """DMLPFFN Reference (Phase 3 Architecture)"""
    dropout: float = 0.7
    expected_params: int = 447780
    expected_vram_gb: float = 1.5
    input_format: str = "NCHW"  # Different from baselines!


# =============================================================================
# Seed Configuration (5 seeds for DL baselines)
# =============================================================================

SEEDS = [42, 123, 456, 789, 999]
N_SEEDS_DL = len(SEEDS)
N_SEEDS_CLASSICAL = 1  # SVM/RF deterministic


# =============================================================================
# Experiment Configurations (Complete)
# =============================================================================

EXPERIMENTS = {
    "simple_cnn": {
        "model_class": "SimpleCNN",
        "module": "CODES.baselines.simple_cnn",
        "config": SimpleCNNConfig(name="simple_cnn"),
        "training_config": TrainingConfig(),
        "seeds": SEEDS,
        "output_dir": "artifacts/baselines/simple_cnn",
        "is_dl": True,
    },
    "mlp_mixer_hsi": {
        "model_class": "MLPMixer_HSI",
        "module": "CODES.baselines.mlp_mixer_hsi",
        "config": MLPMixerConfig(name="mlp_mixer_hsi"),
        "training_config": TrainingConfig(),
        "seeds": SEEDS,
        "output_dir": "artifacts/baselines/mlp_mixer_hsi",
        "is_dl": True,
    },
    "resnet18_hsi": {
        "model_class": "ResNet18_HSI",
        "module": "CODES.baselines.resnet18_hsi",
        "config": ResNet18Config(name="resnet18_hsi"),
        "training_config": TrainingConfig(
            batch_size=BATCH_SIZE_RESNET18,
            # Note: If OOM, will auto-reduce to fallback_batch_size
        ),
        "seeds": SEEDS,
        "output_dir": "artifacts/baselines/resnet18_hsi",
        "is_dl": True,
    },
    "svm": {
        "model_class": "SVMPipeline",
        "module": "CODES.baselines.classical_ml",
        "config": SVMConfig(name="svm"),
        "training_config": None,  # Uses sklearn fit directly
        "seeds": [42],  # Single deterministic run
        "output_dir": "artifacts/baselines/svm",
        "is_dl": False,
    },
    "random_forest": {
        "model_class": "RandomForestPipeline",
        "module": "CODES.baselines.classical_ml",
        "config": RandomForestConfig(name="random_forest"),
        "training_config": None,
        "seeds": [42],
        "output_dir": "artifacts/baselines/random_forest",
        "is_dl": False,
    },
}


# =============================================================================
# DMLPFFN Reference Configuration
# =============================================================================

DMLPFFN_REFERENCE = {
    "model_class": "DMLPFFN",
    "module": "CODES.dmlpffn_disease",  # Existing Phase 3 implementation
    "config": DMLPFFNConfig(name="dmlpffn_phase3"),
    "training_config": None,  # Already trained in Phase 3
    "seeds": [42],  # Phase 3 single run
    "output_dir": "artifacts/baselines/dmlpffn_phase3",
    "is_dl": True,
    "phase3_results": {
        "val_accuracy": 0.7875,
        "test_accuracy": 0.9564,
        "macro_f1": 0.9660,
        "weighted_f1": 0.9658,
        "parameters": 447780,
        "train_val_gap": 0.1108,
        "note": "Single run (Phase 3), 4-class head with conditional binary evaluation"
    }
}


# =============================================================================
# Evaluation Configuration
# =============================================================================

EVALUATION_CONFIG = {
    "mode": "conditional_binary",  # Only supported mode
    "binary_classes": [CLASS_HEALTHY, CLASS_LATE],  # Extract from 4-class head
    "class_names": ["Healthy", "Late"],
    "metrics": [
        "accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
        "per_class_precision", "per_class_recall", "per_class_f1",
        "confusion_matrix",
        "auc_roc",
    ],
    "group_level_metrics": True,
    "statistical_tests": {
        "bootstrap_ci": {
            "enabled": True,
            "n_resamples": 1000,
            "confidence": 0.95,
            "group_aware": True,
        },
        "mcnemar": {
            "enabled": True,
            "group_aware": True,
            "alpha": 0.05,
        },
    },
}


# =============================================================================
# Artifact Output Structure
# =============================================================================

ARTIFACT_STRUCTURE = {
    "base_dir": "artifacts/baselines",
    "per_seed": {
        "best_model.pth",       # DL: model weights
        "history.json",         # DL: training history
        "test_results.json",    # All: evaluation results
        "test_logits.npy",      # DL: raw logits for re-evaluation
        "test_targets.npy",     # All: ground truth
        "test_groups.npy",      # All: group IDs
    },
    "aggregate": {
        "aggregate_results.json",  # Mean ± std across seeds
        "comparison_table.md",     # Markdown comparison
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_training_config(model_name: str) -> TrainingConfig:
    """Get training config for a specific model."""
    exp = EXPERIMENTS.get(model_name)
    if exp and exp["training_config"]:
        return exp["training_config"]
    return TrainingConfig()


def get_model_config(model_name: str) -> ModelConfig:
    """Get model config for a specific model."""
    exp = EXPERIMENTS.get(model_name)
    if exp:
        return exp["config"]
    raise ValueError(f"Unknown model: {model_name}")


def get_experiment_config(model_name: str) -> Dict[str, Any]:
    """Get full experiment config for a model."""
    exp = EXPERIMENTS.get(model_name)
    if not exp:
        raise ValueError(f"Unknown model: {model_name}")
    return exp


def get_all_experiment_names() -> List[str]:
    """Get all experiment names in execution order."""
    return ["simple_cnn", "mlp_mixer_hsi", "resnet18_hsi", "svm", "random_forest"]


def get_dl_experiment_names() -> List[str]:
    """Get only DL experiment names."""
    return ["simple_cnn", "mlp_mixer_hsi", "resnet18_hsi"]


def get_classical_experiment_names() -> List[str]:
    """Get only classical ML experiment names."""
    return ["svm", "random_forest"]


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("PHASE 5 BASELINE CONFIGURATIONS")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")
    print(f"Mixed Precision: {USE_AMP}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Seeds (DL): {SEEDS}")
    print(f"Seeds (Classical): {[42]}")

    print("\n--- Deep Learning Baselines ---")
    for name in get_dl_experiment_names():
        exp = EXPERIMENTS[name]
        cfg = exp["config"]
        print(f"\n{name}:")
        print(f"  Module: {exp['module']}")
        print(f"  Class: {exp['model_class']}")
        print(f"  Params: ~{cfg.expected_params:,}")
        print(f"  VRAM: ~{cfg.expected_vram_gb} GB")
        print(f"  Seeds: {exp['seeds']}")

    print("\n--- Classical ML Baselines ---")
    for name in get_classical_experiment_names():
        exp = EXPERIMENTS[name]
        cfg = exp["config"]
        print(f"\n{name}:")
        print(f"  Module: {exp['module']}")
        print(f"  Class: {exp['model_class']}")
        print(f"  VRAM: {cfg.expected_vram_gb}")

    print("\n--- DMLPFFN Reference ---")
    ref = DMLPFFN_REFERENCE
    print(f"  Module: {ref['module']}")
    print(f"  Class: {ref['model_class']}")
    print(f"  Phase 3 Test Acc: {ref['phase3_results']['test_accuracy']:.4f}")
    print(f"  Input Format: {ref['config'].input_format}")

    print("\n--- Training Config (Phase 3, identical for all DL) ---")
    tc = TrainingConfig()
    print(f"  Batch Size: {tc.batch_size}")
    print(f"  LR: {tc.learning_rate}")
    print(f"  Weight Decay: {tc.weight_decay}")
    print(f"  Epochs: {tc.epochs} (patience={tc.early_stop_patience})")
    print(f"  Label Smoothing: {tc.label_smoothing}")
    print(f"  Mixup α: {tc.mixup_alpha} (p={tc.mixup_prob})")
    print(f"  CutMix α: {tc.cutmix_alpha} (p={tc.cutmix_prob})")
    print(f"  Scheduler: {tc.scheduler_type}(T_0={tc.scheduler_T_0})")
    print(f"  Loss: {tc.loss_type}")

    print("\n--- Evaluation Protocol ---")
    print(f"  Mode: {EVALUATION_CONFIG['mode']}")
    print(f"  Binary Classes: {EVALUATION_CONFIG['class_names']}")
    print(f"  Metrics: {', '.join(EVALUATION_CONFIG['metrics'])}")
    print(f"  Bootstrap CI: {EVALUATION_CONFIG['statistical_tests']['bootstrap_ci']['n_resamples']} resamples")
    print(f"  McNemar: group_aware={EVALUATION_CONFIG['statistical_tests']['mcnemar']['group_aware']}")

    print("\n" + "=" * 60)