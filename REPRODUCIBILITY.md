# Reproducibility Guide

This document provides comprehensive instructions for reproducing experimental results from the DMLPFFN Tomato Leaf Spot Detection project.

## Project Structure

```
DMLPFFN-TomatoLeafSpot-Detection-main/
├── CODES/                           # Modern modular implementation
│   ├── baselines/                     # Modular baseline implementations
│   ├── dmlpffn_disease.py            # Main DMLPFFN architecture
│   ├── phase5_baseline_ablation.py   # Phase 5 experiment script
│   └── config_baselines.py           # Baseline configurations
├── _archive/CODES/                    # Historical implementation archive
│   └── baseline_models.py            # Historical baselines (preserved source)
├── artifacts/
│   ├── frozen/                        # Frozen artifacts for reproducibility
│   │   └── baseline_models_signatures.json
│   ├── best_model.pth
│   ├── dmlpffn_test_logits_frozen.json
│   ├── split_manifest_frozen.json
│   ├── preprocessing_artifacts_sklearn.joblib
│   └── preprocessing_artifacts_arrays.npz
└── REPRODUCIBILITY.md                 # This file
```

**Note:** `baseline_models_historical.pth` does NOT exist as a physical checkpoint file. The historical baseline models are available as source code only in `_archive/CODES/baseline_models.py`.

## Architecture: Historical vs Modern

### Historical Implementation (`CODES/legacy/baseline_models.py`)

The historical DMLPFFN ablation implementation has been restored at:
- **Location**: `CODES/legacy/baseline_models.py` (source-preserved from `_archive/CODES/baseline_models.py`)

Key implementation details:
- Model naming uses snake_case without underscores: `simplecnn`, `resnet18`, `resnet50`, `vit_small`, `mlp_mixer`, `1dcnn`, `spectralnet`, `dbda`
- Source code only - no compiled checkpoints available
- **11 ablation configurations** (historical):
  - `full` - All branches, all perceptrons, default dilation [1,2,5]
  - `no_global` - Disable global perceptron
  - `no_partition` - Disable partition perceptron
  - `no_local` - Disable local perceptron
  - `only_global` - Only global perceptron enabled
  - `only_partition` - Only partition perceptron enabled
  - `only_local` - Only local perceptron enabled
  - `fusion_concat` - Concat fusion instead of additive
  - `dilation_137` - Dilation rates [1,3,7]
  - `dilation_1_only` - Single dilation rate [1]
  - `dilation_15` - Dilation rates [1,5]
- Parameter count for `full` config: 447,780

### Modern Modular Architecture (`CODES/baselines/`)
- Separated into modular files per model
- Model naming uses underscores: `simple_cnn`, `resnet18_hsi`, `mlp_mixer_hsi`
- Factory pattern via `get_baseline_model()` in `factory.py`
- Backward compatibility alias at `factory.py:576`: `ABLATION_CONFIGS = DMLPFFN_ABLATION_CONFIGS`
- Created in commit 5e9c0a6 (Phase 5, 2026-11-21)

## Ablation Configuration Counts

| Implementation | Config Count | Location |
|----------------|--------------|----------|
| Historical | 11 configs | `_archive/CODES/baseline_models.py` |
| Modern | 23 configs | `CODES/baselines/factory.py` |

**Key Distinction:** The 4 additional full-model variants (`full_dmlpffn`, `full_resnet_style`, `full_with_attention`, `full_enhanced`) were introduced in Phase 5 (commit 5e9c0a6) and are NOT part of the historical implementation.

## Data Split Manifest (Frozen)

The data was split with seed=42 using stratified group-wise splitting:

```json
{
  "train_samples": 3205,  // Classes: Healthy(1316), Early(756), Mid(1512), Late(5670)
  "val_samples": 1694,    // Classes: Healthy(196), Mid(182), Late(1316)
  "test_samples": 1330    // Classes: Healthy(574), Late(756)
}
```

## Frozen Scientific Artifacts (SHA-256 Hashes)

| File | SHA-256 Hash |
|------|--------------|
| `artifacts/best_model.pth` | `0b81f04de14fb73b587a198e04918308b13047c37b2a190289e90fe38edf5903` |
| `artifacts/dmlpffn_test_logits_frozen.json` | `537626f567b3b6eb2b65b7fe51fea429e53bd7d7c4257df0b9be538b0e17689d` |
| `artifacts/split_manifest_frozen.json` | `b7b341e22a6f9fd06661777c0c0ac6ce6641ff8f96ed418fd33472c4de64f255` |
| `artifacts/preprocessing_artifacts_sklearn.joblib` | (computed on demand) |
| `artifacts/preprocessing_artifacts_arrays.npz` | (computed on demand) |

## Metadata Reference

The file `artifacts/frozen/baseline_models_signatures.json` contains metadata about baseline configurations:

- `ablations_historical.json`: 11 configurations from `_archive/CODES/baseline_models.py` (source-available, not physically exported)
- `ablations_modern.json`: 23 configurations from `CODES/baselines/factory.py` (Phase 5, commit 5e9c0a6)
- `export_source: "archive_snapshot"`: Documents where the metadata was derived from; does NOT indicate a physical snapshot file exists

## Reproducing Phase 5 Results

### Step 1: Activate Environment

```bash
cd DMLPFFN-TomatoLeafSpot-Detection-main
conda activate dmlpffn_env  # or your preferred environment
```

### Step 2: Verify Dependencies

```bash
pip install torch torchvision scikit-learn numpy thop
```

### Step 3: Run Phase 5 Experiment

```python
python CODES/phase5_baseline_ablation.py
```

### Step 4: Expected Outputs

The experiment produces:
- `artifacts/baselines/*/best_model.pth` - Best model weights per seed
- `artifacts/baselines/*/history.json` - Training history
- `artifacts/baselines/*/test_results.json` - Test evaluation results
- `phase5_results_YYYYMMDD_HHMMSS.json` - Complete aggregated results

## Baseline Models

| Model | Module | Config Key | Parameters |
|-------|--------|------------|------------|
| SimpleCNN | `CODES.baselines.simple_cnn` | `simple_cnn` | ~86K |
| ResNet-18 HSI | `CODES.baselines.resnet18_hsi` | `resnet18_hsi` | ~11M |
| MLP-Mixer HSI | `CODES.baselines.mlp_mixer_hsi` | `mlp_mixer_hsi` | ~2.3M |
| SVM (RBF) | `CODES.baselines.classical_ml` | `svm` | CPU |
| Random Forest | `CODES.baselines.classical_ml` | `random_forest` | CPU |
| DMLPFFN (Full) | `CODES.dmlpffn_disease` | `dmlpffn_phase3` | ~448K |

## DMLPFFN Ablation Study Configurations

The `factory.py` provides 23 ablation configurations via `DMLPFFN_ABLATION_CONFIGS`:

### Branch Ablations (3 configs)
- `branch_ablation_only_low` - Only low branch, disable mid and high
- `branch_ablation_only_mid` - Only mid branch, disable low and high
- `branch_ablation_only_high` - Only high branch, disable low and mid

### Two Branch Ablations (3 configs)
- `two_branch_no_low` - Disable low branch only
- `two_branch_no_mid` - Disable mid branch only
- `two_branch_no_high` - Disable high branch only

### Perceptron Ablations (6 configs)
- `perc_ablation_no_global` - Disable global perceptron in blocks
- `perc_ablation_no_partition` - Disable partition perceptron in blocks
- `perc_ablation_no_local` - Disable local perceptron in blocks
- `perc_ablation_only_global` - Only global perceptron enabled
- `perc_ablation_only_partition` - Only partition perceptron enabled
- `perc_ablation_only_local` - Only local perceptron enabled

### Width Ablations (2 configs)
- `width_ablation_narrow_32ch` - Narrow configuration (8/16/32 channels)
- `width_ablation_wide_64ch` - Wide configuration (32/64/128 channels)

### Fusion Mode Ablations (1 config)
- `fusion_concat` - Concat fusion instead of additive

### Dilation Rate Ablations (3 configs)
- `dilation_137` - Dilation rates [1, 3, 7] instead of [1, 2, 5]
- `dilation_15` - Dilation rates [1, 5]
- `dilation_1_only` - Single dilation rate [1]

### Full Model Variants (5 configs) - **Modern additions (Phase 5)**
- `full` - Default reference: all branches, all perceptrons, default width
- `full_dmlpffn` - DMLPFFN full architecture variant
- `full_resnet_style` - ResNet-style downsampling configuration
- `full_with_attention` - Full architecture with attention mechanism
- `full_enhanced` - Enhanced activation configuration

**Total: 23 configurations** (3 branch ablations + 3 two-branch ablations + 6 perceptron ablations + 2 width ablations + 1 fusion mode + 3 dilation variants + 5 full model variants)

## Usage Examples

### Loading Historical Baseline Models (Source Only)

```python
from _archive.CODES.baseline_models import get_baseline_model

model = get_baseline_model('simplecnn', in_channels=128, num_classes=4)
```

### Loading Modern Baseline Models

```python
from CODES.baselines.factory import get_baseline_model, get_ablation_model

# Standard baseline
model = get_baseline_model('simple_cnn', in_channels=128, num_classes=4)

# DMLPFFN ablation
model = get_ablation_model('full', in_channels=128, num_classes=4, dropout=0.7)
```

### Accessing Frozen Artifacts

```python
import torch
import json

# Load frozen baseline models
state_dict = torch.load('artifacts/best_model.pth')

# Load signatures metadata (describes configs, not actual checkpoints)
with open('artifacts/frozen/baseline_models_signatures.json', 'r') as f:
    signatures = json.load(f)
```

## Conditional Binary Evaluation

The model uses 4-class output but evaluates conditionally on binary task:

- **Classes**: 0=Healthy, 1=Early, 2=Mid, 3=Late
- **Binary Evaluation**: Healthy (0) vs Late (3)
- **Test Set Condition**: Only contains Healthy (0) and Late (3) classes

## Statistical Tests

Results include:
- Bootstrap confidence intervals (1000 resamples, 95% confidence)
- McNemar's test for pairwise comparison (group-aware, α=0.05)

## Expected Results (Phase 3 Reference)

| Metric | Value |
|--------|-------|
| Val Accuracy | 78.75% |
| Test Accuracy | 95.64% |
| Macro F1 | 0.9660 |
| Weighted F1 | 0.9658 |
| Parameters | 447,780 |
| Train/Val Gap | 11.08% |

---

For questions or issues, refer to `PHASE5_RESULTS_VERIFICATION.md` and `PHASE5_BASELINE_RESULTS.md`.