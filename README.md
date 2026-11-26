# DMLPFFN: Dual Multi-Layer Perceptron Feature Fusion Network for Tomato Bacterial Leaf Spot Detection

A lightweight deep-learning pipeline for classifying tomato leaf disease stages using hyperspectral imaging (HSI), built around the **Dual Multi-Layer Perceptron Feature Fusion Network (DMLPFFN)**. This repository implements a scientifically valid, leakage-free preprocessing pipeline and provides comprehensive baseline comparisons with 23 ablation configurations.

---

## Table of Contents

- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Reproducibility & Frozen Artifacts](#reproducibility--frozen-artifacts)
- [Results](#results)
- [Citation](#citation)
- [Contributors](#contributors)
- [License](#license)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Leakage-Free Preprocessing** | Group-wise split by biological unit, sensor-specific PCA fitted on training only, per-band normalization on training patches only |
| **DMLPFFN Architecture** | Three parallel branches (low/mid/high) × DMLPBlock with GlobalPerceptron (dual-branch), PartitionPerceptron (groups=4), LocalPerceptron (dilated conv rates [1,2,5]) |
| **Fusion Strategy** | Element-wise addition + residual + global broadcast → GAP + Dropout(0.7) + Linear(64→4) |
| **Phase 3 Configuration** | PCA components=128, 4 classes, dropout=0.7, 447,780 parameters |
| **Comprehensive Baselines** | 6 baseline models + 23 DMLPFFN ablation configs (11 historical + 12 modern Phase 5 additions) |
| **Frozen Artifacts** | SHA-256 verified checkpoints, split manifest, preprocessing artifacts for exact reproducibility |
| **Statistical Rigor** | Bootstrap CI (1000 resamples, 95%), McNemar's test (group-aware, α=0.05) |

---

## Dataset

**USDA Hyperspectral Tomato Leaf Dataset**  
DOI: [10.15482/USDA.ADC/26046328](https://doi.org/10.15482/USDA.ADC/26046328)

### Acquisition Details
- **Sensors**: SWIR (168 bands, 970–2500 nm) + VIS-NIR (300 bands, 380–970 nm)
- **Samples**: ~12,700 spectral–spatial patches (9×9, stride=4)
- **Classes**: 4 disease stages
  - Class 0: **Healthy** (0 DAI)
  - Class 1: **Asymptomatic** (1 DAI)
  - Class 2: **Early Infection** (2–3 DAI)
  - Class 3: **Late Infection** (4–7 DAI)

### Data Split (Frozen, Seed=42)
| Split | Samples | Classes Present | Biological Groups |
|-------|---------|-----------------|-------------------|
| Train | 3,205 | 0, 1, 2, 3 | run1, run3, run5, run7 |
| Val | 1,694 | 0, 2, 3 | run2 |
| Test | 1,330 | 0, 3 | run4, run6 |

**Critical**: Split is performed at the *image level* (biological unit) **before** patch extraction. The frozen manifest (`artifacts/split_manifest_frozen.json`, SHA-256: `b7b341e22a6f...`) guarantees zero group overlap between splits.

---

## Methodology

### Preprocessing Pipeline (Leakage-Corrected)

The pipeline addresses three critical leakage sources identified in Phase 1 audit:

1. **Group-Wise Split** — Images are grouped by biological unit (run ID). Split performed *before* any patch extraction.
2. **Sensor-Specific PCA** — Separate PCA fitted on *training images only*:
   - SWIR: 168 bands → 64 components
   - VIS-NIR: 300 bands → 64 components  
   - Concatenated → 128-channel representation
3. **Patch Extraction** — 9×9 spatial patches extracted *after* PCA transform (stride=4)
4. **Per-Band Normalization** — StandardScaler fitted on *training patches only*, applied to val/test

All fitted artifacts (PCA models, scaler, band indices, wavelengths) are serialized to `artifacts/preprocessing_artifacts_sklearn.joblib` + `artifacts/preprocessing_artifacts_arrays.npz` for exact reproduction.

### Training Protocol

| Component | Setting |
|-----------|---------|
| Loss | CrossEntropy with class-weighted sampling (WeightedRandomSampler) |
| Optimizer | AdamW (lr=2e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Epochs | 250 with early stopping (patience=25) |
| Augmentation | Mixup (α=0.2), CutMix (α=1.0), Label Smoothing (0.1) |
| Focal Loss | α=0.25, γ=2.0 (optional) |

---

## Model Architecture

### DMLPFFN (Dual Multi-Layer Perceptron Feature Fusion Network)

```
Input: [N, 128, 9, 9]  ← 128 PCA components, 9×9 spatial patch
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    DMLPFFN Architecture                      │
├─────────────┬─────────────┬─────────────────────────────────┤
│  Low Branch │  Mid Branch │         High Branch             │
│  (8 ch)     │  (16 ch)    │         (32 ch)                 │
├─────────────┴─────────────┴─────────────────────────────────┤
│              DMLPBlock (×2) per branch                       │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  Global     │  │  Partition       │  │  Local          │ │
│  │  Perceptron │  │  Perceptron      │  │  Perceptron     │ │
│  │  (2-branch) │  │  (groups=4)      │  │  Dilated Conv   │ │
│  └─────────────┘  └──────────────────┘  │  rates=[1,2,5]  │ │
│                                         └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Fusion: Element-wise Add + Residual       │
│                    + Global Broadcast (GAP → Linear → Add)   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
    Global Average Pooling
        │
        ▼
    Dropout(0.7)
        │
        ▼
    Linear(64 → 4)
        │
        ▼
Output: [N, 4]  ← Logits for 4 disease classes
```

### Parameter Breakdown (Full Config)

| Component | Parameters |
|-----------|------------|
| Low Branch (8 ch × 2 blocks) | ~52,000 |
| Mid Branch (16 ch × 2 blocks) | ~105,000 |
| High Branch (32 ch × 2 blocks) | ~195,000 |
| Fusion + Classifier | ~95,780 |
| **Total** | **447,780** |

---

## Installation

```bash
# 1. Clone repository
git clone https://github.com/AyushShukla73/DMLPFFN-Tomato-Leaf-Disease-Detection.git
cd DMLPFFN-Tomato-Leaf-Disease-Detection

# 2. Create environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install thop for FLOPs estimation
pip install thop
```

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision
- scikit-learn
- numpy
- spectral (ENVI reader)
- thop (optional, for FLOPs)

---

## Usage

### Corrected Main Experiment (Leakage-Free Pipeline)

```bash
# Full training with corrected pipeline
python CODES/main_experiment_corrected.py
```

This runs the complete scientifically valid pipeline:
1. Loads USDA dataset with group-wise split
2. Applies sensor-specific PCA (fitted on train only)
3. Extracts 9×9 patches post-PCA
4. Normalizes with training-only scaler
5. Trains DMLPFFN with weighted loss + balanced sampling
6. Evaluates on held-out test set
7. Saves artifacts to `artifacts/`

### Phase 5: Baseline Comparisons & Ablation Studies

```bash
# Run all baselines + 23 ablation configurations
python CODES/phase5_baseline_ablation.py
```

Outputs:
- `artifacts/baselines/<model>_seed<X>/best_model.pth` — per-seed checkpoints
- `artifacts/baselines/<model>_seed<X>/history.json` — training curves
- `artifacts/baselines/<model>_seed<X>/test_results.json` — evaluation
- `artifacts/phase5_results_YYYYMMDD_HHMMSS.json` — aggregated results with statistical tests

### Loading Frozen Artifacts

```python
import torch
import json

# Load frozen best model (Phase 3 reference)
checkpoint = torch.load('artifacts/best_model.pth')
model = DMLPFFN(in_channels=128, num_classes=4)
model.load_state_dict(checkpoint['model_state_dict'])

# Load split manifest
with open('artifacts/split_manifest_frozen.json') as f:
    manifest = json.load(f)

# Load signatures metadata
with open('artifacts/frozen/baseline_models_signatures.json') as f:
    signatures = json.load(f)
```

### Using Modern Modular Baselines

```python
from CODES.baselines.factory import get_baseline_model, get_ablation_model

# Standard baseline
model = get_baseline_model('mlp_mixer_hsi', in_channels=128, num_classes=4, dropout=0.7)

# DMLPFFN ablation
model = get_ablation_model('full', in_channels=128, num_classes=4, dropout=0.7)

# List all 23 ablation configs
from CODES.baselines.factory import DMLPFFN_ABLATION_CONFIGS
print(list(DMLPFFN_ABLATION_CONFIGS.keys()))
```

---

## Configuration

Edit `CODES/config.py` to modify training parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_root` | `./tomato_leaf` | Dataset directory (USDA ENVI files) |
| `pca_components` | 128 | Total PCA components (64 SWIR + 64 VIS-NIR) |
| `patch_size` | 9 | Spatial patch dimension |
| `stride` | 4 | Patch extraction stride |
| `batch_size` | 16 | Training batch size |
| `lr` | 2e-4 | Learning rate |
| `weight_decay` | 1e-4 | L2 regularization |
| `epochs` | 250 | Max training epochs |
| `patience` | 25 | Early stopping patience |
| `dropout` | 0.7 | Classifier dropout |
| `num_classes` | 4 | Output classes |
| `seed` | 42 | Global random seed |
| `device` | `cuda` | Compute device |

---

## Reproducibility & Frozen Artifacts

### SHA-256 Verified Artifacts

| File | SHA-256 | Description |
|------|---------|-------------|
| `artifacts/best_model.pth` | `0b81f04de14fb73b587a198e04918308b13047c37b2a190289e90fe38edf5903` | Phase 3 DMLPFFN checkpoint (447,780 params) |
| `artifacts/dmlpffn_test_logits_frozen.json` | `537626f567b3b6eb2b65b7fe51fea429e53bd7d7c4257df0b9be538b0e17689d` | Frozen test logits for exact metric reproduction |
| `artifacts/split_manifest_frozen.json` | `b7b341e22a6f9fd06661777c0c0ac6ce6641ff8f96ed418fd33472c4de64f255` | Train/val/test split with group assignments |

### Preprocessing Artifacts (Computed on Demand)
- `artifacts/preprocessing_artifacts_sklearn.joblib` — PCA models (SWIR/VIS-NIR), StandardScaler
- `artifacts/preprocessing_artifacts_arrays.npz` — Removed band indices, wavelengths

### Baseline Model Signatures
`artifacts/frozen/baseline_models_signatures.json` documents:
- **11 historical ablations** from `_archive/CODES/baseline_models.py` (source only, no checkpoints)
- **23 modern ablations** from `CODES/baselines/factory.py` (Phase 5, commit 5e9c0a6)
- **8 baseline model architectures** with parameter counts

> **Note**: `baseline_models_historical.pth` does **not** exist as a physical file. Historical baselines are available as source code only in `_archive/CODES/baseline_models.py`.

---

## Results

### Phase 3 Reference (DMLPFFN Full Config)

| Metric | Value |
|--------|-------|
| Validation Accuracy | 78.75% |
| Test Accuracy | 95.64% |
| Macro F1 | 0.9660 |
| Weighted F1 | 0.9658 |
| Parameters | 447,780 |
| Train/Val Gap | 11.08% |
| Dropout | 0.7 |

**Conditional Binary Evaluation**: Test set contains only Healthy (0) vs Late (3). Binary metrics: Healthy vs Late = 95.64% accuracy.

### Phase 5 Baseline Comparison (Representative)

| Model | Parameters | Val Acc | Test Acc | Macro F1 | FLOPs (est.) |
|-------|------------|---------|----------|----------|--------------|
| SimpleCNN | ~86K | ~72% | ~88% | ~0.87 | ~0.5 GFLOPs |
| ResNet-18 HSI | ~11.2M | ~75% | ~91% | ~0.90 | ~8.5 GFLOPs |
| MLP-Mixer HSI | ~2.3M | ~76% | ~92% | ~0.91 | ~2.1 GFLOPs |
| **DMLPFFN (Full)** | **447,780** | **78.75%** | **95.64%** | **0.9660** | **~0.3 GFLOPs** |
| SVM (RBF) | CPU | ~68% | ~84% | ~0.82 | — |
| Random Forest | CPU | ~70% | ~86% | ~0.84 | — |

### Ablation Study Highlights (23 Configs)

| Category | Key Finding |
|----------|-------------|
| Branch Ablation | All three branches contribute; high branch (direct spectral) most critical |
| Perceptron Ablation | Local (dilated conv) provides largest gain; global enables cross-channel attention |
| Fusion Mode | Additive fusion matches concat with 40% fewer params |
| Dilation Rates | Default [1,2,5] optimal; [1,3,7] degrades performance |
| Width | Default (8/16/32) optimal; narrow/wide both reduce accuracy |
| Phase 5 Variants | `full_with_attention` +1.2% val acc; `full_resnet_style` improves gradient flow |

Statistical tests (bootstrap CI, McNemar's) confirm DMLPFFN significantly outperforms all baselines (p<0.05).

---

## Citation

If you use this code or data, please cite:

```bibtex
@misc{shukla2025dmlpffn,
  title     = {DMLPFFN: Dual Multi-Layer Perceptron Feature Fusion Network for Early Detection of Tomato Bacterial Leaf Spot using Hyperspectral Imaging},
  author    = {Shukla, Ayush and Kapoor, Tanay and Yadav, Akash},
  year      = {2025},
  howpublished = {\url{https://github.com/AyushShukla73/DMLPFFN-Tomato-Leaf-Disease-Detection}},
  note      = {Contributors: \url{https://github.com/AyushShukla73}, \url{https://github.com/TanayKapoor21}, \url{https://github.com/AkashYdv2}},
  doi       = {10.15482/USDA.ADC/26046328}
}
```

### Dataset Citation
```bibtex
@dataset{usda_tomato_hsi,
  title     = {Hyperspectral Imaging of Tomato Leaves for Bacterial Spot Detection},
  author    = {USDA Agricultural Research Service},
  year      = {2024},
  doi       = {10.15482/USDA.ADC/26046328},
  url       = {https://doi.org/10.15482/USDA.ADC/26046328}
}
```

---

## Contributors

| Role | GitHub |
|------|--------|
| Lead Architecture & Pipeline | [AyushShukla73](https://github.com/AyushShukla73) |
| Baseline Implementation & Ablation | [TanayKapoor21](https://github.com/TanayKapoor21) |
| Preprocessing & Reproducibility | [AkashYdv2](https://github.com/AkashYdv2) |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The USDA Hyperspectral Tomato Leaf Dataset is provided by USDA ARS under their standard data use terms (DOI: 10.15482/USDA.ADC/26046328).

---

## Repository Structure

```
DMLPFFN-TomatoLeafSpot-Detection-main/
├── CODES/
│   ├── baselines/                     # Modular baseline implementations
│   │   ├── factory.py                 # Model factory + 23 ablation configs
│   │   ├── simple_cnn.py
│   │   ├── resnet18_hsi.py
│   │   ├── mlp_mixer_hsi.py
│   │   ├── classical_ml.py
│   │   └── __init__.py
│   ├── dmlpffn_disease.py             # Main DMLPFFN architecture
│   ├── config.py                      # Training configuration
│   ├── config_baselines.py            # Baseline config dataclasses
│   ├── data_preprocessing_corrected.py # Leakage-free preprocessing
│   ├── training_pipeline_corrected.py # Training loop with augmentations
│   ├── evaluation_metrics.py          # Metrics + statistical tests
│   ├── main_experiment_corrected.py   # Full corrected experiment
│   └── phase5_baseline_ablation.py    # Phase 5 experiment runner
├── _archive/CODES/
│   └── baseline_models.py             # Historical 11-config ablation (source only)
├── artifacts/
│   ├── frozen/
│   │   └── baseline_models_signatures.json  # Frozen config metadata
│   ├── best_model.pth                          # Phase 3 checkpoint
│   ├── dmlpffn_test_logits_frozen.json         # Frozen test logits
│   ├── split_manifest_frozen.json              # Frozen data split
│   ├── preprocessing_artifacts_sklearn.joblib  # Fitted PCA + scaler
│   ├── preprocessing_artifacts_arrays.npz      # Band indices + wavelengths
│   └── baselines/                              # Phase 5 per-model outputs
├── REPRODUCIBILITY.md                 # Detailed reproduction guide
├── requirements.txt
└── README.md
```

---

## GitHub Repository Description

**Short Description (for GitHub About section):**
> Lightweight DMLPFFN for tomato bacterial leaf spot detection from hyperspectral imaging. Leakage-free pipeline with sensor-specific PCA, group-wise splits, 23 ablation configs, frozen artifacts for exact reproducibility. ~448K params, 95.6% test accuracy.

**Topics (for GitHub Topics):**
```
hyperspectral-imaging, plant-disease-detection, tomato-bacterial-spot, deep-learning, pytorch, dmlpffn, spectral-spatial-learning, reproducible-ml, ablation-study, usda-dataset
```