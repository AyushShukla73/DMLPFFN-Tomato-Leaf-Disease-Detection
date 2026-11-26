# Hyperspectral Tomato Disease Classification using DMLPFFN

This project implements a scientifically validated deep-learning pipeline for classifying tomato leaf disease stages using hyperspectral imaging (HSI). It is built around a Dual Multi-Layer Perceptron Feature Fusion Network (**DMLPFFN**) trained on spectral–spatial patches extracted from USDA tomato hyperspectral cubes.

## Key Scientific Contributions

- **Fixed Critical Data Leakage Issues**: Per-image PCA leakage, patch-level split leakage, and normalization leakage
- **Sensor-Specific PCA**: Separate PCA for 168-band SWIR and 300-band VIS-NIR sensors
- **Group-Wise Split**: Stratified 70/15/15 split by biological unit (not random patch split)
- **Frozen Artifacts**: SHA-256 verified manifests and preprocessing artifacts for exact reproducibility
- **Conditional Binary Evaluation**: Exact marginalization from 4-class logits for Healthy vs Late infection

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Dataset

The project uses the USDA Hyperspectral Tomato Leaf Dataset, containing hyperspectral cubes captured across multiple days after inoculation (DAI). Four refined classes are used:

- **Healthy** (0 DAI)
- **Asymptomatic** (1 DAI)
- **Early Infection** (2–3 DAI)
- **Late Infection** (4–7 DAI)

Dataset DOI:  
https://doi.org/10.15482/USDA.ADC/26046328

~12,700 total samples extracted as 9×9 spectral-spatial patches with stride 4.

---

## Features

- **Complete preprocessing pipeline** (scientifically validated)
  - Water-band removal & spectral cleaning (1350-1460nm, 1800-1950nm)
  - Sensor-specific PCA harmonization to **128 components** (fitted on training only)
  - Normalization with scaler fitted on training patches only
- **Patch extraction**
  - 9×9 spectral–spatial patches
  - Stride = 4
  - ~12,700 total samples
- **Stratified dataset split** (70/15/15) using group-wise split by biological unit
- **Frozen split manifest** (`artifacts/split_manifest_frozen.json`) with verified no overlap between train/val/test groups
- **DMLPFFN architecture**
  - Multi-branch *MLP-based* feature fusion (low, mid, high branches)
  - Local Perceptron with dilated operations [1, 2, 5] for extended spatial perception
  - Global Perceptron with SE attention
  - Partition Perceptron with grouped convolutions
  - Element-wise fusion + global average pooling + classifier
- **Class imbalance handling**
  - Weighted loss
  - Balanced sampling
  - Mixup/CutMix augmentation
- **Robust training**
  - Cosine Annealing Warm Restarts
  - Early stopping (patience=25)
  - Label smoothing (0.1)
- **Evaluation**
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix
  - Conditional binary metrics (Healthy vs Late, marginalized from 4-class)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AyushShukla73/DMLPFFN-Tomato-Leaf-Disease-Detection.git
cd DMLPFFN-Tomato-Leaf-Disease-Detection
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Full Training (Corrected Pipeline)

```bash
cd CODES
python main_experiment_corrected.py
```

The corrected pipeline will:
- Load and preprocess hyperspectral cubes with **no data leakage**
- Apply sensor-specific PCA → patch extraction → normalization (all fitted on training only)
- Perform group-wise stratified dataset splitting by biological unit
- Train DMLPFFN with weighted loss, balanced sampling, and Mixup/CutMix
- Save logs, metrics, confusion matrix, and frozen artifacts
- Print final test performance with conditional binary evaluation

### Inference with Frozen Artifacts

```bash
python inference_dmlpffn_frozen.py
```

Uses SHA-256 verified frozen artifacts for exact reproducibility.

### Phase 5 Ablation Study

```bash
cd CODES
python phase5_baseline_ablation.py
```

Runs 23 ablation configurations (3 architectures × 7 configs + dilations + attention variants).

### Smoke Test (Quick Verification)

```bash
cd CODES
python smoke_test.py
```

Runs a 5-epoch quick test (~78% expected accuracy) to verify pipeline integrity.

---

## Configuration

Modify `CODES/config.py` to adjust:

| Parameter | Description | Default |
|-----------|-------------|---------|
| data_root | Dataset directory | ./tomato_leaf/tomato_leaf |
| pca_components | PCA dimension | 128 |
| patch_size | Patch size | 9 |
| stride | Patch stride | 4 |
| batch_size | Training batch size | 16 |
| lr | Learning rate | 2e-4 |
| weight_decay | L2 regularization | 5e-4 |
| epochs | Number of training epochs | 250 |
| patience | Early stopping patience | 25 |
| num_classes | Number of classes | 4 |
| dropout | Dropout rate | 0.7 |

Additional training configs in `CODES/config_baselines.py` for ablation studies.

---

## Model Architecture

DMLPFFN is a Dual Multi-Layer Perceptron Feature Fusion Network designed for spectral-spatial hyperspectral learning.

- **Input**: [N, 128, 9, 9] (NCHW format: batch, channels=PCA_components, height, width)
- **Three parallel feature branches**:
  - **Low-branch** (16→64 channels): Local feature enhancement with dilated conv
  - **Mid-branch** (32→64 channels): Mid-range spectral–spatial learning
  - **High-branch** (64→64 channels): Direct MLP-based spectral modeling
- **Each branch contains**: DMLPBlock with three perceptrons:
  - **GlobalPerceptron**: Channel attention (SE block) + 4-layer MLP
  - **PartitionPerceptron**: Grouped convolution (groups=4) + MLP
  - **LocalPerceptron**: Dilated convolutions [1, 2, 5] + MLP
- **Fusion**: Element-wise aggregation of all three branches → Global Average Pooling → Dropout(0.7) → Linear(64→4)
- **Output**: 4-class classification (healthy → late disease)
- **Total Parameters**: 2,310,812

---

## Results

### Corrected Pipeline (Phase 3 Configuration)
- **Test Accuracy**: ~97%+
- **Test Macro F1**: ~96%+
- **Conditional Binary (Healthy vs Late)**: ~99%+ AUC

### Quick Test (5 epochs)
- **Accuracy**: ~78.4%

### Ablation Studies (Phase 5)
| Configuration | Parameters | Description |
|--------------|------------|-------------|
| full (DMLPFFN ablation) | 447,780 | 3 branches × (Global+Partition+Local) |
| no_global | 298,516 | Partition + Local only |
| no_partition | 353,172 | Global + Local only |
| no_local | 397,828 | Global + Partition only |
| global_only | 199,060 | Global only |
| partition_only | 253,716 | Partition only |
| local_only | 298,372 | Local only |
| fusion_concat | 596,548 | Concatenation fusion |
| dilation_137 | 447,780 | Dilated rates [1,3,7] |
| dilation_15 | 447,780 | Dilated rates [1,5] |
| dilation_1_only | 447,780 | Dilated rate [1] only |
| full_resnet_style | 447,780 | ResNet-style residual blocks |
| full_with_attention | 447,780 | Extra attention modules |
| full_enhanced | 447,780 | Enhanced capacity |

*See `CODES/baselines/factory.py` for complete registry of 23 configurations.*

### Historical Baselines (11 configs)
From `_archive/CODES/baseline_models.py`: 1D-CNN, ResNet-18, SpectralNet, HybridSN, MLP-Mixer, and 6 DMLPFFN variants.

---

## Reproducibility

All results are fully reproducible using frozen artifacts:

1. **Split Manifest**: `artifacts/split_manifest_frozen.json` - SHA-256 verified group assignments
2. **Preprocessing Artifacts**: `artifacts/preprocessing_artifacts_sklearn.joblib` - Fitted PCA, scalers, band indices
3. **Model Signatures**: `artifacts/frozen/baseline_models_signatures.json` - Parameter counts and configs for all 34 ablation configurations (11 historical + 23 modern)
4. **Frozen Logits**: `artifacts/dmlpffn_test_logits_frozen.json` - Exact test set predictions

See `REPRODUCIBILITY.md` for detailed verification procedures.

### Verification Commands

```bash
# Verify historical baselines (11 configs)
python -c "from _archive.CODES.baseline_models import get_baseline_model; print('OK')"

# Verify modern ablations (23 configs)
python -c "from CODES.baselines.factory import get_ablation_model; m = get_ablation_model('full', in_channels=128, num_classes=4, dropout=0.7); print(f'Params: {sum(p.numel() for p in m.parameters()):,}')"

# Run verification tasks
cd CODES
python -m tests.verification_tasks
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{shukla2025dmlpffn,
  title     = {DMLPFFN: Dual Multi-Layer Perceptron Feature Fusion Network for Early Detection of Tomato Bacterial Leaf Spot using Hyperspectral Imaging},
  author    = {Shukla, Ayush and Kapoor, Tanay and Yadav, Akash},
  year      = {2025},
  howpublished = {\url{https://github.com/AyushShukla73/DMLPFFN-Tomato-Leaf-Disease-Detection}},
  note      = {Project contributors: \url{https://github.com/AyushShukla73}, \url{https://github.com/TanayKapoor21}, \url{https://github.com/AkashYdv2}}
}
```

---

## Repository Structure

```
DMLPFFN-Tomato-Leaf-Disease-Detection/
├── CODES/
│   ├── baselines/           # Phase 5 ablation factory (23 configs)
│   ├── legacy/              # Legacy baseline implementations
│   ├── tests/               # Verification tasks
│   ├── config.py            # Main configuration
│   ├── config_baselines.py  # Ablation configurations
│   ├── dmlpffn_disease.py   # Original DMLPFFN (2.3M params)
│   ├── main_experiment_corrected.py  # Corrected training pipeline
│   ├── data_preprocessing_corrected.py  # No-leakage preprocessing
│   ├── training_pipeline_corrected.py   # Training loop
│   ├── phase5_baseline_ablation.py      # Phase 5 ablation runner
│   ├── run_baselines.py     # Historical baseline runner (11 configs)
│   ├── smoke_test.py        # Quick 5-epoch verification
│   └── evaluation_metrics.py
├── _archive/                # Archived code and documentation
│   ├── CODES/baseline_models.py    # 11 historical ablations
│   ├── phase_docs/                 # Phase 1-5 documentation
│   └── ...
├── artifacts/
│   ├── frozen/              # SHA-256 verified artifacts
│   ├── baselines/           # Baseline model checkpoints
│   ├── best_model.pth       # Trained DMLPFFN (1.8MB)
│   └── split_manifest_frozen.json
├── data/                    # Dataset (not in repo - download separately)
├── generate_figures.py      # Publication figure generation
├── inference_dmlpffn_frozen.py  # Reproducible inference
├── MANUSCRIPT_REVISED.md    # Updated manuscript
├── PHASE5_BASELINE_RESULTS.md   # Phase 5 results
├── PHASE5_RESULTS_VERIFICATION.md  # Verification report
├── REPRODUCIBILITY.md       # Reproducibility guide
├── requirements.txt
└── training_plots.png       # Training curves
```

---

## License

This project is released for research purposes. Please cite the paper if you use this code.