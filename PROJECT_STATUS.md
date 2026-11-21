# Project Status — DMLPFFN Tomato Leaf Disease Classification

**Last Updated**: 2026-11-17
**Status**: Phase 5 Complete — Baseline Pipeline Operational

---

## Current State

### Data Pipeline ✓
- **Preprocessing**: Leakage-free pipeline validated
  - Water band removal (SWIR 94-115)
  - Sensor-specific PCA (128 components, train-only)
  - StandardScaler (train-only)
  - Patch extraction (9×9, stride=4)
- **Split Manifest**: `artifacts/split_manifest_frozen.json`
  - Seed: 42, group-wise by biological unit
  - Train: 9,254 patches | 39 images | 18 groups
  - Val: 1,694 patches | 9 images | 4 groups
  - Test: 1,330 patches | 7 images | 4 groups

### Model ✓
- **DMLPFFN Architecture**: 447,780 parameters
  - 3-branch MLP feature fusion
  - Dilated convolutions for spatial context
  - Element-wise fusion + global average pooling

### Training ✓
- **Phase 3 Results**: 95.64% test accuracy
- **Training Configuration**: AdamW, CosineAnnealingWarmRestarts, early stopping
- **Regularization**: LabelSmoothing=0.1, MixUp(0.2)+CutMix(1.0), Dropout=0.7

### Baselines ✓
- **Phase 5 Complete**: All 5 baselines trained with 5 seeds
  - SimpleCNN (86K params)
  - MLP-Mixer HSI (2.3M params)
  - ResNet-18 HSI (11.2M params)
  - SVM (RBF kernel)
  - Random Forest (500 trees)
- **DMLPFFN Reference**: Single run, 95.64% test accuracy

### Evaluation ✓
- **Conditional Binary Protocol**: Exact marginalization from 4-class logits
- **P(Healthy|H/L)** = exp(z₀)/(exp(z₀)+exp(z₃))
- **Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Statistical Tests**: Group-aware bootstrap CI, McNemar's test

---

## File Integrity

| Category | Status | Path |
|----------|--------|------|
| Split Manifest | ✓ Valid | `artifacts/split_manifest_frozen.json` |
| Preprocessing Artifacts | ✓ Valid | `artifacts/preprocessing_artifacts_*.{joblib,npz}` |
| DMLPFFN Checkpoint | ✓ Valid | `artifacts/best_model.pth` |
| Test Logits | ✓ Valid | `artifacts/dmlpffn_test_logits_frozen.json` |
| Baseline Results | ✓ Valid | `artifacts/baselines/*/aggregate_results.json` |

---

## Next Actions

- [ ] Manuscript preparation
- [ ] Publication figure generation
- [ ] Experiment archive

---

## Quick Start

```bash
# Verify pipeline
python CODES/smoke_test.py

# Run inference
python inference_dmlpffn.py

# Generate figures
python generate_figures.py
```