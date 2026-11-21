---
name: artifact-index
description: Comprehensive inventory of all checkpoints, artifacts, and figures for the DMLPFFN project.
metadata:
  type: reference
---

# Artifact Index — DMLPFFN Tomato Leaf Disease Classification

This document provides a complete inventory of the frozen artifacts, model checkpoints, and results generated during the Phase 3 (DMLPFFN) and Phase 5 (Baselines) experiments.

## Core Pipeline Artifacts

These files are essential for reproducing the DMLPFFN results and running inference.

| Artifact | Path | Description |
|----------|------|-------------|
| **Split Manifest** | `artifacts/split_manifest_frozen.json` | Frozen image/patch/group split (seed=42). |
| **Model Checkpoint** | `artifacts/best_model.pth` | Best DMLPFFN model weights (95.64% test accuracy). |
| **Test Logits** | `artifacts/dmlpffn_test_logits_frozen.json` | 4-class logits for the 1,330 test patches. |
| **PCA & Scaler** | `artifacts/preprocessing_artifacts_sklearn.joblib` | Fitted PCA (128 components) and StandardScaler. |
| **Train Arrays** | `artifacts/preprocessing_artifacts_arrays.npz` | Preprocessed training data arrays (X_train, y_train). |

## Baseline Model Artifacts (Phase 5)

Comprehensive results for 5 baseline architectures across 5 seeds.

### Deep Learning Baselines
Each directory contains `seed_*/{best_model.pth, history.json, test_results.json}`.

| Model | Path | Parameters | Mean Test Accuracy |
|-------|------|------------|--------------------|
| **SimpleCNN** | `artifacts/baselines/simple_cnn/` | 86,156 | 52.92% ± 7.30% |
| **MLP-Mixer HSI** | `artifacts/baselines/mlp_mixer_hsi/` | 2,310,812 | 46.39% ± 11.73% |
| **ResNet-18 HSI** | `artifacts/baselines/resnet18_hsi/` | 11,242,884 | 45.67% ± 5.63% |

### Classical ML Baselines
| Model | Path | Description |
|-------|------|-------------|
| **SVM (RBF)** | `artifacts/baselines/svm/` | `model.joblib` + aggregate metrics. |
| **Random Forest** | `artifacts/baselines/random_forest/` | 500 trees, `model.joblib`. |

## Statistical Analysis & Reports

| File | Path | Description |
|------|------|-------------|
| **Full Report** | `CLEANUP_REPORT.md` | Audit trail and validation of cleanup changes. |
| **Stats (Acc)** | `artifacts/statistical_analysis_accuracy.json` | Group-aware bootstrap and McNemar tests for Accuracy. |
| **Stats (F1)** | `artifacts/statistical_analysis_macro_f1.json` | Group-aware bootstrap for Macro F1. |
| **Registry** | `artifacts/baselines/process_registry.json` | Execution log for baseline experiments. |

## Figure Generation Data

- **Inference Results**: `artifacts/experiment_results_recovered.json`
- **Baseline Aggregates**: `artifacts/baselines/*/aggregate_results.json`
- **Frozen Split Stats**: Contained within `artifacts/split_manifest_frozen.json`

---
**Note**: All spectral data processing follows the water-band removal protocol (SWIR 94-115) and sensor-specific PCA dimensionality reduction as documented in `data/README.md`.
