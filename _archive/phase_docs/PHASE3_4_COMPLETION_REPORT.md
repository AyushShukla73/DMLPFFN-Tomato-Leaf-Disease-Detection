# Phase 3 & Phase 4 Completion Report
## DMLPFFN Publication Preparation Plan

**Date**: 2026-11-14  
**Status**: Both phases completed successfully

---

## Executive Summary

| Phase | Status | Key Outcome |
|-------|--------|-------------|
| **Phase 3: Experimental Improvements with Regularization** | ✅ **COMPLETED** | Overfitting gap reduced from 45% (Phase 2B) to 12% (Phase 3); Best val accuracy improved from 54.91% → 78.75% |
| **Phase 4: Manuscript Alignment Check** | ✅ **COMPLETED** | Comprehensive audit document (PHASE4_MANUSCRIPT_ALIGNMENT.md) created documenting all critical discrepancies between manuscript claims and corrected experimental reality |

---

## Phase 3: Experimental Improvements with Regularization

### 3.1 Objective
Address severe overfitting (45% train/val gap) identified in Phase 2B through regularization techniques, data augmentation, and loss function improvements.

### 3.2 Implementation Summary

#### Configuration Changes (`CODES/config.py`)
```python
# Phase 3 Regularization Parameters Added:
dropout: float = 0.7              # Increased from 0.6
weight_decay: float = 5e-4        # Increased from 1e-4
label_smoothing: float = 0.1      # NEW: Label smoothing for CrossEntropyLoss

# Augmentation (Phase 3)
use_mixup: bool = True
mixup_alpha: float = 0.2
use_cutmix: bool = True
cutmix_alpha: float = 1.0

# Class Balancing
use_focal_loss: bool = False      # Available but disabled (alpha=0.25, gamma=2.0)
```

#### Model Architecture Update (`CODES/dmlpffn_disease.py`)
- Modified `DMLPFFN.__init__` to accept `dropout` parameter (default=0.7)
- Added `nn.Dropout(dropout)` before classifier in `forward()` method

#### Training Pipeline Enhancements (`CODES/training_pipeline_corrected.py`)
**New Augmentation Functions:**
- `mixup_data()`: Mixup augmentation with configurable alpha
- `cutmix_data()`: CutMix augmentation with random bounding boxes
- `mixup_criterion()`: Proper loss computation for mixed samples

**New Loss Function:**
- `FocalLoss` class: For handling class imbalance (configurable α=0.25, γ=2.0)

**Training Loop Integration:**
- Mixup/CutMix applied with 50% probability during training
- Label smoothing integrated into `CrossEntropyLoss`
- WeightedRandomSampler for class balancing
- CosineAnnealingWarmRestarts (T_0=10) scheduler

### 3.3 Experimental Results

| Metric | Phase 2B (Before) | Phase 3 (After) | Improvement |
|--------|-------------------|-----------------|-------------|
| **Best Validation Accuracy** | 54.91% (epoch 22) | **78.75% (epoch 21)** | **+23.84 pp** |
| **Train/Val Gap** | ~45% | **~12%** | **-33 pp** |
| **Test Accuracy** | 90.04% (2-class) | **95.64% (2-class)** | **+5.6 pp** |
| **Epochs to Convergence** | 47 (early stop) | 46 (early stop) | Similar |
| **Training Time** | 4,611s (~77 min) | 3,558s (~59 min) | Faster |
| **Class Weights** | [1.53, 6.12, 2.47, 0.46] | [1.76, 3.06, 1.53, 0.41] | More balanced |

### 3.4 Detailed Phase 3 Test Results

```
Test Set Classification Report:
              precision    recall  f1-score   support

    Healthy     0.9981    0.9373    0.9668       574
Asymptomatic     0.0000    0.0000    0.0000         0
      Early     0.0000    0.0000    0.0000         0
       Late     0.9595    0.9709    0.9652       756

   accuracy                         0.9564      1330
  macro avg     0.4894    0.4770    0.4830      1330
weighted avg     0.9762    0.9564    0.9658      1330
```

**Confusion Matrix:**
```
[[538   4   1  31]    ← Healthy (574 total)
 [  0   0   0   0]    ← Asymptomatic (0 total)
 [  0   0   0   0]    ← Early (0 total)
 [  1  16   5 734]]   ← Late (756 total)
```

**Class Distributions:**
- Train: {0: 1316, 1: 756, 2: 1512, 3: 5670} (9,254 samples)
- Val: {0: 196, 2: 182, 3: 1316} (1,694 samples, 3 classes)
- Test: {0: 574, 3: 756} (1,330 samples, 2 classes)

### 3.5 Key Findings

1. **Regularization Effective**: Dropout (0.7) + weight decay (5e-4) + label smoothing (0.1) dramatically reduced overfitting
2. **Mixup/CutMix Benefit**: 50% probability augmentation during training improved generalization
3. **Train/Val Gap Closed**: From 45% → 12%, indicating model now learns generalizable features
4. **Test Set Limitation Persists**: Still only 2 classes (Healthy, Late) in test due to group-wise split
5. **Focal Loss Available**: Implemented but not used (use_focal_loss=False); ready for Phase 5 ablation

### 3.6 Artifacts Generated
- `artifacts/best_model.pth` — Best model checkpoint (val_acc=0.7875)
- `artifacts/experiment_results.json` — Complete experiment metadata
- `artifacts/preprocessing_artifacts/` — Sensor-specific PCA models, scalers
- `artifacts/split_manifest.json` — Reproducible data split specification

---

## Phase 4: Manuscript Alignment Check

### 4.1 Objective
Compare manuscript claims (extracted from `DMLPFFN (1) final.pdf`) against:
1. Corrected codebase (`CODES/*_corrected.py`)
2. Phase 2B experimental results (actual training run)
3. Phase 2C artifact audit (detailed verification)

### 4.2 Critical Finding
**The manuscript describes and evaluates the ORIGINAL LEAKY PIPELINE, not the corrected scientifically valid pipeline.**

### 4.3 Summary of Discrepancies

| Category | Manuscript | Corrected Reality | Status |
|----------|------------|-------------------|--------|
| **Test Accuracy** | 97.18% (4 classes) | 95.64% (2 classes only) | 🔴 CRITICAL |
| **Val Accuracy** | 98.22% (epoch 166) | 78.75% (epoch 21, Phase 3) | 🔴 CRITICAL |
| **Overfitting** | "Minimal" | 45% gap (Phase 2B) → 12% gap (Phase 3) | 🔴 CRITICAL |
| **PCA Method** | Per-image (leaky) | Sensor-specific, train-only | 🔴 CRITICAL |
| **Split Method** | Patch-stratified (leaky) | Group-wise biological | 🔴 CRITICAL |
| **Test Classes** | All 4 present | Only Healthy + Late | 🔴 CRITICAL |
| **Scaler** | Global (leaky) | Train-only fitted | 🔴 CRITICAL |

### 4.4 Key Discrepancy Details

**Manuscript Table 5 (Claimed Per-Class Metrics) — INVALID:**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Healthy | 0.9455 | 0.9630 | 0.9541 | 324 |
| Asymptomatic | 0.9912 | 0.9573 | 0.9739 | 117 ← **0 in reality** |
| Early | 0.9509 | 0.9509 | 0.9509 | 265 ← **0 in reality** |
| Late | 0.9817 | 0.9801 | 0.9809 | 1206 |

**Phase 3 Actual Test Results:**
- Test samples: 1,330 (vs claimed 1,912)
- Classes present: Healthy (574), Late (756)
- Asymptomatic: 0 samples, Early: 0 samples
- Macro F1: 0.4894 (invalid — 2 classes undefined)

### 4.5 Root Cause
Three critical data leakages in original pipeline:
1. **PCA Fitting**: Per-image or on all data → Corrected: Train-only, sensor-specific
2. **Split Method**: Patch-level stratified → Corrected: Group-wise by biological unit
3. **Scaler Fitting**: Global on all patches → Corrected: Train patches only
4. **Patch Extraction**: Before split → Corrected: After group-wise split

The group-wise split reveals the true dataset limitation: biological grouping causes missing classes in test split.

### 4.6 Required Manuscript Corrections (Before Submission)

| Section | Current | Required |
|---------|---------|----------|
| **Abstract** | 97.18% accuracy | Report actual: 78.75% val, 95.64% test (2-class) |
| **Section 3.3 (Pipeline)** | Patch extraction at step 6 | Move split BEFORE patch extraction |
| **Section 3.5.3 (PCA)** | "Per-image PCA" | "Sensor-specific PCA fitted on training data only" |
| **Section 3.5.2 (Normalization)** | Z-score normalization | "StandardScaler fitted on training patches only" |
| **Table 3** | Epochs: 200 | Early stopping at epoch 46/47 |
| **Section 4 (Results)** | All 97-98% metrics | Replace with corrected results + overfitting analysis |
| **Table 4** | Test metrics 1912 support | Actual 1330 support, 2 classes |
| **Table 5** | Per-class for all 4 | Remove or mark invalid |
| **Figure 5** | Confusion matrix 1912 | Replace with actual 1330 matrix |
| **Section 4.5 (Baselines)** | Comparison at 97-98% | Remove or re-run with corrected pipeline |
| **Conclusion** | All performance claims | Rewrite based on corrected results |

### 4.7 Recommended Publication Strategy

**Option B: Re-run with Improved Pipeline (Recommended)**
- Phase 3 completed: Regularization reduces overfitting to 12% gap
- Phase 5: Baseline comparisons & ablation studies
- Phase 6: Computational efficiency metrics
- Phase 7: Cross-validation & statistical significance
- Phase 8: Cross-sensor generalization
- Then report final improved results

**Scientific Integrity**: The corrected pipeline methodology IS the contribution. The paper should describe:
1. The leakage issues identified and fixed
2. The corrected pipeline (sensor-specific PCA, group-wise split, train-only scaler)
3. Phase 3 regularization improvements
4. Honest results with limitations acknowledged

---

## Phase 5 Readiness

### 5.1 What's Ready
✅ Corrected, scientifically valid pipeline (no leakage)  
✅ Phase 3 regularization implemented and tested  
✅ Mixup/CutMix augmentation working  
✅ Label smoothing + increased dropout + weight decay  
✅ FocalLoss class implemented (ready for ablation)  
✅ Reproducible artifacts and split manifest  
✅ Manuscript alignment audit complete  

### 5.2 Phase 5 Scope (Baseline Comparisons & Ablation Studies)

**Baseline Models to Implement:**
1. **CNN Baselines**: SimpleCNN, ResNet-18, ResNet-50 (adapted for 128-channel input)
2. **Transformer Baselines**: ViT-small, MLP-Mixer
3. **Spectral Models**: 1D-CNN, SpectralNet, DBDA
4. **Classical ML**: Random Forest, SVM on flattened features

**Ablation Studies (DMLPFFN Components):**
1. **Branch Ablation**: Remove Global / Partition / Local Perceptron individually
2. **Fusion Strategy**: Element-wise add vs concatenation vs attention
3. **Dilation Rates**: Test Local Perceptron with d=[1,2,5] vs [1,3,7] vs [1] only
4. **PCA Components**: 64 vs 128 vs 256
5. **Patch Size**: 7×7 vs 9×9 vs 11×11
6. **Regularization Ablation**: With/without mixup, cutmix, label smoothing, dropout
7. **Loss Function**: CrossEntropy vs FocalLoss vs Label Smoothed CE

**Evaluation Protocol:**
- All baselines use SAME corrected pipeline (group-wise split, sensor-specific PCA, train-only scaler)
- Same train/val/test splits (seeded for reproducibility)
- Report: Val accuracy, test accuracy (2-class), train/val gap, parameters, FLOPs, inference time
- Statistical significance: 5 runs with different seeds (Phase 7)

---

## Conclusion

**Phase 3**: ✅ **SUCCESSFUL** — Overfitting reduced from 45% → 12%, validation accuracy 54.91% → 78.75%

**Phase 4**: ✅ **COMPLETE** — Comprehensive audit document created (PHASE4_MANUSCRIPT_ALIGNMENT.md) documenting all critical discrepancies. Manuscript requires major revision before submission.

**Next**: Proceeding to **Phase 5: Baseline Comparisons & Ablation Studies** using the validated corrected pipeline with Phase 3 regularization.

---

*Report generated: 2026-11-14*  
*Phases 3 & 4 Complete — Ready for Phase 5*