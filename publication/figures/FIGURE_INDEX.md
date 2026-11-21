# Figure Index — DMLPFFN Manuscript

All figures generated as **SVG** (vector, publication-ready) and **PDF** in `publication/figures/`.

---

## Figure 1: Overall Research Workflow (6 Stages)
**File**: `figure1_workflow.svg` / `figure1_workflow.pdf`

**Description**: Six-stage horizontal workflow diagram showing the complete research pipeline from dataset acquisition through conditional binary evaluation.

**Stages**:
1. **Dataset Acquisition** — USDA Hyperspectral Tomato Leaf Dataset, 2 sensors (VIS-NIR 300 bands + SWIR 168 bands), DAI 2-16 post-inoculation
2. **Preprocessing Pipeline** — Water band removal (SWIR 94-115), Sensor-specific IncrementalPCA (128), Train-only StandardScaler, 9×9 patches stride 4
3. **Group-Wise Data Split** — Biological unit grouping (run number / leaf), zero spatial leakage, Train/Val/Test 75.4/13.8/10.8%, frozen seed=42 manifest
4. **DMLPFFN Architecture** — 3-branch Dual MLP (Global/Partition/Local with dilated convolutions), element-wise fusion, 447,780 params
5. **Training & Regularization** — AdamW lr=2e-4 wd=1e-4, WeightedRandomSampler, Dropout=0.7, LabelSmooth=0.1, MixUp(0.2)+CutMix(1.0), CosineAnnealingWarmRestarts, EarlyStop(patience=25)
6. **Conditional Binary Evaluation** — P(Healthy\|H/L)=e^z0/(e^z0+e^z3), P(Late\|H/L)=e^z3/(e^z0+e^z3), exact marginalization from 4-class logits, group-aware bootstrap CI, McNemar's test

**Source Artifacts**: None (conceptual diagram based on manuscript methodology)

---

## Figure 2: Preprocessing Pipeline (Leakage-Free)
**File**: `figure2_preprocessing.svg` / `figure2_preprocessing.pdf`

**Description**: Detailed preprocessing flowchart showing the leakage-free pipeline with critical callout box.

**Pipeline Steps**:
1. Raw Hyperspectral Cubes (VIS-NIR 300 + SWIR 168)
2. Water Band Removal (SWIR bands 94-115 removed)
3. Group-Wise Split by biological unit (BEFORE patch extraction)
4. Sensor-Specific PCA (IncrementalPCA n=128, fitted on TRAIN only)
5. Train-Only Scaling (StandardScaler fitted on TRAIN only)
6. Patch Extraction (9×9, stride=4, AFTER split & scaling)
7. Class Balancing (WeightedRandomSampler with class weights [1.76, 3.06, 1.53, 0.41])
8. Data Augmentation (MixUp α=0.2 50%, CutMix α=1.0 50%, Label Smoothing 0.1)
9. Final Input Tensor [Batch, 128, 9, 9]

**Critical Callout**: Leakage prevention measures documented

**Source Artifacts**: 
- `artifacts/preprocessing_artifacts/` (PCA models, scalers)
- `artifacts/split_manifest_frozen.json` (frozen group-wise split)

---

## Figure 3: DMLPFFN Architecture (from CODES/dmlpffn_disease.py)
**File**: `figure3_architecture.svg` / `figure3_architecture.pdf`

**Description**: Complete architectural diagram reconstructed from actual source code showing all three branches and internal DMLPBlock structure.

**Architecture Details**:
- **Input**: [Batch, 128, 9, 9] NCHW format
- **Low Branch**: Conv2d(128→16, 3×3) → BN+ReLU → DMLPBlock(16→16) → Conv2d(16→64, 1×1)
- **Mid Branch**: Conv2d(128→32, 5×5) → BN+ReLU → DMLPBlock(32→32) → Conv2d(32→64, 1×1)
- **High Branch**: DMLPBlock(128→64) → Conv2d(64→64, 1×1)
- **DMLPBlock Internals**:
  - GlobalPerceptron: Dual-branch global pooling (spatial mean + avgpool) → FC layers
  - PartitionPerceptron: Grouped Conv2d(128→C, 3×3, groups=4)
  - LocalPerceptron: 3 dilated convolutions (d=1,2,5) summed
- **Fusion**: Element-wise sum (Low + Mid + High, 64 ch each) → GAP → Dropout(0.7) → Linear(64→4)
- **Total Parameters**: 447,780

**Source Artifacts**: `CODES/dmlpffn_disease.py` (lines 1-500+)

---

## Figure 4: Training Curves (Corrected Pipeline, Seed=42)
**File**: `figure4_training_curves.svg` / `figure4_training_curves.pdf`

**Description**: Four-panel training dynamics visualization.

**Panels**:
1. **Accuracy Curves** — Train (45 epochs) vs Val (46 epochs), best val at epoch 21 (0.7875), train-val gap 11.56%
2. **Loss Curves** — Train loss (45 epochs) vs Val loss (44 epochs), both decreasing then val plateauing
3. **Learning Rate Schedule** — CosineAnnealingWarmRestarts (T₀=10), log scale
4. **Training Summary Table** — Key metrics in text box

**Key Metrics**:
- Final Training Accuracy: 0.9031 (90.31%)
- Best Validation Accuracy: 0.7875 (78.75%) at Epoch 21
- Train-Val Gap: 0.1156 (11.56%)
- Test Accuracy (4-class): 0.9564 (95.64%)
- Total Epochs: 45 (Early stopping at epoch 46)
- Model Parameters: 447,780
- Training Time: 3,558 sec

**Source Artifacts**: `artifacts/experiment_results_recovered.json` (history object with train_acc, val_acc, train_loss, val_loss, lr arrays)

---

## Figure 5: Conditional Binary Confusion Matrix (Frozen Manifest, 1330 patches)
**File**: `figure5_confusion_matrix.svg` / `figure5_confusion_matrix.pdf`

**Description**: Two-panel confusion matrix comparison — 4-class and conditional binary (Healthy vs Late).

**Left Panel — 4-Class Confusion Matrix**:
- True labels: Healthy(0)=574, Asympt(1)=0, Early(2)=0, Late(3)=756
- Predictions from 4-class argmax
- Note: Asymptomatic & Early have 0 test samples

**Right Panel — Conditional Binary (Exact Marginalization from Logits)**:
| | Pred Healthy | Pred Late |
|---|---|---|
| **True Healthy** | TN=542 | FP=32 |
| **True Late** | FN=1 | TP=755 |

**Conditional Binary Metrics (from logits)**:
- Accuracy: 0.9977
- Macro F1: 0.9977
- Weighted F1: 0.9977
- AUC-ROC: 0.9997
- Healthy: Precision=0.9982, Recall=0.9941
- Late: Precision=0.9958, Recall=0.9987

**Formula**: P(Healthy\|H/L) = exp(z₀)/(exp(z₀)+exp(z₃)), P(Late\|H/L) = exp(z₃)/(exp(z₀)+exp(z₃))

**Source Artifacts**: `artifacts/dmlpffn_test_logits_frozen.json` (logits array, confusion_matrix_conditional_binary_logits, conditional_binary_metrics)

---

## Figure 6: Baseline Comparison (Single-Run vs 5-Seed)
**File**: `figure6_baseline_comparison.svg` / `figure6_baseline_comparison.pdf`

**Description**: Four-panel baseline comparison distinguishing 5-seed DL baselines from single-run classical baselines.

**Models Compared**:
| Model | Params | Run Type | Test Acc | Macro F1 | Weighted F1 | AUC-ROC |
|-------|--------|----------|----------|----------|-------------|---------|
| **DMLPFFN (ref)** | 447,780 | Single Run | **0.9977** | **0.9977** | **0.9977** | **0.9997** |
| SimpleCNN | 86,156 | 5 Seeds | 0.5292 ± 0.0730 | 0.4026 ± 0.0606 | 0.4233 ± 0.0888 | 0.4758 ± 0.0464 |
| MLP-Mixer HSI | 2,310,812 | 5 Seeds | 0.4639 ± 0.1173 | 0.3988 ± 0.0844 | 0.4109 ± 0.0900 | 0.4758 ± 0.0464 |
| ResNet-18 HSI | 11,242,884 | 5 Seeds | 0.4567 ± 0.0563 | 0.3284 ± 0.0276 | 0.3088 ± 0.0585 | 0.4758 ± 0.0464 |
| SVM (RBF) | N/A | Single Run | 0.5263 | 0.3284 | 0.3088 | N/A |
| Random Forest | N/A | Single Run | 0.5263 | 0.3284 | 0.3088 | N/A |

**Visual Encoding**:
- **Blue solid**: DMLPFFN (reference, single run)
- **Teal hatched (///)**: 5-seed DL baselines (mean ± std shown)
- **Red hatched (\\\)**: Single-run classical baselines (SVM, RF)

**Panels**:
1. Test Accuracy bar chart
2. Macro F1 bar chart
3. Weighted F1 bar chart
4. Parameter Efficiency scatter (log-scale params vs accuracy)

**Statistical Notes**:
- 5-seed DL baselines: seeds [42, 123, 456, 789, 999]
- Group-aware McNemar's test performed (4 test groups = low power)
- McNemar results: all p > 0.05 (no significant pairwise differences among baselines)

**Source Artifacts**:
- `artifacts/baselines/simple_cnn/aggregate_results.json`
- `artifacts/baselines/mlp_mixer_hsi/aggregate_results.json`
- `artifacts/baselines/resnet18_hsi/aggregate_results.json`
- `artifacts/baselines/svm/aggregate_results.json`
- `artifacts/baselines/random_forest/aggregate_results.json`
- `artifacts/dmlpffn_test_logits_frozen.json` (DMLPFFN reference)
- `artifacts/baselines/PHASE5_BASELINE_RESULTS.md` (summary report)

---

## Regeneration Instructions

To regenerate all figures:
```bash
cd DMLPFFN-TomatoLeafSpot-Detection-main
python generate_figures.py
```

**Requirements**: Python 3.10+, matplotlib, numpy, scikit-learn

**Output**: All figures saved as both SVG (editable vector) and PDF (publication-ready) in `publication/figures/`.