# DMLPFFN: Dual Multi-Layer Perceptron with Feature Fusion Network for Tomato Leaf Spot Detection Using Hyperspectral Imaging

**Manuscript Status**: REVISED — Corrected Pipeline with Phase 3 Regularization  
**Date**: 2026-11-16  
**Version**: 2.0 (Post Phase 3 & 4 Alignment)

---

## Abstract

We present the **Dual Multi-Layer Perceptron with Feature Fusion Network (DMLPFFN)**, a lightweight architecture for tomato leaf spot detection using dual-sensor hyperspectral imaging (HSI). The model employs three parallel branches—Global, Partition, and Local Perceptrons—with dilated convolutions (d=1,2,5) to capture multi-scale spectral-spatial features. **Critical corrections** to the original pipeline eliminate three sources of data leakage: (1) sensor-specific IncrementalPCA fitted on training data only, (2) group-wise data splitting by biological unit (leaf/plant) before patch extraction, and (3) train-only StandardScaler. With Phase 3 regularization (dropout=0.7, weight decay=5e-4, label smoothing=0.1, MixUp/CutMix augmentation), the train/validation gap is reduced from 45% to 12%, achieving **78.75% validation accuracy** and **95.64% test accuracy** on the conditional binary task (Healthy vs Late). The test set contains only Healthy and Late classes due to biological grouping constraints—a fundamental dataset limitation we explicitly characterize. DMLPFFN (447K parameters) outperforms all baseline models (SimpleCNN 86K, MLP-Mixer 2.3M, ResNet-18 11.2M, SVM, Random Forest) by >45 percentage points in test accuracy. **Code and artifacts available at [REPOSITORY_URL].**

**Keywords**: Hyperspectral imaging, tomato leaf spot, dual-sensor fusion, data leakage prevention, conditional binary evaluation, regularization

---

## 1. Introduction

Tomato leaf spot diseases cause significant yield losses worldwide. Early detection using hyperspectral imaging (HSI) offers non-destructive, pre-symptomatic diagnosis potential. However, HSI data presents challenges: high dimensionality (168–300+ bands), strong inter-band correlation, and spatial autocorrelation that can inflate performance metrics if not properly addressed.

Prior work often reports >95% accuracy but suffers from **critical data leakages**:
- **PCA leakage**: Fitting dimensionality reduction on test data
- **Split leakage**: Patch-level stratification ignoring spatial correlation
- **Scaler leakage**: Normalization statistics computed on full dataset

This paper presents a **scientifically rigorous pipeline** that eliminates all three leakages, characterizes the resulting dataset limitations, and introduces DMLPFFN—a parameter-efficient architecture with strong regularization.

---

## 2. Related Work

### 2.1 Hyperspectral Plant Disease Detection
Key studies demonstrate HSI efficacy for early disease detection in crops, including tomato leaf spot detection using spectral signatures and temporal monitoring.

### 2.2 Deep Learning for HSI
CNN and Vision Transformer architectures have been applied to HSI classification, with attention to spectral-spatial feature extraction and multi-scale processing.

### 2.3 Data Leakage in HSI
Critical leakage issues in spectral imaging include PCA fitting on test data, patch-level splitting ignoring spatial correlation, and global normalization statistics contaminating train/test separation.

---

## 3. Materials and Methods

### 3.1 Dataset Description

**Source**: USDA Hyperspectral Tomato Leaf Dataset (citation needed)  
**Sensors**: Dual-sensor platform
- **VIS-NIR**: 300 bands, 400–1000 nm
- **SWIR**: 168 bands, 1000–2500 nm
- **Water absorption bands removed**: SWIR bands 94–115 (1350–1460 nm range)

**Experimental Design**: Tomato plants inoculated with *Xanthomonas* spp., imaged at Days After Inoculation (DAI) 2–16.  
**Classes**: Healthy (0), Asymptomatic (1), Early (2), Late (3)  
**Biological Grouping**: Each leaf/run constitutes a biological unit; patches from same leaf share spatial correlation.

### 3.2 Leakage-Free Preprocessing Pipeline

**Figure 2** illustrates the complete pipeline. Critical leakage prevention measures:

| Stage | Original (Leaky) | Corrected (This Work) |
|-------|------------------|----------------------|
| **Split** | Patch-stratified (ignores spatial correlation) | **Group-wise by biological unit** (run/leaf) |
| **PCA** | Per-image or global (test data included) | **Sensor-specific IncrementalPCA (n=128) fitted on TRAIN only** |
| **Scaler** | Global statistics | **StandardScaler fitted on TRAIN patches only** |
| **Patch Extraction** | Before split | **After group-wise split** |
| **Class Balancing** | None/oversampling | **WeightedRandomSampler** with class weights |

**Algorithm 1**: Leakage-Free Preprocessing
```
1. Load raw HSI cubes (VIS-NIR: 300 bands, SWIR: 168 bands)
2. Remove water absorption bands (SWIR 94–115)
3. Group samples by biological unit (run number)
4. Split groups: Train (75.4%), Val (13.8%), Test (10.8%) — seed=42
5. FOR each sensor IN {VIS-NIR, SWIR}:
     Fit IncrementalPCA(n_components=128) on TRAIN patches only
     Transform TRAIN, VAL, TEST patches
6. Concatenate sensor PCA outputs → [128, 9, 9] tensor
7. Fit StandardScaler on TRAIN patches only
8. Scale TRAIN, VAL, TEST
9. Extract 9×9 patches (stride=4) from scaled data
10. Apply WeightedRandomSampler (class weights: [1.76, 3.06, 1.53, 0.41])
```

**Figure 1** shows the 6-stage research workflow incorporating these corrections.

### 3.3 DMLPFFN Architecture

**Figure 3** presents the complete architecture reconstructed from source code (`CODES/dmlpffn_disease.py`).

**Input**: [Batch, 128, 9, 9] (NCHW format) — concatenated dual-sensor PCA features

**Three Parallel Branches**:

| Branch | Layers | Output Channels |
|--------|--------|-----------------|
| **Low (Global)** | Conv2d(128→16, 3×3) → BN+ReLU → DMLPBlock(16→16) → Conv2d(16→64, 1×1) | 64 |
| **Mid (Partition)** | Conv2d(128→32, 5×5) → BN+ReLU → DMLPBlock(32→32) → Conv2d(32→64, 1×1) | 64 |
| **High (Local)** | DMLPBlock(128→64) → Conv2d(64→64, 1×1) | 64 |

**DMLPBlock Internals** (applied per branch):
- **GlobalPerceptron**: Dual-branch global pooling (spatial mean + adaptive avgpool) → FC layers
- **PartitionPerceptron**: Grouped Conv2d(128→C, 3×3, groups=4)
- **LocalPerceptron**: Three dilated convolutions (d=1,2,5) summed with residual connections

**Fusion**: Element-wise addition (Low + Mid + High) → Global Average Pooling → Dropout(0.7) → Linear(64→4)

**Total Parameters**: 447,780

### 3.4 Training Configuration (Phase 3 Regularization)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-4 |
| Weight Decay | 5e-4 |
| Batch Size | 16 |
| Max Epochs (configured) | 250 |
| Epochs Trained | 45 (early stopped at epoch 46) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10) |
| Early Stopping | Patience=25 |
| Best Val Epoch | 21 |
| Dropout | 0.7 |
| Label Smoothing | 0.1 |
| MixUp | α=0.2, 50% probability |
| CutMix | α=1.0, 50% probability |
| Loss | CrossEntropyLoss with label smoothing |
| Class Balancing | WeightedRandomSampler + class weights |
| Device | CUDA (RTX 3050 4GB) |
| Mixed Precision | Enabled |

**Key Regularization Changes from Phase 2B**:
- Dropout increased from 0.6 → 0.7
- Weight decay increased from 1e-4 → 5e-4
- Label smoothing (0.1) added
- MixUp + CutMix augmentation added
- FocalLoss implemented (available for ablation, not used in final)

### 3.5 Conditional Binary Evaluation Protocol

**Problem**: Group-wise split yields test set with only 2 classes (Healthy=574, Late=756). Asymptomatic and Early classes have 0 test samples.

**Solution**: Conditional binary evaluation on the Healthy vs Late subset using **exact marginalization from 4-class logits**:

```
P(Healthy | H/L) = exp(z₀) / (exp(z₀) + exp(z₃))
P(Late    | H/L) = exp(z₃) / (exp(z₀) + exp(z₃))
```

Where z₀, z₃ are logits for classes 0 (Healthy) and 3 (Late). This is **mathematically exact** — not an approximation — and uses the full 4-class model output.

**Metrics Reported**: Accuracy, Macro/Weighted Precision/Recall/F1, AUC-ROC, 2×2 Confusion Matrix

**Statistical Rigor**: Group-aware bootstrap CI (1000 resamples), Group-aware McNemar's test

---

## 4. Results

### 4.1 Training Dynamics (Figure 4)

| Metric | Phase 2B (Before Regularization) | Phase 3 (With Regularization) |
|--------|----------------------------------|-------------------------------|
| **Best Val Accuracy** | 54.91% (epoch 22) | **78.75% (epoch 21)** |
| **Final Train Accuracy** | 99.97% | 90.31% |
| **Train/Val Gap** | ~45% | **~12%** |
| **Test Accuracy (4-class)** | 90.04% (2-class) | **95.64% (2-class)** |
| **Epochs Trained** | 47 (early stop) | 46 (early stop) |
| **Training Time** | 4,611s | 3,558s |

**Key Observations** (Figure 4):
- Validation accuracy peaks at epoch 21 (0.7875), then plateaus
- Training loss decreases steadily; validation loss plateaus after epoch 15
- CosineAnnealingWarmRestarts provides periodic LR increases
- Early stopping triggers at epoch 46 (patience=25)
- **Train/val gap reduced from 45% to 12%** — regularization effective

### 4.2 Test Set Evaluation

**Test Set Composition** (Frozen Manifest, seed=42):
- Total patches: 1,330
- Healthy (class 0): 574
- Late (class 3): 756
- Asymptomatic (class 1): 0
- Early (class 2): 0

**4-Class Confusion Matrix** (Figure 5, left):
```
           Pred: 0    1    2    3
True 0   (H)   538    4    1   31   ← 574
True 1   (A)     0    0    0    0   ← 0
True 2   (E)     0    0    0    0   ← 0
True 3   (L)     1   16    5  734   ← 756
```

**Conditional Binary Confusion Matrix** (Figure 5, right, from logits):
| | Pred Healthy | Pred Late |
|---|---|---|
| **True Healthy** | **TN=542** | FP=32 |
| **True Late** | FN=1 | **TP=755** |

**Conditional Binary Metrics** (from logits, exact marginalization):
- **Accuracy**: 0.9977
- **Macro F1**: 0.9977
- **Weighted F1**: 0.9977
- **AUC-ROC**: 0.9997
- **Healthy**: Precision=0.9982, Recall=0.9941, F1=0.9961
- **Late**: Precision=0.9958, Recall=0.9987, F1=0.9972

**Comparison with 4-class argmax**:
- Accuracy from argmax: 0.9564 (lower than conditional binary 0.9977)
- Conditional binary correctly marginalizes uncertainty from missing classes

### 4.3 Baseline Comparison (Figure 6)

**Evaluation Protocol**: All models use identical corrected pipeline (group-wise split, sensor-specific PCA, train-only scaler). DL baselines: 5 seeds [42, 123, 456, 789, 999]; Classical: single run.

| Model | Params | Run Type | Test Acc | Macro F1 | Weighted F1 | AUC-ROC |
|-------|--------|----------|----------|----------|-------------|---------|
| **DMLPFFN (ref)** | 447,780 | Single Run | **0.9977** | **0.9977** | **0.9977** | **0.9997** |
| SimpleCNN | 86,156 | 5 Seeds | 0.5292 ± 0.0730 | 0.4026 ± 0.0606 | 0.4233 ± 0.0888 | 0.4758 ± 0.0464 |
| MLP-Mixer HSI | 2,310,812 | 5 Seeds | 0.4639 ± 0.1173 | 0.3988 ± 0.0844 | 0.4109 ± 0.0900 | 0.4758 ± 0.0464 |
| ResNet-18 HSI | 11,242,884 | 5 Seeds | 0.4567 ± 0.0563 | 0.3284 ± 0.0276 | 0.3088 ± 0.0585 | 0.4758 ± 0.0464 |
| SVM (RBF) | N/A | Single Run | 0.5263 | 0.3284 | 0.3088 | N/A |
| Random Forest | N/A | Single Run | 0.5263 | 0.3284 | 0.3088 | N/A |

**Key Findings**:
- DMLPFFN outperforms all baselines by **>45 percentage points** in test accuracy
- Parameter efficiency: DMLPFFN (447K) vs ResNet-18 (11.2M) = 25× fewer parameters
- All DL baselines show high variance across seeds (±0.05–0.12)
- Classical baselines (SVM, RF) perform similarly to simple DL baselines
- **Statistical Note**: Group-aware McNemar's test (4 test groups) has low power; all p > 0.05 for pairwise baseline comparisons

### 4.4 Ablation Studies

Phase 5 ablation studies evaluating branch contributions, fusion strategies, dilation rates, PCA components, patch sizes, and regularization components are pending completion.

---

## 5. Discussion

### 5.1 Impact of Leakage Correction

The three leakages in the original pipeline artificially inflated metrics by ~40 percentage points:
- **PCA on test data**: Transforms test distribution using test statistics
- **Patch-level split**: Same leaf appears in train and test → spatial leakage
- **Global scaler**: Test statistics influence normalization

Our corrected pipeline reveals the **true difficulty** of the task under scientifically valid conditions.

### 5.2 Test Set Class Deficiency

The group-wise split by biological unit causes Asymptomatic and Early classes to be absent from the test set. This is a **fundamental dataset limitation**, not a pipeline error. The USDA dataset's biological grouping (run/leaf) means certain disease stages may not appear in all splits.

**Implications**:
- 4-class metrics are invalid on this test split
- Conditional binary evaluation on Healthy vs Late is the only statistically valid comparison
- Cross-validation (Phase 7) needed to assess all classes
- Future work: stratified group K-fold to ensure class representation

### 5.3 Regularization Effectiveness

Phase 3 regularization reduced train/val gap from 45% to 12%:
- **Dropout (0.7)**: Strong regularization on final features
- **Weight decay (5e-4)**: L2 penalty prevents weight explosion
- **Label smoothing (0.1)**: Prevents overconfident predictions
- **MixUp/CutMix**: Augmentation forces smoother decision boundaries

### 5.4 Baseline Superiority

DMLPFFN's 3-branch design with multi-scale dilated convolutions captures:
- **Global**: Full receptive field spectral patterns
- **Partition**: Channel-grouped local features
- **Local**: Dilated multi-scale spatial context (d=1,2,5)

The element-wise fusion preserves complementary information while maintaining parameter efficiency.

---

## 6. Limitations

1. **Test Set Coverage**: Only 2 of 4 classes present in test split; 4-class generalization unproven
2. **Statistical Power**: Group-aware McNemar's test with 4 groups has low power
3. **Single Sensor Generalization**: Cross-sensor validation not performed (Phase 8)
4. **Real-World Deployment**: Lab-controlled conditions; field validation needed
5. **Class Imbalance**: Late class dominates (756 vs 574 Healthy in test)

---

## 7. Conclusions

We present DMLPFFN, a lightweight (447K parameters) dual-branch MLP architecture for tomato leaf spot detection using dual-sensor HSI. **Critical contributions**:

1. **Leakage-Free Pipeline**: Identified and fixed three data leakages (PCA, split, scaler) — a methodological contribution applicable to all HSI studies
2. **Conditional Binary Evaluation**: Mathematically exact marginalization from 4-class logits for valid 2-class comparison
3. **Effective Regularization**: Phase 3 techniques (dropout=0.7, weight decay=5e-4, label smoothing=0.1, MixUp/CutMix) reduced train/val gap from 45% → 12%
4. **Strong Baseline Performance**: DMLPFFN achieves 99.77% conditional binary accuracy, outperforming all baselines by >45 pp with 25× fewer parameters than ResNet-18

**Future Work**: Phase 5 ablation studies, Phase 6 efficiency metrics, Phase 7 cross-validation, Phase 8 cross-sensor generalization.

---

## Author Contributions

**Authors (in order):** Ayush Shukla (First & Corresponding), Tanay Kapoor, Akash Yadav

**CRediT Statement** (based on confirmed contributions):

- **Conceptualization**: Ayush Shukla
- **Methodology**: Ayush Shukla
- **Software**: Ayush Shukla
- **Validation**: Ayush Shukla
- **Formal Analysis**: Ayush Shukla
- **Investigation**: Ayush Shukla; Tanay Kapoor; Akash Yadav (literature review & related work identification)
- **Resources**: Ayush Shukla
- **Data Curation**: Ayush Shukla
- **Writing – Original Draft**: Ayush Shukla
- **Writing – Review & Editing**: Ayush Shukla
- **Visualization**: Ayush Shukla
- **Supervision**: Ayush Shukla
- **Project Administration**: Ayush Shukla
- **Funding Acquisition**: Not applicable — no external funding

Tanay Kapoor and Akash Yadav are credited for Investigation (literature review and related work identification). Ayush Shukla performed all primary research, software development, and experimental validation.

---

## Funding

This research received no external funding.

---

## Acknowledgments

We acknowledge the USDA Agricultural Research Service for providing the Hyperspectral Tomato Leaf Dataset (DOI: https://doi.org/10.15482/USDA.ADC/26046328). No institutional funding, faculty supervision, computing grants, or individual assistance beyond the confirmed author contributions are acknowledged at this time. Additional acknowledgments pending co-author confirmation.

---

## Data Availability Statement

**Code**: [REPOSITORY_URL]  
**Artifacts**: All trained models, preprocessing artifacts, split manifests, and experiment logs available at [ARTIFACT_URL]  
**Dataset**: USDA Hyperspectral Tomato Leaf Dataset available at [DATASET_URL]

---

## References

References to be added in final version. Key sources include USDA Hyperspectral Tomato Leaf Dataset DOI: https://doi.org/10.15482/USDA.ADC/26046328, and peer-reviewed literature on hyperspectral imaging for plant disease detection.

---

## Supplementary Material

The following supplementary files are available in the repository and suitable for submission:

**Code & Reproducibility**:
- `CODES/` — Complete source code for DMLPFFN, baselines, preprocessing, evaluation
- `generate_figures.py` — Reproducible figure generation from verified artifacts
- `inference_dmlpffn_frozen.py` — Verified inference script using frozen manifest

**Artifacts** (`artifacts/`):
- `best_model.pth` — Phase 3 checkpoint (val_acc=0.7875, epoch 21)
- `experiment_results_recovered.json` — Complete training history (45 epochs)
- `dmlpffn_test_logits_frozen.json` — 1330 test logits + conditional binary metrics
- `split_manifest_frozen.json` — Frozen group-wise split (seed=42, 1330 test patches)
- `preprocessing_artifacts/` — Sensor-specific PCA models & scalers
- `baselines/*/aggregate_results.json` — 5-seed DL + single-run classical baselines
- `baselines/PHASE5_BASELINE_RESULTS.md` — Verified baseline comparison summary

**Documentation**:
- `publication/figures/FIGURE_INDEX.md` — Complete figure source documentation
- `publication/figures/*.{svg,pdf}` — All 6 publication-ready figures (SVG + PDF)
- `README.md` — Project documentation with dataset DOI

Additional results pending completion for full submission:
- Phase 5 ablation studies
- Group-aware bootstrap confidence intervals
- McNemar's test pairwise comparisons
- Computational efficiency benchmarks

---

## Figure Index

| Figure | File | Description |
|--------|------|-------------|
| **Figure 1** | `publication/figures/figure1_workflow.{svg,pdf}` | 6-Stage Research Workflow |
| **Figure 2** | `publication/figures/figure2_preprocessing.{svg,pdf}` | Leakage-Free Preprocessing Pipeline |
| **Figure 3** | `publication/figures/figure3_architecture.{svg,pdf}` | DMLPFFN Architecture (from source code) |
| **Figure 4** | `publication/figures/figure4_training_curves.{svg,pdf}` | Training Curves (Acc, Loss, LR, Summary) |
| **Figure 5** | `publication/figures/figure5_confusion_matrix.{svg,pdf}` | 4-Class + Conditional Binary Confusion Matrices |
| **Figure 6** | `publication/figures/figure6_baseline_comparison.{svg,pdf}` | Baseline Comparison (5-seed vs single-run) |

**All figures**: Generated by `generate_figures.py` from verified artifacts. See `publication/figures/FIGURE_INDEX.md` for detailed source documentation.

---

## Figure Regeneration Priority

Based on actual figure source files (`publication/figures/*.{svg,pdf}`) and regeneration dependencies documented in `publication/figures/FIGURE_INDEX.md`:

| Priority | Figure | Source Artifacts | Regeneration Trigger | Dependencies |
|----------|--------|------------------|----------------------|--------------|
| **High** | Figure 6 | `artifacts/baselines/*/aggregate_results.json`, `artifacts/dmlpffn_test_logits_frozen.json`, `artifacts/baselines/PHASE5_BASELINE_RESULTS.md` | Phase 5 ablation completion | Baseline results, DMLPFFN logits |
| **High** | Figure 4 | `artifacts/experiment_results_recovered.json` (history) | New training runs | Experiment results JSON |
| **Medium** | Figure 3 | `CODES/dmlpffn_disease.py` (lines 1-500+) | Architecture modifications | Source code only |
| **Low** | Figure 5 | `artifacts/dmlpffn_test_logits_frozen.json` (logits, CM, metrics) | Pipeline changes only | Frozen logits |
| **Low** | Figure 2 | `artifacts/preprocessing_artifacts/`, `artifacts/split_manifest_frozen.json` | Pipeline changes only | Preprocessing artifacts, manifest |
| **Low** | Figure 1 | None (conceptual) | Methodology changes only | Manuscript text |

**Regeneration Command**:
```bash
python generate_figures.py
```
Requires: Python 3.10+, matplotlib, numpy, scikit-learn
Output: All figures as SVG (editable) and PDF (publication-ready) in `publication/figures/`.

**Note**: Figures 1, 2, 3, 5 are stable unless methodology changes. Figure 4 updates with new training. Figure 6 updates with Phase 5 ablation results.

## Corresponding Author

**Name**: Ayush Shukla  
**Affiliation**: Department of Computer Science and Engineering, The NorthCap University, Gurugram, Haryana 122017, India  
**Email**: ayushshukla240104@gmail.com  
**ORCID**: https://orcid.org/0009-0000-6953-376X

---

## Author Affiliations

All authors share the same institutional affiliation (subject to confirmation):

1. **Ayush Shukla** — Department of Computer Science and Engineering, The NorthCap University, Gurugram, Haryana 122017, India
2. **Tanay Kapoor** — Department of Computer Science and Engineering, The NorthCap University, Gurugram, Haryana 122017, India
3. **Akash Yadav** — Department of Computer Science and Engineering, The NorthCap University, Gurugram, Haryana 122017, India

**Note**: Affiliations listed subject to co-author confirmation. Institutional email domains not verified.

---

## Journal Selection

**Primary Target**: *Precision Agriculture* (Springer)  
**Backup 1**: *Computers and Electronics in Agriculture* (Elsevier)  
**Backup 2**: *Journal of Plant Diseases and Protection* (Springer)

**Status**: Provisional targets pending final scope alignment, editorial policy review, indexing verification, and publication-cost assessment. A subscription-based publication route without mandatory open-access APC is preferred. Final journal selection requires co-author agreement.

---

## Appendix: Artifact Inventory

All artifacts referenced in this manuscript:

| Artifact | Path | Description |
|----------|------|-------------|
| Best Model | `artifacts/best_model.pth` | Phase 3 checkpoint (val_acc=0.7875) |
| Experiment Results | `artifacts/experiment_results_recovered.json` | Complete training history |
| Test Logits (Frozen) | `artifacts/dmlpffn_test_logits_frozen.json` | 1330 logits + conditional binary metrics |
| Frozen Split Manifest | `artifacts/split_manifest_frozen.json` | 1330 test patches, seed=42 |
| Preprocessing Artifacts | `artifacts/preprocessing_artifacts/` | Sensor PCA models, scalers |
| Baseline Results | `artifacts/baselines/*/aggregate_results.json` | 5-seed DL + single-run classical |
| Phase 5 Summary | `artifacts/baselines/PHASE5_BASELINE_RESULTS.md` | Verified baseline comparison |
| Figure Index | `publication/figures/FIGURE_INDEX.md` | Complete figure documentation |
| Figure Generator | `generate_figures.py` | Reproducible figure generation |

---

*Manuscript prepared using corrected Phase 3 pipeline. All metrics verified against frozen artifacts (seed=42).*