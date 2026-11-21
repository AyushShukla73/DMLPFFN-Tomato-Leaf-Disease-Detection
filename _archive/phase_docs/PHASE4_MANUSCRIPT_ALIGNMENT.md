# Phase 4: Manuscript Alignment Audit
## Comparison of Manuscript Claims vs. Corrected Implementation & Experimental Reality

---

## Executive Summary

This audit compares the manuscript (extracted from `DMLPFFN (1) final.pdf`) against:
1. **Corrected codebase** (`CODES/*_corrected.py` files)
2. **Phase 2B experimental results** (actual training run)
3. **Phase 2C artifact audit** (detailed artifact verification)

**Overall Finding**: The manuscript reports **highly optimistic results** (97.18% test accuracy, 98.22% validation accuracy, all 4 classes well-represented) that **do not match** the scientifically valid corrected pipeline's actual results (90.04% test accuracy on 2-class subset, 54.91% validation accuracy, severe overfitting).

---

## 1. Critical Discrepancies

### 1.1 Test Set Evaluation — FUNDAMENTAL MISMATCH

| Aspect | Manuscript Claim | Experimental Reality (Phase 2C) | Severity |
|--------|-----------------|--------------------------------|----------|
| **Test Accuracy** | 0.9718 (97.18%) | 0.9004 (90.04%) — **but only on 2 classes** | 🔴 CRITICAL |
| **Test Samples** | 1,912 | 1,134 | 🔴 CRITICAL |
| **Test Classes Present** | All 4 (Healthy, Asymptomatic, Early, Late) | **Only 2** (Healthy: 378, Late: 756) | 🔴 CRITICAL |
| **Asymptomatic Support** | 117 (Table 5) | **0** — completely absent | 🔴 CRITICAL |
| **Early Support** | 265 (Table 5) | **0** — completely absent | 🔴 CRITICAL |
| **Weighted F1-Score** | 0.9718 | 0.9077 (on 2-class subset) | 🔴 CRITICAL |
| **Macro F1-Score** | 0.9650 | 0.4448 (invalid — 2 classes undefined) | 🔴 CRITICAL |

**Manuscript Table 5 (Per-Class Metrics) — INVALID**:
```
Class              Precision  Recall   F1-Score  Support
0 (Healthy)        0.9455     0.9630   0.9541    324
1 (Asymptomatic)   0.9912     0.9573   0.9739    117   ← 0 in reality
2 (Early)          0.9509     0.9509   0.9509    265   ← 0 in reality
3 (Late)           0.9817     0.9801   0.9809    1206
```

**Phase 2C Actual Test Confusion Matrix**:
```
[[271   2  22  83]    ← Healthy (378 total)
 [  0   0   0   0]    ← Asymptomatic (0 total)
 [  0   0   0   0]    ← Early (0 total)
 [  0   4   2 750]]   ← Late (756 total)
```

### 1.2 Validation Accuracy — FUNDAMENTAL MISMATCH

| Aspect | Manuscript Claim | Experimental Reality | Severity |
|--------|-----------------|---------------------|----------|
| **Best Val Accuracy** | 0.9822 (98.22%) at Epoch 166 | 0.5491 (54.91%) at Epoch 22 | 🔴 CRITICAL |
| **Train/Val Gap** | "Close alignment... minimal overfitting" | **45.06% gap** (99.97% vs 54.91%) | 🔴 CRITICAL |
| **Epochs Run** | 200 (final run) | 47 (early stopping triggered) | 🟠 HIGH |
| **Early Stopping** | Not mentioned as triggered | Triggered at epoch 47 (patience=25) | 🟠 HIGH |

### 1.3 Training Dynamics

| Aspect | Manuscript Claim | Experimental Reality | Severity |
|--------|-----------------|---------------------|----------|
| **Training Accuracy** | 0.9988 (99.88%) | ~99.97% | 🟡 MEDIUM |
| **Convergence Speed** | Fast convergence within 20 epochs | Fast convergence, but no generalization | 🟡 MEDIUM |
| **Overfitting Assessment** | "Minimal overfitting" | **Severe overfitting** — model memorizes training | 🔴 CRITICAL |

---

## 2. Preprocessing Pipeline Discrepancies

### 2.1 PCA Implementation

| Aspect | Manuscript Claim (Section 3.3, 3.5.3, Conclusion) | Corrected Code Reality | Severity |
|--------|---------------------------------------------------|------------------------|----------|
| **PCA Type** | "Per-image PCA reduction to 128 components" (Conclusion, p.15) | **Sensor-specific IncrementalPCA** fitted on **training data only** | 🔴 CRITICAL |
| **Leakage Prevention** | Not addressed (implies per-image = leakage) | **Train-only fitting** — no data leakage | 🟢 FIXED in code |
| **Water Bands** | "Drop water-absorption bands 1350–1460 nm and 1800–1950 nm" | Same — correctly implemented | 🟢 MATCH |
| **Components** | 128 for both sensors | 128 for both (harmonized) | 🟢 MATCH |

**Critical Finding**: The manuscript's Conclusion (p.15) states: *"standardizing different spectral dimensions (168–300 bands) through **per-image Principal Component Analysis (PCA)** reduction to 128 components."* This describes the **leaky original approach**, not the corrected pipeline.

### 2.2 Data Splitting Strategy

| Aspect | Manuscript Claim (Section 3.3, p.6-7) | Corrected Code Reality | Severity |
|--------|--------------------------------------|------------------------|----------|
| **Split Method** | "70/15/15 stratified train/validation/test ratio" | **Group-wise split by biological unit** (run number) | 🔴 CRITICAL |
| **Leakage Prevention** | Not addressed (patch-level stratified = leakage) | **Biological unit grouping** — zero overlap verified | 🟢 FIXED in code |
| **Split Ratios Achieved** | 70/15/15 | 75.4% / 13.8% / 10.8% (by images) | 🟡 MEDIUM |
| **Stratification** | Implied at patch level | At biological group level (label presence) | 🔴 CRITICAL |

**Critical Finding**: Manuscript Figure 2 (p.7) and Section 3.3 describe patch extraction **before** split (step 6), which is the **original leaky pipeline**. Corrected code extracts patches **after** group-wise split.

### 2.3 Normalization

| Aspect | Manuscript Claim | Corrected Code Reality | Severity |
|--------|-----------------|------------------------|----------|
| **Scaler Type** | "Z-score normalization across bands" | **StandardScaler fitted on training patches only** | 🟢 FIXED in code |
| **Leakage** | Not addressed | Train-only fitting prevents leakage | 🟢 FIXED in code |

### 2.4 Patch Extraction

| Aspect | Manuscript Claim | Corrected Code Reality | Severity |
|--------|-----------------|------------------------|----------|
| **Patch Size** | 9×9 with stride 4 | 9×9 with stride 4 | 🟢 MATCH |
| **Total Patches** | ~12,740 | 12,278 (actual after class filtering) | 🟡 MEDIUM |
| **Extraction Timing** | Step 6 in pipeline (before split implied) | **After group-wise split** | 🔴 CRITICAL |

---

## 3. Model Architecture Alignment

### 3.1 Architecture Specification

| Aspect | Manuscript (Tables 1, 2) | Corrected Code (`dmlpffn_disease.py`) | Status |
|--------|--------------------------|--------------------------------------|--------|
| **Input Tensor** | (128, 9, 9) | (128, 9, 9) | 🟢 MATCH |
| **Branches** | Global, Partition, Local (dilated) | GlobalPerceptron, PartitionPerceptron, LocalPerceptron | 🟢 MATCH |
| **Local Dilations** | 1, 2, 5 (Table 2) | 1, 2, 5 | 🟢 MATCH |
| **Fusion** | Element-wise addition | Element-wise addition | 🟢 MATCH |
| **Total Parameters** | Not explicitly stated | ~447,780 | 🟢 CONSISTENT |
| **Output Classes** | 4 | 4 | 🟢 MATCH |
| **Dropout** | Not in tables | 0.6 (config) | 🟡 MEDIUM |

### 3.2 Architecture Details — Minor Discrepancies

| Aspect | Manuscript Table 1 | Code Reality | Notes |
|--------|-------------------|--------------|-------|
| **Low Branch** | Conv2D 128→16, DMLPBlock, Conv2D 16→64 | Similar structure | 🟢 CLOSE |
| **Mid Branch** | Conv2D 128→32, DMLPBlock, Conv2D 32→64 | Similar structure | 🟢 CLOSE |
| **High Branch** | DMLPBlock 128→64, Conv2D 64→64 | Similar structure | 🟢 CLOSE |
| **Fusion Output** | 64 channels (element-wise add of 64, 32, 64?) | Code concatenates then projects | 🟡 CHECK NEEDED |

---

## 4. Training Configuration Alignment

| Hyperparameter | Manuscript Table 3 | Config (`config.py`) | Phase 2B Actual | Status |
|---------------|-------------------|---------------------|-----------------|--------|
| **Optimizer** | AdamW | AdamW | Adam (code uses AdamW but results say Adam) | 🟡 CHECK |
| **Loss** | Weighted CrossEntropyLoss | CrossEntropyLoss (config) + class weights | CrossEntropy with class weights | 🟢 MATCH |
| **Learning Rate** | 0.0002 | 2e-4 | 2e-4 | 🟢 MATCH |
| **Weight Decay** | 0.0001 | 1e-4 | 1e-4 | 🟢 MATCH |
| **Batch Size** | 16 | 16 | 16 | 🟢 MATCH |
| **Max Epochs** | 250 | 250 | 250 (early stop at 47) | 🟢 MATCH |
| **Epochs Used** | 200 | N/A | 47 (early stop) | 🔴 MISMATCH |
| **Scheduler** | Cosine Annealing Warm Restarts | CosineAnnealingWarmRestarts (T_0=10) | Same | 🟢 MATCH |
| **Early Stopping** | Not in table | Patience 25 | Triggered at epoch 47 | 🟢 MATCH |
| **Imbalance Handling** | Weighted Loss + WeightedRandomSampler | WeightedRandomSampler + compute_class_weight | Same | 🟢 MATCH |
| **Split** | 70/15/15 stratified | 0.7/0.15/0.15 group-wise | 75.4/13.8/10.8 | 🟡 DIFFERENT |

---

## 5. Results & Discussion Discrepancies

### 5.1 Training Curves (Figure 4)

| Aspect | Manuscript Claim | Experimental Reality | Severity |
|--------|-----------------|---------------------|----------|
| **Val Accuracy at Epoch 166** | 0.9822 | N/A (training stopped at epoch 47) | 🔴 CRITICAL |
| **Val Accuracy Peak** | 0.9822 (epoch 166) | 0.5491 (epoch 22) | 🔴 CRITICAL |
| **Train/Val Gap** | "Small difference... minimal overfitting" | **45% gap** — severe overfitting | 🔴 CRITICAL |
| **Loss Curves** | "Steady improvement" | Training loss → 0, Val loss plateaus high | 🔴 CRITICAL |

### 5.2 Test Set Evaluation (Section 4.2, Tables 4, 5)

| Metric | Manuscript Table 4 | Phase 2C Reality | Notes |
|--------|-------------------|-----------------|-------|
| **Accuracy** | 0.9718 | 0.9004 (2-class only) | 🔴 CRITICAL |
| **Macro Avg F1** | 0.9650 | 0.4448 (invalid) | 🔴 CRITICAL |
| **Weighted Avg F1** | 0.9718 | 0.9077 (2-class) | 🔴 CRITICAL |
| **Support** | 1912 | 1134 | 🔴 CRITICAL |

### 5.3 Confusion Matrix (Figure 5)

| Aspect | Manuscript Figure 5 | Phase 2C Actual | Severity |
|--------|--------------------|----------------|----------|
| **Late → Late** | 1182 correct | 750 correct | 🔴 CRITICAL |
| **Healthy → Healthy** | 312 correct | 271 correct | 🔴 CRITICAL |
| **Late → Healthy** | 17 cases | 4 cases | 🟡 DIFFERENT |
| **Early → Late** | 12 cases | N/A (no Early in test) | 🔴 CRITICAL |
| **Total Test** | 1912 | 1134 | 🔴 CRITICAL |

### 5.4 Per-Class Analysis (Section 4.3, Table 5)

**Manuscript claims all 4 classes have high precision/recall/F1** — **INVALID** because test set lacks Asymptomatic and Early classes entirely.

---

## 6. Comparison with Baselines (Section 4.5)

| Claim | Reality Check | Status |
|-------|---------------|--------|
| "Outperforms LSTM/DBO-BiLSTM" | Cannot verify — no baseline experiments run | 🟡 UNVERIFIED |
| "Outperforms RF/SVM" | Cannot verify — no baseline experiments run | 🟡 UNVERIFIED |
| "Outperforms ResNet/ResNet-50" | Cannot verify — no baseline experiments run | 🟡 UNVERIFIED |
| "Fewer parameters, lightweight" | True: ~447K params vs millions for ResNet | 🟢 TRUE |
| "97-98% accuracy" | **False** — actual corrected: 54.91% val, 90.04% test (2-class) | 🔴 FALSE |

---

## 7. Conclusion Section Discrepancies (Section 5)

| Manuscript Claim (p.15-16) | Reality | Severity |
|---------------------------|---------|----------|
| "per-image PCA reduction" | Corrected: sensor-specific PCA on train only | 🔴 CRITICAL |
| "validation accuracy of 0.9822" | Actual: 0.5491 | 🔴 CRITICAL |
| "test accuracy of 0.9718" | Actual: 0.9004 (2-class subset) | 🔴 CRITICAL |
| "weighted average F1-score of 0.9718" | Actual: 0.9077 (2-class) | 🔴 CRITICAL |
| "excellent generalization and stability" | Severe overfitting (45% gap) | 🔴 CRITICAL |
| "all four classes... high reliability" | 2 classes missing from test | 🔴 CRITICAL |

---

## 8. Root Cause Analysis

### Why Manuscript Results Don't Match Corrected Pipeline

| Factor | Manuscript Pipeline (Leaky) | Corrected Pipeline | Impact |
|--------|----------------------------|-------------------|--------|
| **PCA Fitting** | Per-image (or on all data) | Train-only, sensor-specific | Leakage inflates metrics |
| **Split Method** | Patch-level stratified | Group-wise by biological unit | Leakage from spatial correlation |
| **Scaler Fitting** | Global (all patches) | Train patches only | Leakage inflates metrics |
| **Patch Extraction** | Before split | After split | Leakage inflates metrics |
| **Test Set Classes** | All 4 present (stratified patches) | Only 2 present (group-wise) | Group distribution imbalance |

**The fundamental issue**: The manuscript describes and evaluates the **original leaky pipeline**, not the corrected scientifically valid pipeline. The corrected pipeline:
- Prevents all 3 leakages (PCA, split, scaler)
- Uses group-wise splitting which reveals the true dataset limitation (missing classes in test groups)
- Shows the model actually overfits severely when leakage is removed

---

## 9. Required Manuscript Corrections

### 9.1 Must-Fix Before Submission

| Section | Current Text | Required Correction |
|---------|-------------|---------------------|
| **Abstract (p.1)** | "accuracy of 0.9718" | Report actual corrected results: ~54.91% val, explain test limitation |
| **Section 3.3 (p.6-7)** | Pipeline steps 1-7 with patch extraction at step 6 | Revise to show group-wise split **before** patch extraction |
| **Section 3.5.3 (p.10-11)** | "PCA is applied separately to each cube" (implies per-image) | Clarify: **Sensor-specific PCA fitted on training data only** |
| **Section 3.5.2 (p.10)** | Z-score normalization | Specify: **StandardScaler fitted on training patches only** |
| **Table 3 (p.12)** | Epochs Used: 200 | Report actual: early stopping at epoch 47 |
| **Section 4 (p.12-14)** | All results (97-98% accuracy, minimal overfitting) | **Replace with corrected results**: 54.91% val, severe overfitting |
| **Table 4 (p.13)** | Test metrics with 1912 support | Report actual: 1134 support, 2 classes only |
| **Table 5 (p.14)** | Per-class metrics for all 4 classes | **Remove or mark as invalid** — test lacks 2 classes |
| **Figure 5 (p.14)** | Confusion matrix 1912 samples | Replace with actual 1134-sample matrix (2 classes) |
| **Section 4.5 (p.14-15)** | Baseline comparisons with 97-98% | Remove or re-run with corrected pipeline |
| **Conclusion (p.15-16)** | All performance claims | **Rewrite entirely** based on corrected results |

### 9.2 Scientific Integrity Requirements

1. **Acknowledge the leakage fixes** — The paper should describe the corrected pipeline, not the leaky one
2. **Report negative results honestly** — Severe overfitting (45% gap) and test set class deficiency are real findings
3. **Explain the class distribution issue** — The USDA dataset's biological grouping causes missing classes in test split
4. **Propose solutions** — Phase 3 work (regularization, augmentation, resplit strategies) should be framed as future work

---

## 10. Recommended Path Forward

### Option A: Report Corrected Results (Scientifically Honest)
- Rewrite Results section with actual corrected pipeline results
- Validation accuracy: 54.91%, Test: 90.04% (2-class), severe overfitting
- Frame as: "After fixing critical data leakage issues, we find the task is significantly harder..."
- Discuss overfitting causes and Phase 3 mitigation strategies

### Option B: Re-run with Improved Pipeline (Phase 3+)
- Implement Phase 3 regularization (dropout, weight decay, label smoothing, mixup)
- Implement Phase 6 stratified group K-fold cross-validation
- Re-train and report improved results
- This requires completing Phases 3, 5, 6, 7, 8 first

### Option C: Hybrid
- Report corrected pipeline methodology (scientifically valid)
- Report Phase 2B results as "initial corrected baseline"
- Present Phase 3+ improvements as the main contribution

---

## 11. Summary of Alignment Status

| Category | Aligned? | Notes |
|----------|----------|-------|
| **Dataset Description** | 🟢 Mostly | Class definitions, sensors, DAI mapping match |
| **Preprocessing Pipeline** | 🔴 No | Manuscript describes leaky pipeline; code fixes all 3 leakages |
| **Split Strategy** | 🔴 No | Manuscript: patch-stratified; Code: group-wise biological |
| **Model Architecture** | 🟢 Yes | 3-branch DMLPFFN with dilated Local Perceptron matches |
| **Training Config** | 🟡 Mostly | Hyperparams match; epochs used differs (200 vs 47 early stop) |
| **Results — Validation** | 🔴 No | Manuscript: 98.22%; Reality: 54.91% |
| **Results — Test** | 🔴 No | Manuscript: 97.18% (4 classes); Reality: 90.04% (2 classes) |
| **Results — Overfitting** | 🔴 No | Manuscript: minimal; Reality: severe (45% gap) |
| **Results — Per-Class** | 🔴 No | Manuscript: all 4 classes high; Reality: 2 classes missing |
| **Baseline Comparisons** | 🟡 Unverified | No baseline experiments in corrected pipeline |
| **Conclusion Claims** | 🔴 No | All performance claims invalid for corrected pipeline |

---

## 12. Action Items for Phase 4 Completion

1. ✅ **Document all discrepancies** (this document)
2. ⬜ **Create corrected results tables** for manuscript replacement
3. ⬜ **Draft revised methodology section** describing corrected pipeline
4. ⬜ **Prepare "Limitations" section** acknowledging overfitting and test set issues
5. ⬜ **Decide publication strategy** (Option A, B, or C above)
6. ⬜ **Proceed to Phase 3** (Experimental Improvements) to address overfitting

---

*Report generated: 2026-11-13*  
*Phase 4: Manuscript Alignment Audit Complete*  
*Next: Phase 3 — Experimental Improvements for Regularization & Robustness*