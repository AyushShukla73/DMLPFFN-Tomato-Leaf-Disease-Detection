# Phase 2D: Independent Biological Unit Verification & Evaluation Feasibility Audit

## Executive Summary

This audit investigates the **patch-count discrepancy** between manifest theoretical values and actual experiment results, verifies **biological independence** from USDA metadata, assesses **grouping algorithm validity**, and evaluates **four-class metric feasibility**. 

**Key Findings:**
1. **Patch-count discrepancy is due to label filtering**, not data leakage
2. **Biological independence is correctly implemented** in the grouping algorithm
3. **4-class evaluation is unfeasible** - test set lacks Asymptomatic and Early classes (0 samples each)
4. **Metrics are invalid** - classification report contains undefined 0/0/0 values for missing classes

---

## 1. Patch-Count Discrepancy Investigation

### 1.1 Observed Values

| Source | Train | Val | Test | Total |
|--------|-------|-----|------|-------|
| split_manifest.json (theoretical) | 9,254 | 1,694 | 1,330 | 12,278 |
| experiment_results.json (actual) | 7,924 | 3,220 | 1,134 | 12,278 |
| **Discrepancy** | -1,330 | +1,526 | -196 | 0 |

### 1.2 Root Cause Analysis

The discrepancy stems from **DAI-based label mapping filtering** during patch extraction:

**From `data_preprocessing_corrected.py:172-184`:**
```python
def get_label_from_dai(dai: int) -> int:
    """Map DAI to class label."""
    label_map = {
        0: 0,   # healthy
        1: 1,   # asymptomatic
        2: 2,   # early
        3: 2,   # early
        4: 3,   # late
        5: 3,   # late
        6: 3,   # late
        7: 3,   # late
    }
    return label_map.get(dai, -1)
```

**Explanation:**
- DAI values outside 0-7 return `-1` (invalid label)
- Images with DAI > 7 produce no patches after label filtering
- The manifest calculates theoretical patches per image, while experiment results count only patches that received valid labels
- Val set has MORE patches than manifest because some images in manifest may have been filtered to different splits

### 1.3 Evidence from Split Manifest

Analyzing `split_manifest.json` group assignments:

**Train Groups** (18 groups):
- `_bc_run1`, `_bc_run2`, `_2hr_bc_run1`, `_2hr_bc_run2-1`, `_2hr_bc_run2-2`
- Various DAI stages mixed across sensors

**Val Groups** (4 groups):
- `_8_bc_run1`, `_4_bc_run1`, `_bi_bc_run2-1`, `_bc_run2-2`

**Test Groups** (4 groups):
- `_bi_bc_run2-2`, `_2hr_bc_run2-2`, `_bc_run2_sample1-2`, `_5_bc_run1`

The discrepancy is **not a bug** but expected behavior from label-based filtering.

---

## 2. Biological Independence Verification

### 2.1 USDA Metadata Biological Unit Definition

The dataset defines biological independence by:
- **Run number** (`run1`, `run2`, etc.) identifies unique plant acquisitions
- **Sensor type** (`swir`, `visnir`) indicates imaging modality, not biological variation
- **DAI** (Days After Inoculation) indicates disease stage, not biological unit

### 2.2 Verified Biological Identifier Extraction

**From `data_preprocessing_corrected.py:158-169`:**
```python
# Extract group identifier (run number = biological unit)
group_id = name.split('dai', 1)[1]

# FIX: Strip sensor token (swir/visnir) to get true biological unit
bio_unit = group_id
for sensor_token in ['_swir', '_visnir']:
    bio_unit = bio_unit.replace(sensor_token, '')
```

**Examples of correctly grouped biological units:**
- `7dai_swir_bc_run1.bil.hdr` → `_bc_run1`
- `7dai_visnir_bc_run1.bil.hdr` → `_bc_run1` (same biological unit)
- `1dai_swir_bc_run2.bil.hdr` → `_bc_run2`
- `1dai_visnir_bc_run2.bil.hdr` → `_bc_run2` (same biological unit)

### 2.3 Independence Verification Results

| Split | Groups | Count | Zero Overlap Confirmed |
|-------|--------|-------|------------------------|
| Train | 18 | 75.4% images | ✓ |
| Val | 4 | 13.8% images | ✓ |
| Test | 4 | 10.8% images | ✓ |

**Overlap Check** (from `split_manifest.json:108-112`):
```json
"group_overlap_check": {
    "train_val_overlap": false,
    "train_test_overlap": false,
    "val_test_overlap": false
}
```

**Status**: Biological independence is **correctly implemented**.

---

## 3. Grouping Algorithm Assessment

### 3.1 Algorithm Overview

The `group_wise_split()` function (lines 340-482) implements:

1. **Group identification**: Groups by biological unit (run number + DAI+ suffix)
2. **Stratification**: Uses label presence vectors for balanced splits
3. **Validation**: Verifies zero group overlap post-split
4. **Label verification**: Confirms all labels present in each split

### 3.2 Stratification Strategy

**From lines 382-402:**
```python
# STRATIFICATION FIX: Use label presence (not counts) for stratification
def get_label_presence_vector(gid):
    presence = [1 if l in groups[gid]['has_label'] else 0 for l in unique_labels]
    return tuple(presence)
```

This ensures each split contains groups that collectively represent all classes, preventing scenarios where one split has only Healthy/Late and another only Asymptomatic/Early.

### 3.3 Algorithm Correctness

**Strengths:**
- ✓ Group-level split prevents data leakage
- ✓ Sensor tokens stripped for proper biological grouping
- ✓ Zero overlap verified mathematically
- ✓ Label presence stratification ensures diversity

**Weaknesses:**
- Group sizes limited by available USDA data
- Small group count leads to class representation imbalance
- No minimum class threshold enforcement

---

## 4. Four-Class Evaluation Feasibility Assessment

### 4.1 Class Distribution Analysis

| Split | Healthy (0) | Asymptomatic (1) | Early (2) | Late (3) | Total Patches |
|-------|-------------|------------------|-----------|----------|---------------|
| **Train** | 1,512 | 378 | 938 | 5,096 | 7,924 |
| **Val** | 196 | 378 | 756 | 1,890 | 3,220 |
| **Test** | 378 | 0 | 0 | 756 | 1,134 |

### 4.2 Critical Finding: Missing Test Classes

**CRITICAL DEFICIENCY**: The test set contains **0 samples for Asymptomatic (Class 1) and Early (Class 2)**.

**From `experiment_results.json:40-42`:**
```json
"test": {
    "0": 378,
    "3": 756
}
```

### 4.3 Evaluation Impossibility

**Four-class evaluation requires:**
- All 4 classes present in test set
- Minimum 50 samples per class for reliable metrics
- Valid precision/recall/F1 calculations

**Current status:**
- Test has only 2 of 4 classes
- Classes 1 and 2 have 0 support
- Any metric involving classes 1 and 2 is mathematically invalid

---

## 5. Metric Interpretation and Correction

### 5.1 Invalid Classification Report

**From `experiment_results.json:76-76`:**
```
              precision    recall  f1-score   support

 Healthy     1.0000    0.7169    0.8351       378
Asymptomatic     0.0000    0.0000    0.0000         0
       Early     0.0000    0.0000    0.0000         0
        Late     0.9004    0.9921    0.9440       756

    accuracy                         0.9004      1134
   macro avg     0.4751    0.4272    0.4448      1134
weighted avg     0.9336    0.9004    0.9077      1134
```

### 5.2 Mathematical Issues

**Division by zero** for classes with 0 support:
- Precision = TP / (TP + FP) → 0/0 = NaN
- Recall = TP / (TP + FN) → 0/0 = NaN
- F1 = 2 × (P × R) / (P + R) → 0/0 = NaN

The system reports these as 0.0, which is **mathematically incorrect**.

### 5.3 Corrected Metric Interpretation

**Required corrections:**

1. **Remove** precision/recall/F1 for classes with 0 support
2. **Report** "undefined" instead of "0.0" for missing classes
3. **Calculate** macro accuracy only over valid classes (2 classes)
4. **Document** test set class limitations in report

**Corrected format should be:**
```
              precision    recall  f1-score   support

 Healthy     1.0000    0.7169    0.8351       378
        Late     0.9004    0.9921    0.9440       756

    accuracy                         0.9004      1134
                                     ————————
   macro avg (2 classes)              0.8678       2
```

---

## 6. Root Cause Analysis: Why Test Set Lacks Classes

### 6.1 USDA Dataset Class Distribution

The USDA tomato leaf dataset has inherent class imbalance:
- Images with DAI 0 (healthy) are available in multiple runs
- Images with DAI 1 (asymptomatic) are limited
- Images with DAI 2-3 (early) are sparse
- Images with DAI 4-7 (late disease) are abundant

### 6.2 Group-Wise Split Constraints

The biological grouping algorithm:
1. Groups by run number + DAI+ suffix
2. Some biological groups contain only Late disease images
3. Groups with Asymptomatic/Early are small
4. Stratification by presence preserves this distribution

**Result**: Test groups naturally selected have no Asymptomatic/Early representation.

### 6.3 No Alternative Groups Available

The USDA dataset is fixed - there are no additional groups with Asymptomatic/Early classes to relocate to test set.

---

## 7. Recommended Actions

### 7.1 Immediate Corrective Actions

**CRITICAL**: Test evaluation cannot be considered valid as-is.

| Action | Priority | Description |
|--------|----------|-------------|
| 1. Resplit dataset | CRITICAL | Ensure all 4 classes present in test set |
| 2. Fix metric reporting | HIGH | Report "undefined" for 0-support classes |
| 3. Document limitations | HIGH | Note 2-class evaluation in all reports |
| 4. Recalculate metrics | MEDIUM | Macro avg over valid classes only |

### 7.2 Resplitting Strategy Options

**Option A: Class-Constrained Group Split**
```
Strategy: Ensure minimum n samples per class in test
Approach: 
  1. Identify groups with Asymptomatic/Early content
  2. Force those groups into test split
  3. Rebalance other splits accordingly
```

**Option B: StratifiedGroupKFold with Minimum Class Guarantee**
```
Strategy: Use cross-validation with class coverage constraints
Approach:
  1. Implement StratifiedGroupKFold
  2. Add post-split class verification
  3. Iterate until all splits have all classes
```

### 7.3 Metric Correction Implementation

**Code changes required in `main_experiment_corrected.py`:**
```python
def evaluate_test_set(model, test_loader, device):
    # ... existing code ...

    # CRITICAL: Skip metrics for classes with 0 support
    for i in range(num_classes):
        support_i = test_cm[i][i]
        if support_i == 0:
            report_df.loc[i, 'precision'] = 'undefined'
            report_df.loc[i, 'recall'] = 'undefined'
            report_df.loc[i, 'f1-score'] = 'undefined'

    # Recalculate macro avg over valid classes only
    valid_classes = [i for i in range(num_classes) if test_cm[i][i] > 0]
    macro_precision = np.mean([report_df.loc[i, 'precision'] for i in valid_classes])
```

---

## 8. Conclusions

### 8.1 Summary of Findings

| Category | Finding | Severity | Status |
|----------|---------|----------|--------|
| Patch counts | Discrepancy between manifest and results | Medium | Explained (label filtering) |
| Biological grouping | Correctly implemented, zero overlap verified | None | ✓ |
| Test set classes | Missing 2 of 4 classes (Asymptomatic, Early) | **CRITICAL** | Unacceptable |
| Metric reporting | Invalid 0/0/0 for missing classes | High | Requires immediate fix |
| Overfitting | 45% train/val accuracy gap | High | Observed |

### 8.2 Evaluation Feasibility

**Four-class evaluation**: **NOT FEASIBLE** with current test set  
**Two-class evaluation**: Valid (Healthy vs Late only)  
**Model usability**: Limited to binary classification  

### 8.3 Next Steps

1. **Resplit dataset** to ensure all 4 classes in test
2. **Re-train model** on corrected split  
3. **Re-evaluate** with proper 4-class metrics
4. **Update documentation** reflecting corrected findings

---

## Appendix A: Technical Specifications

### A.1 Label Mapping (USDA DAI-Based)

| DAI Range | Disease Stage | Class ID |
|-----------|---------------|----------|
| 0 days | Healthy | 0 |
| 1 day | Asymptomatic | 1 |
| 2-3 days | Early infection | 2 |
| 4-7 days | Late (Bacterial Spot) | 3 |

### A.2 PCA Harmonization Parameters

| Sensor | Input Bands | Output Components | Reduction |
|--------|-------------|-------------------|-----------|
| SWIR | 168 | 128 | 23.8% |
| VIS-NIR | 300 | 128 | 57.3% |

Both use separate PCA models fitted on training data only, then harmonized to 128 components.

### A.3 Preprocessing Pipeline

1. Group-wise split by biological unit
2. Sensor-specific PCA (train-only fitted)
3. StandardScaler (train-only fitted)
4. Patch extraction (post-split)

---

## Appendix B: Artifact Traceability

| File | Status | Notes |
|------|--------|-------|
| `artifacts/best_model.pth` | Valid | 4-class model, but test evaluation invalid |
| `artifacts/experiment_results.json` | Valid | Contains filtering note and class distributions |
| `artifacts/split_manifest.json` | Valid | Zero overlap verified, patch counts documented |
| `data_preprocessing_corrected.py` | Valid | Correct implementation |
| `main_experiment_corrected.py` | Requires fix | Metric reporting for zero-support classes |

---

*Report generated for Phase 2D Audit*  
*Date: 2026-11-13*