# Phase 2E: Corrections to Final Evaluation Protocol

## Overview
This document records all 6 required corrections to `PHASE2E_FINAL_EVALUATION_PROTOCOL.md` with exact code references and evidence from the artifacts. Each correction addresses a specific issue identified during the audit.

---

## Correction 1: Biological Independence Claim

### Issue in Protocol
**Section 1.2 (Lines 20-35)**: States "**VERIFIED: Biological independence is correctly implemented.**" and presents this as conclusive proof of independent biological sampling.

### Evidence from Code
**`data_preprocessing_corrected.py:158-169`** (parse_filename function):
```python
# Extract group identifier (run number = biological unit)
group_id = name.split('dai', 1)[1]

# FIX: Strip sensor token (swir/visnir) to get true biological unit
bio_unit = group_id
for sensor_token in ['_swir', '_visnir']:
    bio_unit = bio_unit.replace(sensor_token, '')
```

**`split_manifest.json:108-112`** confirms zero group overlap:
```json
"group_overlap_check": {
    "train_val_overlap": false,
    "train_test_overlap": false,
    "val_test_overlap": false
}
```

### Corrected Claim
**Change from**: "VERIFIED: Biological independence is correctly implemented."

**Change to**: "VERIFIED: **Zero group overlap is confirmed** between all splits (train/val/test). The algorithm correctly strips sensor tokens (`_swir`, `_visnir`) to isolate run identifiers as biological units. However, **independent biological sampling cannot be conclusively established from filename evidence alone** — true independence requires verification that each `_bc_runX` represents a distinct plant/leaf acquisition event in the USDA metadata, which is outside the scope of filename parsing."

### Rationale
The code correctly prevents data leakage by grouping sensor variants of the same run together. However, the claim "biological independence is correctly implemented" overstates what the filename-based grouping can prove. The protocol should distinguish between:
1. **What the code does**: Correctly groups sensor variants, zero overlap verified
2. **What this proves**: No data leakage from sensor modality
3. **What remains unproven**: Whether different run identifiers (`_bc_run1` vs `_bc_run2`) represent truly independent biological samples (different plants/leaves) — this requires USDA metadata verification beyond filenames

---

## Correction 2: Patch-Count Discrepancy Classification

### Issue in Protocol
**Section 2.3 (Lines 85-94)**: States "Discrepancy is **expected behavior**, not a bug. Total patches conserved (12,278)." without reproducing the per-image calculation.

### Evidence from Artifacts
**`split_manifest.json`** (Lines 73-75): Theoretical patch counts
- Train: 9,254 | Val: 1,694 | Test: 1,330 | Total: 12,278

**`experiment_results.json`** (Lines 6-24): Actual patch counts  
- Train: 7,924 | Val: 3,220 | Test: 1,134 | Total: 12,278

**`data_preprocessing_corrected.py:172-184`** (get_label_from_dai):
```python
def get_label_from_dai(dai: int) -> int:
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
    return label_map.get(dai, -1)  # Returns -1 for DAI > 7
```

**`data_preprocessing_corrected.py:704-797`** (process_metadata_to_patches):
- Patches with label=-1 are **discarded during extraction** (line 794-795)

### Root Cause Not Reproduced
The protocol claims the discrepancy is due to "DAI-based label mapping filter removes DAI > 7 images" but **no per-image patch count reproduction is provided**. The manifest calculates theoretical patches from image dimensions × patch parameters, while the experiment counts actual patches after:
1. DAI > 7 filtering (label = -1 → discarded)
2. Patch extraction with stride=4

### Corrected Classification
**Change from**: "Discrepancy is expected behavior, not a bug."

**Change to**: "**UNRESOLVED: Root cause classified as unverified pending per-image reproduction.** The total patch conservation (12,278) suggests no data loss, but the exact per-image contribution to the discrepancy has not been reproduced. The stated explanation (DAI > 7 filtering) is plausible based on code inspection of `get_label_from_dai()` returning -1 for DAI > 7 and `process_metadata_to_patches()` discarding label=-1 patches, but **no quantitative per-image trace** connects manifest theoretical counts to experiment actual counts. This remains an open item requiring reproduction."

### Required Action
To resolve: Write a script that iterates through each image in `split_manifest.json`, computes theoretical patches from dimensions, applies DAI filter, and sums to verify exact match with experiment counts.

---

## Correction 3: Stratification Claim Contradiction

### Issue in Protocol
**Section 3.2 (Lines 110-117)** and **Section 3.3 (Lines 120-133)**: Claims "The stratification ensures **at least one group with each label** is in each split" and "The stratification algorithm is correctly implemented but **cannot overcome fundamental dataset limitations**."

### Evidence from Code
**`data_preprocessing_corrected.py:392-444`** (group_wise_split function):
```python
# Stratified split at group level using label presence vectors
np.random.seed(seed)

# Strategy: Use stratified split only when it provides meaningful balance
# Check if stratification by presence vectors is feasible
unique_vectors = len(set(group_presence_vectors))
min_vector_count = min([group_presence_vectors.count(v) for v in set(group_presence_vectors)]) if unique_vectors > 1 else 0

# For stratification to work reasonably, we need multiple groups per vector pattern
use_stratification = unique_vectors > 1 and min_vector_count >= 2

if use_stratification:
    # ... stratified splits ...
else:
    # Fall back to random split when stratification not feasible
    train_groups, temp_groups = train_test_split(
        group_ids, test_size=1-train_ratio, random_state=seed
    )
    # ...
```

**Key finding at Line 402**: `use_stratification = unique_vectors > 1 and min_vector_count >= 2`

**Post-split verification at Lines 476-480** (prints but does NOT enforce):
```python
# Verify all labels are present in each split
for name, meta_list in [('Train', train_meta), ('Val', val_meta), ('Test', test_meta)]:
    labels = [m.label for m in meta_list]
    unique_labels_in_split = sorted(set(labels))
    print(f"{name} labels present: {unique_labels_in_split}")
```

### Contradiction
The protocol claims stratification "ensures at least one group with each label is in each split" but the code:
1. **Only attempts stratification when feasible** (Line 402: requires `min_vector_count >= 2`)
2. **Falls back to random split** when not feasible (Lines 432-444)
3. **Only prints label presence post-split** (Line 480) — does NOT raise error or re-split

With 26 groups and 4 classes, many label-presence vectors are unique or have count=1, making `min_vector_count >= 2` false. The code likely fell back to random split.

### Corrected Claim
**Change from**: "The stratification ensures at least one group with each label is in each split"

**Change to**: "**Stratification is attempted but not guaranteed.** The algorithm uses label-presence vectors and **only applies stratified splitting when `unique_vectors > 1 and min_vector_count >= 2`** (`data_preprocessing_corrected.py:402`). When this feasibility check fails, it **falls back to random split** (Lines 432-444). The post-split label presence check (Lines 476-480) **only prints** the labels present — it does **not enforce** class coverage, raise errors, or trigger re-splitting. Therefore, the test set lacking Classes 1 and 2 is a direct consequence of the fallback to random split, not a limitation of the stratification algorithm itself."

---

## Correction 4: Cross-Validation Feasibility Statements

### Issue in Protocol
**Section 4.2 (Lines 160-173)**: Conflates "fold generation feasibility" with "class coverage feasibility." States "Standard cross-validation is infeasible with current USDA dataset" without distinguishing these two distinct constraints.

### Evidence from Code and Data
**`split_manifest.json:96-107`** (Group assignments):
- Train: 18 groups
- Val: 4 groups  
- Test: 4 groups
- Total: 26 biological groups

**Class distribution per group** (from PHASE2D_EVALUATION_FEASIBILITY.md, Lines 144-157):
- Class 1 (Asymptomatic): Present in only **3 groups** (`_4_bc_run1`, `_bi_bc_run2-1`, `_bc_run2-2`)
- Class 2 (Early): Present in only **3 groups** (`_4_bc_run1`, `_bc_run2-2`, `_bi_bc_run2-2`)

### Two Distinct Constraints
| Constraint | Requirement | Current Status | Feasibility |
|------------|-------------|----------------|-------------|
| **Fold Generation** | Minimum 5 groups for 5-fold CV | 26 total groups | ✓ **FEASIBLE** |
| **Class Coverage** | Each class in ≥ 5 groups for 5-fold | Classes 1,2 in only 3 groups | ✗ **INFEASIBLE** |

### Corrected Statement
**Change from**: "Standard cross-validation is infeasible with current USDA dataset."

**Change to**: "**Standard 5-fold cross-validation is infeasible for 4-class metrics due to class coverage constraints, not fold generation constraints.** 
- **Fold generation**: **FEASIBLE** — 26 biological groups support 5-fold CV (5-6 groups per fold)
- **Class coverage per fold**: **INFEASIBLE** — Classes 1 and 2 appear in only 3 groups each, fewer than the 5 folds required for each fold to contain all classes
- **Leave-One-Group-Out (LOGO)**: **INFEASIBLE** for 4-class metrics — would produce folds missing Classes 1 and/or 2
- **Custom constrained split**: **MARGINAL** — requires forcing specific Class 1/2 groups into test, but only 3 such groups exist total

The distinction matters: fold *generation* is possible; class *coverage per fold* is the binding constraint."

---

## Correction 5: Metric Calculations for Missing Classes

### Issue in Protocol
**Section 5.2 (Lines 204-209)** and **Section 5.4 (Lines 237-238)**: States "Precision = 0/0 = **NaN** (not 0.0)" for Classes 1 & 2, but the confusion matrix shows **FPs exist** for these classes, making precision = 0 (not NaN), while recall/F1 = undefined (zero support).

### Evidence from Confusion Matrix
**`experiment_results.json:50-75`** (test_cm):
```
                Predicted
              0    1    2    3
Actual  0   271   2   22   83   (Healthy: 378)
        1     0   0    0    0   (Asymptomatic: 0)
        2     0   0    0    0   (Early: 0)
        3     0   4    2  750   (Late: 756)
```

### Correct Mathematical Derivation

**Class 1 (Asymptomatic) — Support = 0:**
- TP = 0 (no actual Class 1 samples)
- FP = 2 (from Healthy) + 4 (from Late) = **6**
- FN = 0 (no actual Class 1 samples)
- **Precision = TP / (TP + FP) = 0 / (0 + 6) = 0.0** (NOT NaN — denominator is 6)
- **Recall = TP / (TP + FN) = 0 / 0 = undefined**
- **F1 = undefined** (requires both P and R defined)

**Class 2 (Early) — Support = 0:**
- TP = 0
- FP = 22 (from Healthy) + 2 (from Late) = **24**
- FN = 0
- **Precision = TP / (TP + FP) = 0 / (0 + 24) = 0.0** (NOT NaN — denominator is 24)
- **Recall = TP / (TP + FN) = 0 / 0 = undefined**
- **F1 = undefined**

### Corrected Metrics Table
**Change Section 5.3 (Lines 214-222) from**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Healthy (0) | 1.0000 | 0.7169 | 0.8351 | 378 |
| Late (3) | 0.9004 | 0.9921 | 0.9440 | 756 |

**To**:
| Class | Precision | Recall | F1-Score | Support | Notes |
|-------|-----------|--------|----------|---------|-------|
| Healthy (0) | 1.0000 | 0.7169 | 0.8351 | 378 | Valid |
| Asymptomatic (1) | **0.0000** | **undefined** | **undefined** | 0 | FP=6 from Classes 0,3 |
| Early (2) | **0.0000** | **undefined** | **undefined** | 0 | FP=24 from Classes 0,3 |
| Late (3) | 0.9004 | 0.9921 | 0.9440 | 756 | Valid |

**Change Section 5.4 (Lines 237-238) from**:
```
Asymptomatic: UNDEFINED (0 support)
Early: UNDEFINED (0 support)
```

**To**:
```
Asymptomatic: Precision=0.0 (FP=6), Recall=undefined, F1=undefined (0 support)
Early: Precision=0.0 (FP=24), Recall=undefined, F1=undefined (0 support)
```

**Change Section 5.2 (Lines 204-208) from**:
```
- Classes 1 & 2: Precision = TP/(TP+FP) = 0/0 = **NaN** (not 0.0)
- Classes 1 & 2: Recall = TP/(TP+FN) = 0/0 = **NaN** (not 0.0)
- Classes 1 & 2: F1 = 2PR/(P+R) = 0/0 = **NaN** (not 0.0)
```

**To**:
```
- Class 1: Precision = 0/(0+6) = **0.0** (FP from Class 0:2, Class 3:4); Recall = 0/0 = **undefined**; F1 = **undefined**
- Class 2: Precision = 0/(0+24) = **0.0** (FP from Class 0:22, Class 3:2); Recall = 0/0 = **undefined**; F1 = **undefined"
```

### Macro Average Correction
**Section 5.3 Line 221**: "Macro Average (2 classes): Precision=0.9502, Recall=0.8545, F1=0.8896" is correct for 2-class macro.

**But if 4-class macro were reported**: It must either:
- Exclude undefined classes (2-class macro = 0.8896 F1), OR
- Report as "undefined" for 4-class macro

The protocol must explicitly state: **4-class macro average is undefined** because 2 of 4 classes have undefined recall/F1.

---

## Correction 6: Synthetic Augmentation Recommendation Removal

### Issue in Protocol
**Section 6.2 (Lines 268-275)** and **Section 7.2 (Lines 338-339)**: Recommends "Consider synthetic oversampling for rare classes" and "Investigate data augmentation for Classes 1, 2" as solutions for cross-validation feasibility.

### Evidence from Code
**`data_preprocessing_corrected.py:340-482`** (group_wise_split): The entire split logic operates on **biological groups**. Any synthetic augmentation that creates patches without corresponding biological groups would:
1. Violate the group-wise split principle (patches from same biological unit must stay together)
2. Create artificial groups that don't correspond to real USDA acquisitions
3. Break the zero-overlap guarantee

**`data_preprocessing_corrected.py:704-797`** (process_metadata_to_patches): Patches are extracted from **real images** with real DAI labels. There is no mechanism for synthetic patch generation in the pipeline.

### Group-Forcing Constraint Clarification
The protocol suggests "Force groups with Classes 1,2 into test" (Line 271) but doesn't address that this reduces training data for those classes. With only 3 groups containing Class 1 and 3 groups containing Class 2:
- Forcing 1 group of Class 1 to test → only 2 groups left for train+val
- Forcing 1 group of Class 2 to test → only 2 groups left for train+val
- This makes training on these classes even more data-starved

### Corrected Recommendations
**Remove from Section 6.2 (Line 272)**:
- ~~"Augmentation: Consider synthetic oversampling for rare classes"~~

**Remove from Section 7.2 (Line 339)**:
- ~~"MEDIUM: Investigate data augmentation for Classes 1, 2"~~

**Replace with in Section 6.2**:
1. **Group-Forcing Constraints**: Any resplit must force specific biological groups containing Classes 1,2 into test. With only 3 groups per rare class, this leaves ≤2 groups for train+val, severely limiting training data.
2. **No Synthetic Augmentation**: The group-wise split architecture prohibits synthetic patch generation — all patches must originate from real USDA images with verifiable biological group membership.
3. **Alternative: External Data**: If 4-class evaluation is required, new USDA acquisitions or external datasets with verified biological independence are needed.

**Replace Section 7.2 (Lines 338-339) with**:
```
3. MEDIUM: Document group-forcing constraints — forcing Class 1/2 groups to test leaves ≤2 groups for training
4. MEDIUM: Evaluate external data acquisition — synthetic augmentation incompatible with group-wise split architecture
```

---

## Summary of All Corrections

| # | Section | Original Claim | Corrected Claim |
|---|---------|----------------|-----------------|
| 1 | 1.2 | "Biological independence correctly implemented" | "Zero overlap verified; independent biological sampling not conclusively established from filenames alone" |
| 2 | 2.3 | "Discrepancy is expected behavior" | "UNRESOLVED — root cause unverified pending per-image reproduction" |
| 3 | 3.2, 3.3 | "Stratification ensures class coverage" | "Stratification attempted only when feasible; falls back to random split; no enforcement" |
| 4 | 4.2 | "CV infeasible" | "Fold generation feasible; class coverage per fold infeasible — distinct constraints" |
| 5 | 5.2-5.4 | "Precision=NaN for missing classes" | "Precision=0.0 (FPs exist); Recall/F1=undefined (zero support)" |
| 6 | 6.2, 7.2 | "Synthetic augmentation recommended" | "Removed — incompatible with group-wise split; group-forcing constraints documented" |

---

## References
- `data_preprocessing_corrected.py:132-169` (parse_filename)
- `data_preprocessing_corrected.py:172-184` (get_label_from_dai)
- `data_preprocessing_corrected.py:340-482` (group_wise_split)
- `data_preprocessing_corrected.py:704-797` (process_metadata_to_patches)
- `split_manifest.json` (theoretical counts, group assignments, overlap check)
- `experiment_results.json` (actual counts, confusion matrix, class distributions)
- `PHASE2D_EVALUATION_FEASIBILITY.md` (group class composition table)

*Document created: 2026-11-13*