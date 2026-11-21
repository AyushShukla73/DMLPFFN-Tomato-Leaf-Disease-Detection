# Phase 3 & Phase 4 Final Metric Corrections Report

**Date**: 2026-11-14  
**Status**: Final Corrections Applied  
**Scope**: Corrections to PHASE3_4_AUDIT_CORRECTIONS.md based on 4 identified issues  
**Constraint**: No file modifications to existing experiments or model code  

---

## Executive Summary

| Correction Item | Status | Key Finding |
|-----------------|--------|-------------|
| 1. Confusion matrix using column totals | ✅ **CORRECTED** | Late precision = 0.9595, F1 = 0.9652; aggregates recalculated |
| 2. Epoch-indexing same-epoch gap | ✅ **CORRECTED** | Best val epoch 21: train 89.83%, val 78.75% → **gap = 11.08 pp** |
| 3. Filename-group vs biological independence | ✅ **CLARIFIED** | Group = "run" identifier; plant/leaf independence not verified from metadata |
| 4. Artifact preservation | ✅ **CONFIRMED** | All checkpoints, manifests, results preserved; recovered JSON separate |

---

## Correction Item 1: Confusion Matrix Using Column Totals

**Status**: ✅ **CORRECTED — Late class precision = 0.9595, F1 = 0.9652**

### Verified Confusion Matrix (from `artifacts/best_model.pth`):
```
test_cm = [
    [538,   4,   1,  31],  # Actual Healthy (574 total)
    [  0,   0,   0,   0],  # Actual Asymptomatic (0 total)
    [  0,   0,   0,   0],  # Actual Early (0 total)
    [  1,  16,   5, 734]   # Actual Late (756 total)
]
```

### Column Totals (Predicted Class Distribution):
| Predicted Class | Column Sum | Calculation |
|-----------------|------------|-------------|
| Healthy (0) | **539** | 538 + 0 + 0 + 1 |
| Asymptomatic (1) | **20** | 4 + 0 + 0 + 16 |
| Early (2) | **6** | 1 + 0 + 0 + 5 |
| Late (3) | **765** | 31 + 0 + 0 + 734 |

### Per-Class Metrics (Column-Based Precision, Row-Based Recall):

| Class | Support (Row) | TP | FP (Col-TP) | FN (Row-TP) | Precision | Recall | F1 |
|-------|---------------|----|-------------|-------------|-----------|--------|-----|
| **Healthy (0)** | 574 | 538 | 1 | 36 | **0.9981** | **0.9373** | **0.9668** |
| Asymptomatic (1) | 0 | 0 | 20 | 0 | 0.0000* | undefined | undefined |
| Early (2) | 0 | 0 | 6 | 0 | 0.0000* | undefined | undefined |
| **Late (3)** | 756 | 734 | 31 | 22 | **0.9595** | **0.9709** | **0.9652** |

*\*Precision = 0 when support = 0 and FP > 0; recall undefined (0/0)*

### Aggregate Metrics (2-Class Valid Only: Healthy + Late):

| Metric | Value | Calculation |
|--------|-------|-------------|
| **Accuracy** | **0.9564** | (538 + 734) / 1330 = 1272/1330 |
| **Macro Precision (2-class)** | **0.9788** | (0.9981 + 0.9595) / 2 |
| **Macro Recall (2-class)** | **0.9541** | (0.9373 + 0.9709) / 2 |
| **Macro F1 (2-class)** | **0.9660** | (0.9668 + 0.9652) / 2 |
| **Weighted Precision** | **0.9762** | (574×0.9981 + 756×0.9595) / 1330 |
| **Weighted Recall** | **0.9564** | (574×0.9373 + 756×0.9709) / 1330 |
| **Weighted F1** | **0.9658** | (574×0.9668 + 756×0.9652) / 1330 |

**Correction Summary**: 
- Previous audit used row totals for FP → Late precision 0.9532, F1 0.9620
- **Correct (column totals)**: Late precision **0.9595**, F1 **0.9652**
- 4-class macro metrics remain invalid (2 classes have 0 support)
- 2-class macro and weighted metrics are the valid reporting basis

---

## Correction Item 2: Epoch-Indexing Same-Epoch Gap

**Status**: ✅ **CORRECTED — Gap = 11.08 pp at epoch 21 (not 9.77 pp)**

### History Array Indexing (from `experiment_results_recovered.json`):

The `history` arrays have 46 entries (indices 0–45) corresponding to **epochs 1–46** (1-indexed epoch numbering).

| Index | Epoch | Train Acc | Val Acc | Gap (pp) |
|-------|-------|-----------|---------|----------|
| 19 | 20 | 0.8942 | 0.7869 | 10.73 |
| **20** | **21** | **0.8983** | **0.7875** | **11.08** ← **Best Val Epoch** |
| 21 | 22 | 0.8852 | 0.7844 | 10.08 |

### Checkpoint Verification:
- `best_model.pth` contains: `"epoch": 21`, `"val_acc": 0.7874852420306966`
- This matches `history["val_acc"][20]` = 0.7875 (epoch 21, index 20)
- Corresponding train accuracy: `history["train_acc"][20]` = **0.8983**

### Phase 2B Same-Epoch Comparison (from PHASE2B_EXPERIMENT_RESULTS.md):
| Phase | Epoch | Train Acc | Val Acc | Gap (pp) |
|-------|-------|-----------|---------|----------|
| **Phase 2B** | 22 (best val) | **99.10%** | **54.91%** | **44.19** |
| **Phase 3** | 21 (best val) | **89.83%** | **78.75%** | **11.08** |

**Correction Summary**:
- Previous audit used epoch 22 train (88.52%) for Phase 3 → gap 9.77 pp
- **Correct**: Phase 3 best val is epoch 21 with train 89.83% → **gap = 11.08 pp**
- Overfitting reduction: **44.19 pp → 11.08 pp** (33.11 pp reduction, 75% relative)

---

## Correction Item 3: Filename-Group vs Biological Independence

**Status**: ✅ **CLARIFIED — Group = "run" identifier; plant/leaf independence NOT verified**

### Group Assignment Algorithm (from `data_preprocessing_corrected.py`):
```python
def parse_filename(filename):
    # Strips sensor tokens: "_swir", "_visnir"
    # e.g., "7dai_swir_bc_run1.bil.hdr" → "_bc_run1"
    # e.g., "1dai_visnir_bc_run1.bil.hdr" → "_bc_run1"
```

### What the Group ID Represents:
| Filename Example | Extracted Group | Interpretation |
|------------------|-----------------|----------------|
| `7dai_swir_bc_run1.bil.hdr` | `_bc_run1` | Biological culture run 1 |
| `7dai_visnir_bc_run1.bil.hdr` | `_bc_run1` | Same run, different sensor |
| `0dai_2hr_swir_bc_run2-1.bil.hdr` | `_2hr_bc_run2-1` | Run 2, sub-batch 1 |
| `7dai_5_swir_bc_run1.bil.hdr` | `_5_bc_run1` | Run 1, sub-batch 5 |

### Verified Independence:
- ✅ **No group overlap** between train/val/test (confirmed in `split_manifest.json`)
- ✅ **Sensor independence**: SWIR and VIS-NIR images from same run grouped together
- ✅ **Temporal independence**: Different days (0dai, 1dai, 7dai) can be in different splits

### NOT Verified (Requires USDA Metadata):
- ❌ **Plant-level independence**: Unknown if multiple images in same run = same plant
- ❌ **Leaf-level independence**: Unknown if patches from same image share leaves
- ❌ **Biological replicate structure**: "run" may or may not = distinct biological unit

### Correct Claim:
> "Group-wise split ensures no **run-level** overlap between splits. Plant/leaf-level independence is not verified from available metadata."

---

## Correction Item 4: Artifact Preservation

**Status**: ✅ **CONFIRMED — All Original Artifacts Preserved**

### Preserved Files (Unmodified):
| File | Purpose |
|------|---------|
| `artifacts/best_model.pth` | Phase 3 best checkpoint (epoch 21, val_acc=0.7875) |
| `artifacts/split_manifest.json` | Reproducible split specification (seed=42) |
| `artifacts/experiment_results.json` | Original truncated output (743 bytes) |
| `artifacts/preprocessing_artifacts/` | Sensor-specific PCA models, scalers |
| `PHASE2B_EXPERIMENT_RESULTS.md` | Phase 2B completion report |
| `PHASE2C_ARTIFACT_AND_SPLIT_AUDIT.md` | Phase 2C audit document |
| `PHASE3_4_COMPLETION_REPORT.md` | Phase 3/4 completion report |
| `PHASE3_4_VERIFICATION_AUDIT.md` | Original verification audit |
| `PHASE3_4_AUDIT_CORRECTIONS.md` | Previous correction report |
| `PHASE4_MANUSCRIPT_ALIGNMENT.md` | Manuscript alignment audit |

### New File Created (Non-Destructive):
| File | Purpose |
|------|---------|
| `artifacts/experiment_results_recovered.json` | Complete results recovered from checkpoint |
| `PHASE3_4_FINAL_METRIC_CORRECTIONS.md` | **This report** |

---

## Summary of Final Corrected Metrics

| Metric | Previous Audit | Previous Correction | **Final Corrected** |
|--------|----------------|---------------------|---------------------|
| Late Precision | 0.9595* | 0.9532 | **0.9595** |
| Late F1 | 0.9652* | 0.9620 | **0.9652** |
| Phase 3 Same-Epoch Gap | 11.56%** | 9.77 pp | **11.08 pp** |
| Overfitting Reduction | 45.06→11.56 | 44.19→9.77 | **44.19→11.08** |

*\*Phase 3 completion report value (correct)  
**Final train vs best val (not same-epoch)*

---

**All 4 final corrections applied. No existing artifacts modified. Phase 5 may proceed with Phase 3 split manifest as fixed split.**

*Report generated: 2026-11-14*