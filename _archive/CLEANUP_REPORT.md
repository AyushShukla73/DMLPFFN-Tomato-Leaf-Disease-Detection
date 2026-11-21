---
name: cleanup-report
description: Audit trail and validation report for Stage B repository cleanup.
metadata:
  type: project
---

# Cleanup Report — DMLPFFN Tomato Leaf Disease Classification

**Date**: 2026-11-17  
**Stage**: B (Repository Cleanup & Documentation)  
**Status**: Complete

---

## Summary

This report documents the validation and correction steps performed during Stage B of the 6-phase cleanup process. All critical path discrepancies were resolved, documentation was created, and all artifacts were verified intact and archived (not deleted).

---

## Path Discrepancy Corrections

### Issue
Multiple files referenced a non-existent `artifacts/split_manifest.json` instead of the correct `artifacts/split_manifest_frozen.json`.

### Files Corrected

| File | Line | Change |
|------|------|--------|
| `verify_metrics.py` | 206 | `split_manifest.json` → `split_manifest_frozen.json` |
| `CODES/smoke_test.py` | 214 | `split_manifest.json` → `split_manifest_frozen.json` |

### Files Already Correct (Verified)
- `inference_dmlpffn.py` ✓
- `CODES/main_experiment_corrected.py` ✓
- `README.md` ✓

### Validation
The corrected manifest structure matches the expected split:
- **Train**: 39 images, 9,254 patches, 18 groups
- **Val**: 9 images, 1,694 patches, 4 groups
- **Test**: 7 images, 1,330 patches, 4 groups
- **Seed**: 42 (reproducible)
- **Overlap Check**: No train/val/test group overlap confirmed

---

## Documentation Created

| File | Description |
|------|-------------|
| `data/README.md` | Dataset documentation with USDA DOI, sensor configuration, preprocessing pipeline |
| `PROJECT_STATUS.md` | Project status, phase completion, quick start guide |
| `ARTIFACT_INDEX.md` | Comprehensive inventory of all checkpoints, artifacts, figures |
| `CLEANUP_REPORT.md` | This file — audit trail of all cleanup changes |

---

## Artifact Preservation

All files preserved per user request ("archive files, not delete"):

```
artifacts/
├── split_manifest_frozen.json          [✓ Valid]
├── preprocessing_artifacts_sklearn.joblib [✓ Valid]
├── preprocessing_artifacts_arrays.npz   [✓ Valid]
├── best_model.pth                       [✓ Valid - 95.64% accuracy]
├── dmlpffn_test_logits_frozen.json      [✓ Valid - 1,330 test logits]
├── experiment_results_recovered.json    [✓ Valid]
├── statistical_analysis_accuracy.json   [✓ Valid]
├── statistical_analysis_macro_f1.json   [✓ Valid]
├── baselines/
│   ├── simple_cnn/
│   │   ├── aggregate_results.json       [✓ Valid - 5 seeds]
│   │   ├── best_model.pth
│   │   ├── seed_42/best_model.pth       [✓ Valid]
│   │   └── ... (seed_123, seed_456, seed_789, seed_999)
│   ├── mlp_mixer_hsi/
│   │   ├── aggregate_results.json       [✓ Valid - 5 seeds]
│   │   └── seed_* files
│   ├── resnet18_hsi/
│   │   ├── aggregate_results.json       [✓ Valid - 5 seeds]
│   │   └── seed_* files
│   ├── svm/
│   │   └── aggregate_results.json       [✓ Valid]
│   └── random_forest/
│       └── aggregate_results.json       [✓ Valid]
└── baselines/process_registry.json      [✓ Valid]
```

---

## Baseline Results Summary

**Model Performance (Test Accuracy)**

| Model | Mean Accuracy | ± Std | Parameters | Notes |
|-------|--------------|-------|------------|-------|
| DMLPFFN | **95.64%** | - | 447,780 | Baseline reference |
| SimpleCNN | 52.92% | ±7.30% | 86,156 | 5 seeds: 42, 123, 456, 789, 999 |
| MLP-Mixer HSI | 46.39% | ±11.73% | 2,310,812 | 5 seeds: 42, 123, 456, 789, 999 |
| ResNet-18 HSI | 45.67% | ±5.63% | 11,242,884 | 5 seeds: 42, 123, 456, 789, 999 |
| SVM (RBF) | - | - | - | Classical baseline |
| Random Forest | - | - | - | 500 trees |

---

## Verification Checklist

- [x] All path references corrected
- [x] Split manifest validated (no overlap, seed=42)
- [x] All model checkpoints present and valid
- [x] Preprocessing artifacts present (PCA, scaler)
- [x] Test logits present for DMLPFFN evaluation
- [x] Baseline results verified for all 5 seeds
- [x] Statistical analysis files present
- [x] Documentation files created/updated
- [x] Smoke test input format bug fixed (NHWC → NCHW)

---

## Additional Fix: Smoke Test Input Format

### Issue
`CODES/smoke_test.py` passed NHWC tensor `[B, 9, 9, 128]` to models expecting NCHW format.

### Root Cause
PyTorch Conv2d layers use NCHW format (batch, channels, height, width), but the smoke test created input in NHWC (batch, height, width, channels).

### Resolution
Changed line 103:
```diff
- dummy_input = torch.randn(batch_size, 9, 9, 128).to(device)  # NHWC
+ dummy_input = torch.randn(batch_size, 128, 9, 9).to(device)  # NCHW
```

### Verification
All smoke tests now pass:
- SimpleCNN: 86,156 params ✓
- ResNet18_HSI: 11,242,884 params ✓
- MLPMixer_HSI: 2,310,812 params ✓

---

## Next Actions (Stage C)

- [ ] Manuscript preparation
- [ ] Publication figure generation
- [ ] Experiment archive

---

## Audit Trail

All changes are reversible. Git diff summary:

```diff
- "split_manifest.json"
+ "split_manifest_frozen.json"
```

(Initial fix in 2 files: `verify_metrics.py`, `CODES/smoke_test.py`)

```diff
- dummy_input = torch.randn(batch_size, 9, 9, 128).to(device)  # NHWC
+ dummy_input = torch.randn(batch_size, 128, 9, 9).to(device)  # NCHW
```

(Additional fix in `CODES/smoke_test.py` line 103: corrected tensor format for PyTorch Conv2d compatibility)

---

**Report Generated**: 2026-11-17  
**Cleanup Phase Completed**: B (Repository Cleanup & Documentation)