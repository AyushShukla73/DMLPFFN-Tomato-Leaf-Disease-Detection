# Phase 5 Continuation State - DMLPFFN Tomato Leaf Spot Detection

**Date**: 2026-11-16  
**Status**: IN PROGRESS - Continuation after power loss  
**Previous Session Date**: 2026-11-14

---

## Summary

This document captures the state of Phase 5 baseline experiments after a power loss interrupted the work. The goal is to complete all 5 baseline models across 5 seeds, compute aggregate statistics, run statistical analysis, and generate the final PHASE5_BASELINE_RESULTS.md report.

---

## Current State of Baseline Results

### Completed Models (5/5 baselines have all 5 seeds trained)

| Model | Seeds Complete | Aggregate Status | Notes |
|-------|----------------|------------------|-------|
| **simple_cnn** | 5/5 (42, 123, 456, 789, 999) | ✅ Complete (5 seeds) | Macro/Weighted F1 = 0 for all seeds (evaluation issue) |
| **mlp_mixer_hsi** | 5/5 (42, 123, 456, 789, 999) | ⚠️ **INCOMPLETE** | Seeds 42, 123, 456 missing full metrics; only test_acc/test_loss |
| **resnet18_hsi** | 5/5 (42, 123, 456, 789, 999) | ⚠️ **INCOMPLETE** | Aggregate only has 2 values per metric (seeds 789, 999?) |
| **svm** | 1/1 (single run) | ✅ Complete | Classical ML - single run |
| **random_forest** | 1/1 (single run) | ✅ Complete | Classical ML - single run |

---

## Key Issues Identified

### 1. mlp_mixer_hsi - Seeds 42, 123, 456 Missing Full Metrics
**Files affected:**
- `artifacts/baselines/mlp_mixer_hsi/seed_42/test_results.json` - only has `test_acc`, `test_loss`
- `artifacts/baselines/mlp_mixer_hsi/seed_123/test_results.json` - only has `test_acc`, `test_loss`
- `artifacts/baselines/mlp_mixer_hsi/seed_456/test_results.json` - only has `test_acc`, `test_loss`
- Seeds 789, 999 have complete metrics (val_accuracy, macro_f1, weighted_f1, etc.)

**Root cause:** Original evaluation didn't run conditional binary metrics for early seeds.  
**Solution:** Re-evaluate seeds 42, 123, 456 using their saved `best_model.pth` weights (inference only, no retraining needed).

### 2. resnet18_hsi - Aggregate Results Incomplete
**Files affected:**
- `artifacts/baselines/resnet18_hsi/aggregate_results.json` - only 2 values in each "values" array
- Need to verify all 5 seed test_results.json have complete metrics

### 3. simple_cnn - Macro F1 = 0 for All Seeds
**Files affected:**
- All 5 seeds show `macro_precision: 0`, `macro_recall: 0`, `macro_f1: 0`, `weighted_precision: 0`, `weighted_recall: 0`, `weighted_f1: 0`
- **Root cause:** Conditional binary evaluation (Healthy vs Late from 4-class head) not properly applied
- Test accuracy is valid (~0.93), but macro/weighted metrics are wrong

### 4. Missing Test Artifacts for Statistical Analysis
**No test_logits.npy or test_groups.npy files exist for ANY model/seed**
- Required for: Group-aware bootstrap CI (1000 resamples), Group-aware McNemar's test
- `trainer.py::validate()` was modified to accept `save_logits`/`save_groups` parameters but never called with them

---

## Data Pipeline (Immutable - From Phase 3)

**Split Manifest:** `artifacts/split_manifest.json` (seed=42, frozen)

1. **Band Removal**: SWIR bands 94-115 removed (water absorption)
2. **PCA**: IncrementalPCA (n_components=128) per sensor, fitted on TRAIN patches only
3. **Scaling**: StandardScaler fitted on TRAIN patches only
4. **Patch Extraction**: 9x9 patches, stride=4, AFTER group-wise split
5. **Group-wise Split**: By biological unit (run number)
   - Test groups: `["_bi_bc_run2-2", "_2hr_bc_run2-2", "_bc_run2_sample1-2", "_5_bc_run1"]`
   - Test set: 4 test groups, 1,330 patches (Healthy=574, Late=756)

---

## Training Configuration (Identical for all DL baselines, from Phase 3)

- Batch size: 16
- Optimizer: AdamW
- Learning rate: 2e-4
- Weight decay: 5e-4
- Epochs: 250 (early stop patience=25)
- Scheduler: CosineAnnealingWarmRestarts(T_0=10)
- Label smoothing: 0.1
- Mixup: alpha=0.2 (50% probability)
- CutMix: alpha=1.0 (50% probability)
- Class balancing: WeightedRandomSampler
- Seed: 42 (all RNGs)
- Device: CUDA (RTX 3050 4GB)
- Mixed precision: Enabled

---

## Evaluation Protocol

**Conditional Binary**: P(Healthy|H/L) = exp(z0)/(exp(z0)+exp(z3)), P(Late|H/L) = exp(z3)/(exp(z0)+exp(z3))

**Metrics**: Accuracy, Macro/Weighted Precision/Recall/F1, Per-class P/R/F1, 2x2 Confusion Matrix, AUC-ROC  
**Group-level**: Majority vote per group (4 test groups)  
**Statistical rigor**: 5 seeds, group-aware bootstrap CI (1000 resamples), group-aware McNemar's test

---

## Files Modified in Previous Session

1. **CODES/baselines/trainer.py** (lines 233-284): Modified `validate()` method to accept `save_logits`/`save_groups` parameters and save `test_logits.npy` and `test_groups.npy`
2. **CODES/evaluation/compare.py** (lines 43-46): Fixed `load_baseline_results()` bug - now checks for "mean" key in nested aggregate format `{"metric": {"mean": X, "std": Y, "values": [...]}}`

---

## Critical Blockers for Statistical Analysis

### HSIDataset Class Does NOT Store Group Information
**File:** `CODES/data_preprocessing_corrected.py` (lines ~1000-1016)

```python
class HSIDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)  # NHWC -> NCHW
        self.y = torch.LongTensor(y)
        self.augment = augment
    # NO group information stored!
```

This blocks `test_groups.npy` generation. The group information must be extracted from the test dataset created by `load_tomato_dataset_corrected()` using the split manifest, NOT from HSIDataset.

### Function: `load_tomato_dataset_corrected()`
- Creates group-wise splits at image level before patch extraction
- Returns split manifest with test_groups
- Test groups available in: `split_manifest['test_groups']`

---

## Next Steps (Priority Order)

### Step 1: Re-evaluate mlp_mixer_hsi Seeds 42, 123, 456
- Load each seed's `best_model.pth`
- Run inference on test set with conditional binary evaluation
- Save complete `test_results.json` with all metrics
- Save `test_logits.npy` and `test_groups.npy`

### Step 2: Generate test_logits.npy and test_groups.npy for ALL seeds/models
- Need to extract group info from split manifest, not HSIDataset
- Run validation with `save_logits=True, save_groups=True`
- For simple_cnn: also fix conditional binary evaluation

### Step 3: Re-run compute_aggregate_results() for affected models
- mlp_mixer_hsi (after step 1)
- simple_cnn (after fixing evaluation)
- resnet18_hsi (verify completeness)

### Step 4: Run Statistical Analysis
- Group-aware McNemar's test (pairwise model comparison)
- Group-aware bootstrap CI (1000 resamples of test GROUPS)

### Step 5: Regenerate Final PHASE5_BASELINE_RESULTS.md
- With corrected data and statistical analysis

---

## Commands to Resume Work

```bash
cd C:\Users\ayush\OneDrive\Desktop\DMLPFFN\DMLPFFN-TomatoLeafSpot-Detection-main

# Check current state
python -c "from CODES.run_baselines import *; print('Imports OK')"

# Run specific re-evaluation for mlp_mixer_hsi seeds 42, 123, 456
# (Need to create a script for this)
```

---

## Model Specifications Reference

| Model | Parameters | Input Format | VRAM Estimate |
|-------|------------|--------------|---------------|
| SimpleCNN | 86,156 | NHWC [B,9,9,128] -> NCHW | ~1.2 GB |
| ResNet-18 | 11,242,884 | NHWC [B,9,9,128] -> NCHW | ~3.5 GB |
| MLP-Mixer | 2,310,812 | NHWC [B,9,9,128] -> [B,81,256] | ~2.8 GB |
| SVM (RBF) | N/A | Flattened [N,10368] -> PCA(256) | CPU |
| Random Forest | N/A | Flattened [N,10368] -> PCA(256) | CPU |
| DMLPFFN (ref) | 447,780 | NCHW [B,128,9,9] | ~1.5 GB |

---

## Notes for Next Session

1. **Start with mlp_mixer_hsi seeds 42, 123, 456** - they have trained models, just need re-evaluation
2. **Extract group info from split manifest** - not from HSIDataset
3. **Fix simple_cnn conditional binary evaluation** - macro metrics are all 0
4. **All statistical analysis blocked** until test_logits.npy and test_groups.npy exist
5. **The split_manifest.json is the source of truth** for test groups