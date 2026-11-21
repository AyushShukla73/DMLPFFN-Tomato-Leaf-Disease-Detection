# Phase 5 Interrupted Runs Documentation

**Date:** 2026-11-14  
**Reason:** Emergency power loss — immediate shutdown required  
**GPU State at Shutdown:** NVIDIA RTX 3050 Laptop GPU, 0.00 GB allocated, 0.00 GB reserved (verified clean)

---

## Summary of Completed Work

| Model | Seeds Completed | Seeds Interrupted | Seeds Not Started |
|-------|-----------------|-------------------|-------------------|
| SimpleCNN | 42, 123, 456, 789, 999 (all 5) | — | — |
| ResNet-18 | 42, 123, 456 | — | 789, 999 |
| MLP-Mixer | 456 | 789 (epoch 28/250) | 999 |
| SVM | Complete (all seeds) | — | — |
| Random Forest | Complete (all seeds) | — | — |

---

## MLP-Mixer (mlp_mixer_hsi)

### Seed 456 — COMPLETED
- **Test Accuracy:** 0.9180 (per `test_results.json`), log shows 0.9030
- **Best Val Accuracy:** 0.8406 at epoch 16
- **Training Time:** 25.0 min (1497.1s)
- **Epochs Run:** 41/250 (early stopped at epoch 41, patience=25)
- **Model Parameters:** 2,310,812
- **Checkpoint:** `artifacts/baselines/mlp_mixer_hsi/seed_456/best_model.pt`
- **Logs:** `artifacts/baselines/mlp_mixer_hsi/seed_456/training.log`, `run_missing_456_789_999.log`

### Seed 789 — INTERRUPTED at Epoch 28/250
- **Last Completed Epoch:** 28
- **Best Val Accuracy So Far:** 0.8164 at epoch 26
- **Training Time at Interrupt:** ~53 min (3193s for seed 456 + partial for 789)
- **Checkpoint:** `artifacts/baselines/mlp_mixer_hsi/seed_789/best_model.pt` (saved at epoch 26)
- **Logs:** `run_missing_456_789_999.log` (epochs 1-28 recorded)
- **Resume Command:** `python -m CODES.run_missing_seeds mlp_mixer_hsi 789,999`

### Seed 999 — NOT STARTED
- **Resume Command:** `python -m CODES.run_missing_seeds mlp_mixer_hsi 999` (or combined with 789 above)

---

## ResNet-18 (resnet18_hsi)

### Seeds 42, 123, 456 — COMPLETED
| Seed | Test Acc | Best Val Acc | Best Epoch | Epochs Run |
|------|----------|--------------|------------|------------|
| 42   | 0.9534   | —            | —          | —          |
| 123  | 0.9602   | —            | —          | —          |
| 456  | 0.9045   | —            | —          | —          |

- **Checkpoints:** `artifacts/baselines/resnet18_hsi/seed_{42,123,456}/best_model.pt`
- **Logs:** Individual `training.log` files in each seed directory

### Seeds 789, 999 — NOT STARTED
- **Resume Command:** `python -m CODES.run_missing_seeds resnet18_hsi 789,999`

---

## SimpleCNN (simple_cnn) — ALL COMPLETE
- All 5 seeds [42, 123, 456, 789, 999] completed
- **Checkpoints:** `artifacts/baselines/simple_cnn/seed_{42,123,456,789,999}/best_model.pt`
- **Aggregate:** `artifacts/baselines/simple_cnn/aggregate_results.json` (only has `{"parameters": 86156}` — incomplete)

---

## SVM & Random Forest — COMPLETE
- Both models fully trained and evaluated
- **Aggregate Results:** Complete in `artifacts/baselines/svm/aggregate_results.json` and `artifacts/baselines/random_forest/aggregate_results.json`

---

## Incomplete Aggregate Files (Need Regeneration)

| Model | Current State | Action Needed |
|-------|---------------|---------------|
| simple_cnn | `{"parameters": 86156}` only | Run `compute_aggregate_results()` for all 5 seeds |
| resnet18_hsi | `{}` (empty) | Run `compute_aggregate_results()` for seeds 42,123,456 + 789,999 after they complete |
| mlp_mixer_hsi | `{}` (empty) | Run `compute_aggregate_results()` for seeds 456,789,999 after they complete |

---

## Critical Bug to Fix Before Report Generation

**File:** `CODES/evaluation/compare.py`  
**Function:** `load_baseline_results()` (lines 12-39)

**Problem:** Checks `aggregate_results.json` FIRST. If file exists (even empty `{}`), uses it and NEVER falls back to seed directories. This causes N/A values in the final report.

**Fix Required:** Modify to fall back to seed directories when `aggregate_results.json` is empty or missing required keys (`test_acc_mean`, `test_acc_std`, etc.).

---

## Documentation Fix Remaining

**File:** `CODES/evaluation/compare.py`  
**Line:** 114 (comparison table footer)  
**Current:** "7 images"  
**Should be:** "4 test groups"  
(Already fixed in 4 other locations: compare.py lines 217, 261 and PHASE5_BASELINE_RESULTS.md lines 7, 22)

---

## Resume Commands (Run After Power Restored)

```bash
# 1. Complete MLP-Mixer seeds 789 and 999
cd C:/Users/ayush/OneDrive/Desktop/DMLPFFN/DMLPFFN-TomatoLeafSpot-Detection-main
python -m CODES.run_missing_seeds mlp_mixer_hsi 789,999

# 2. Complete ResNet-18 seeds 789 and 999
python -m CODES.run_missing_seeds resnet18_hsi 789,999

# 3. Regenerate aggregate results for all DL models
python -c "
from CODES.run_baselines import compute_aggregate_results, save_aggregate_results
from CODES.config_baselines import EXPERIMENTS
import json, pathlib

for model in ['simple_cnn', 'resnet18_hsi', 'mlp_mixer_hsi']:
    seeds = EXPERIMENTS[model]['seeds']
    agg = compute_aggregate_results(model, seeds)
    save_aggregate_results(model, agg)
    print(f'{model}: {agg}')
"

# 4. Fix compare.py load_baseline_results() bug, then generate report
python -m CODES.run_baselines --report-only
```

---

## Checkpoint Validity Assessment

| Checkpoint | Status | Notes |
|------------|--------|-------|
| MLP-Mixer seed_456 best_model.pt | **Valid** | Completed training, early stopped at epoch 41 |
| MLP-Mixer seed_789 best_model.pt | **Valid (partial)** | Saved at epoch 26 (best val acc 0.8164). Training can resume from epoch 29. |
| ResNet-18 seeds 42,123,456 best_model.pt | **Valid** | All completed |
| SimpleCNN all seeds best_model.pt | **Valid** | All completed |
| SVM / RF models | **Valid** | Completed, no epochs concept |

**Note:** All checkpoints saved via `torch.save()` with model state_dict, optimizer state, epoch, and best_val_acc. Compatible with `run_baselines.py` resume logic if implemented, or can be used for inference directly.

---

## Log Files Available

- `artifacts/baselines/mlp_mixer_hsi/run_missing_456_789_999.log` — MLP-Mixer seeds 456 (full), 789 (epochs 1-28)
- `artifacts/baselines/resnet18_hsi/run_missing_seeds.log` — ResNet-18 seeds 42,123,456 (if exists)
- Individual `training.log` in each seed directory for all completed seeds
- `artifacts/baselines/*/seed_*/test_results.json` — Per-seed test metrics (accuracy, per-class F1, confusion matrix)

---

## Next Steps Priority

1. **Fix compare.py `load_baseline_results()`** — Critical blocker for report generation
2. **Run MLP-Mixer seeds 789, 999** — Resume from epoch 29 for seed 789
3. **Run ResNet-18 seeds 789, 999** — Fresh training
4. **Regenerate all aggregate_results.json** — Using `compute_aggregate_results()`
5. **Fix remaining "7 images" → "4 test groups" in compare.py line 114**
6. **Generate final PHASE5_BASELINE_RESULTS.md** — Verify no N/A values