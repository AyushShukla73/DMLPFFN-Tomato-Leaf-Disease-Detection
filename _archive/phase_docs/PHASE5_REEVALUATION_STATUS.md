# Phase 5 Re-evaluation Status Report

**Date**: 2026-11-16  
**Status**: COMPLETE  
**Script**: `CODES/evaluation/re_evaluate_saved_models.py`  
**Data Split**: `artifacts/split_manifest_frozen.json` (seed=42, frozen)  
**Evaluation Protocol**: Conditional Binary (Healthy vs Late from 4-class head)

---

## Summary

All 15 DL baseline model checkpoints (3 models × 5 seeds) have been successfully re-evaluated on the frozen test set. Aggregate results computed and saved. Statistical analysis (group-aware bootstrap CIs + pairwise McNemar's tests) completed.

---

## Re-evaluation Results by Model

### SimpleCNN (5 seeds)

| Seed | Test Accuracy | Macro F1 | Weighted F1 | Group Accuracy | Status |
|------|---------------|----------|-------------|----------------|--------|
| 42   | 0.4549        | 0.3819   | 0.4110      | 0.5000         | ✅ Complete |
| 123  | 0.5955        | 0.4633   | 0.4998      | 0.5000         | ✅ Complete |
| 456  | 0.5534        | 0.4003   | 0.4417      | 0.5000         | ✅ Complete |
| 789  | 0.4316        | 0.3015   | 0.2602      | 0.5000         | ✅ Complete |
| 999  | 0.6105        | 0.4660   | 0.5040      | 0.5000         | ✅ Complete |

**Aggregate (Mean ± Std)**:
- Accuracy: 0.5292 ± 0.0730
- Macro F1: 0.4026 ± 0.0606
- Weighted F1: 0.4233 ± 0.0888
- Group Accuracy: 0.5000 ± 0.0000
- Parameters: 86,156

**Artifacts saved to**: `artifacts/baselines/simple_cnn/aggregate_results.json`

---

### MLP-Mixer HSI (5 seeds)

| Seed | Test Accuracy | Macro F1 | Weighted F1 | Group Accuracy | Status |
|------|---------------|----------|-------------|----------------|--------|
| 42   | 0.5977        | 0.5306   | 0.5549      | 0.5000         | ✅ Complete |
| 123  | 0.5023        | 0.4291   | 0.4012      | 0.5000         | ✅ Complete |
| 456  | 0.5519        | 0.3920   | 0.4347      | 0.5000         | ✅ Complete |
| 789  | 0.2714        | 0.2699   | 0.2745      | 0.0000         | ✅ Complete |
| 999  | 0.3962        | 0.3724   | 0.3892      | 0.0000         | ✅ Complete |

**Aggregate (Mean ± Std)**:
- Accuracy: 0.4639 ± 0.1173
- Macro F1: 0.3988 ± 0.0844
- Weighted F1: 0.4109 ± 0.0900
- Group Accuracy: 0.3000 ± 0.2739
- Parameters: 2,310,812

**Artifacts saved to**: `artifacts/baselines/mlp_mixer_hsi/aggregate_results.json`

---

### ResNet-18 HSI (5 seeds)

| Seed | Test Accuracy | Macro F1 | Weighted F1 | Group Accuracy | Status |
|------|---------------|----------|-------------|----------------|--------|
| 42   | 0.4316        | 0.3015   | 0.2602      | 0.5000         | ✅ Complete |
| 123  | 0.4368        | 0.3160   | 0.2767      | 0.5000         | ✅ Complete |
| 456  | 0.5684        | 0.3624   | 0.4120      | 0.5000         | ✅ Complete |
| 789  | 0.4150        | 0.3606   | 0.3351      | 0.5000         | ✅ Complete |
| 999  | 0.4316        | 0.3015   | 0.2602      | 0.5000         | ✅ Complete |

**Aggregate (Mean ± Std)**:
- Accuracy: 0.4567 ± 0.0563
- Macro F1: 0.3284 ± 0.0276
- Weighted F1: 0.3088 ± 0.0585
- Group Accuracy: 0.5000 ± 0.0000
- Parameters: 11,242,884

**Artifacts saved to**: `artifacts/baselines/resnet18_hsi/aggregate_results.json`

---

## Artifacts Generated Per Seed

For each model/seed combination, the following files were saved to `artifacts/baselines/<model>/seed_<seed>/`:

| File | Description |
|------|-------------|
| `test_logits.npy` | Raw 4-class logits [N, 4] |
| `test_targets.npy` | Original targets [N] (0=Healthy, 3=Late) |
| `test_groups.npy` | Group index per patch [N] (0-3) |
| `test_results.json` | Complete evaluation metrics (patch-level + group-level) |

---

## Aggregate Results Location

| Model | Path |
|-------|------|
| SimpleCNN | `artifacts/baselines/simple_cnn/aggregate_results.json` |
| MLP-Mixer HSI | `artifacts/baselines/mlp_mixer_hsi/aggregate_results.json` |
| ResNet-18 HSI | `artifacts/baselines/resnet18_hsi/aggregate_results.json` |

---

## Statistical Analysis

Completed via `CODES/evaluation/run_statistical_analysis.py`:

- **Group-aware Bootstrap CIs** (1000 resamples, 4 test groups): Saved to `artifacts/baselines/statistical_analysis.json`
- **Pairwise McNemar's Tests** (group-aware): All 3 pairwise comparisons computed
- **Note**: Only 4 test groups → very low statistical power; results are exploratory only

---

## Issues Fixed During Re-evaluation

1. **Import error**: `from evaluation.metrics import compute_conditional_binary_metrics` → Fixed by using `evaluate_model_conditional_binary` from metrics.py
2. **Model class mismatch**: Script now uses actual model classes from `CODES/baselines/` (SimpleCNN, MLPMixer_HSI, ResNet18_HSI) matching trained checkpoints
3. **compare.py load_baseline_results**: Fixed REQUIRED_AGG_KEYS from `{"test_accuracy", ...}` to `{"accuracy", ...}` to match re-evaluation aggregate format
4. **Path issues**: Corrected working directory to `DMLPFFN-TomatoLeafSpot-Detection-main` subdirectory

---

## Verification Checklist

- [x] All 15 DL checkpoints re-evaluated (3 models × 5 seeds)
- [x] Conditional binary evaluation applied correctly (Healthy vs Late from 4-class head)
- [x] Group indices extracted from frozen manifest (4 test groups)
- [x] Logits, targets, groups, predictions saved per seed
- [x] Complete metrics (patch-level + group-level) saved per seed
- [x] Aggregate results (mean ± std across 5 seeds) computed and saved
- [x] Statistical analysis (bootstrap CIs + McNemar's) completed
- [x] Final report `PHASE5_BASELINE_RESULTS.md` regenerated with updated results
- [x] SVM/RF results already present in `artifacts/baselines/svm/` and `artifacts/baselines/random_forest/`

---

## Next Steps

1. Review `PHASE5_RESULTS_VERIFICATION.md` for integrity verification
2. Verify `PHASE5_BASELINE_RESULTS.md` is up-to-date
3. All Phase 5 baseline evaluation deliverables complete