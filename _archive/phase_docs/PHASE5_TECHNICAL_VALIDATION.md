# Phase 5 Technical Validation Report
## DMLPFFN Tomato Leaf Disease Detection - Baseline Experiments

**Date**: 2026-11-14  
**Status**: COMPLETE - Awaiting Approval to Proceed with Training  
**Constraint**: NO training until plan approved

---

## Executive Summary

All 5 baseline models + DMLPFFN reference model have been **implemented, tested, and validated**. Key corrections applied:
- SimpleCNN: Fixed `base_channels=44` → **86,156 params** (was 148K)
- MLP-Mixer: Proper token/channel mixing architecture → **2.3M params** (was 17.2M)
- Two-class evaluation protocol established (Healthy vs Late only)
- All models verified: forward pass, gradient flow, parameter counts, memory estimates

---

## 1. Frozen Data Split Validation (Immutable)

**Source**: `artifacts/split_manifest.json` (seed=42) — **NEVER MODIFY**

| Split | Images | Patches | Groups | Classes Present |
|-------|--------|---------|--------|-----------------|
| Train | 52 | 9,254 | 18 | Healthy(0): 1,316, Asymptomatic(1): 756, Early(2): 1,512, Late(3): 5,670 |
| Val | 10 | 1,694 | 4 | Healthy(0): 196, Early(2): 182, Late(3): 1,316 |
| Test | 7 | 1,330 | 4 | **Healthy(0): 574, Late(3): 756** |

**Critical Finding**: Test set contains ONLY 2 classes (Healthy, Late). Asymptomatic & Early have 0 support → 4-class metrics mathematically invalid.

**Group Overlap Check**: All False (zero leakage confirmed)

---

## 2. Preprocessing Pipeline Validation (Immutable)

| Step | Method | Fitted On | Output |
|------|--------|-----------|--------|
| 1. Band Removal | Remove SWIR bands 94-115 | N/A | SWIR: 146 bands, VIS-NIR: 300 bands |
| 2. PCA | IncrementalPCA (sensor-specific, n=128) | **Training patches ONLY** | 128 components per sensor |
| 3. Scaling | StandardScaler | **Training patches ONLY** | 128-dim z-scored |
| 4. Patch Extraction | 9×9 patches, stride=4 | **AFTER group-wise split** | [N, 9, 9, 128] |

**Artifacts** (immutable, load-only):
- `preprocessing_artifacts_sklearn.joblib`: `pca_swir_168`, `pca_visnir_300`, `scaler`
- `preprocessing_artifacts_arrays.npz`: wavelengths, removed band indices

---

## 3. Baseline Model Validation Results

### 3.1 SimpleCNN (HSI-Adapted) — ✅ VALIDATED

| Property | Value |
|----------|-------|
| **Parameters** | 86,156 (base_channels=44) |
| **Input Format** | NHWC [B, 9, 9, 128] → NCHW via permute |
| **Architecture** | 2 Conv blocks (128→44→88) + AdaptiveAvgPool2d + Linear |
| **VRAM Estimate** | ~1.2 GB (fp32, batch=16) |
| **Forward Pass** | ✅ [2, 9, 9, 128] → [2, 4] |
| **Gradient Flow** | ✅ grad_norm = 20.23 |
| **Target Role** | Lightweight spectral-spatial baseline |

**Correction Applied**: Changed `base_channels=64` → `44` to hit ~85K target (was 148,356)

---

### 3.2 ResNet-18 (128-Channel Adapted) — ✅ VALIDATED

| Property | Value |
|----------|-------|
| **Parameters** | 11,242,884 (~11.2M) |
| **Input Format** | NHWC [B, 9, 9, 128] → NCHW via permute |
| **Architecture** | torchvision ResNet-18 with conv1: kernel=3, stride=1, maxpool=Identity |
| **VRAM Estimate** | ~3.5 GB (fp32, batch=16) |
| **Forward Pass** | ✅ [2, 9, 9, 128] → [2, 4] |
| **Gradient Flow** | ✅ grad_norm = 521.55 |
| **Target Role** | Strong CNN baseline with residual connections |

**VRAM Risk**: ⚠️ Tight on RTX 3050 4GB — mitigation: batch=8 + grad_accum=2, AMP

---

### 3.3 MLP-Mixer (Patch-Based) — ✅ VALIDATED (CORRECTED)

| Property | Value |
|----------|-------|
| **Parameters** | 2,310,812 (~2.3M) |
| **Input Format** | NHWC [B, 9, 9, 128] → flatten to [B, 81, 128] |
| **Architecture** | PatchEmbed(128→256) → 8× MixerBlocks → LayerNorm → GAP → Linear(256→4) |
| **VRAM Estimate** | ~2.8 GB (fp32, batch=16) |
| **Forward Pass** | ✅ [2, 9, 9, 128] → [2, 4] |
| **Token Mixing Verified** | ✅ Output varies across spatial positions |
| **Gradient Flow** | ✅ grad_norm = 138.68 |
| **Target Role** | Pure MLP baseline, no conv inductive bias |

**Corrections Applied** (per user requirements):
- Token-mixing MLP operates across spatial tokens (N=81), NOT channels
- Channel-mixing MLP operates across embedding dimension (C=256)
- Separate LayerNorm + residual connections for each mixing type
- Reduced: `embedding_dim=512→256`, `token_mixing_dim=256→128`, `channel_mixing_dim=2048→512`
- **Before fix**: 17,224,468 params → **After fix**: 2,310,812 params

---

### 3.4 SVM (RBF Kernel) — ✅ VALIDATED

| Property | Value |
|----------|-------|
| **Pipeline** | PCA(256) → StandardScaler → SVC(RBF, C=10, gamma='scale', class_weight='balanced') |
| **Input** | Flattened [N, 10368] |
| **Training Data** | 9,254 patches (train only) |
| **Save/Load** | ✅ joblib works |
| **Target Role** | Classical ML baseline |

---

### 3.5 Random Forest — ✅ VALIDATED

| Property | Value |
|----------|-------|
| **Pipeline** | PCA(256) → StandardScaler → RF(n_estimators=500, max_depth=20, class_weight='balanced') |
| **Input** | Flattened [N, 10368] |
| **Training Data** | 9,254 patches (train only) |
| **Save/Load** | ✅ joblib works |
| **Target Role** | Ensemble baseline |

---

### 3.6 DMLPFFN (Reference Model, Phase 3) — ✅ VALIDATED

| Property | Value |
|----------|-------|
| **Parameters** | 447,780 (~448K) |
| **Input Format** | **NCHW [B, 128, 9, 9]** (DIFFERENT from baselines!) |
| **Architecture** | 3-branch DMLPFFN (low/mid/high) → Fusion → GAP → Dropout(0.7) → Linear |
| **VRAM Estimate** | ~1.5 GB (fp32, batch=16) |
| **Forward Pass** | ✅ [2, 128, 9, 9] → [2, 4] |
| **Gradient Flow** | ✅ grad_norm = 192.62 |
| **Phase 3 Results** | Val Acc 78.75%, Test Acc 95.64% (2-class: Healthy/Late only) |
| **Target Role** | Proposed method from Phase 3 for comparison |

**Note**: DMLPFFN expects NCHW while baselines use NHWC — data loader handles conversion.

---

## 4. Unified Training Configuration (Fixed for All DL Baselines)

| Parameter | Value | Source |
|-----------|-------|--------|
| Batch Size | 16 | Phase 3 config |
| Optimizer | AdamW | Phase 3 config |
| Learning Rate | 2e-4 | Phase 3 config |
| Weight Decay | 5e-4 | Phase 3 config |
| Epochs | 250 (early stop patience=25) | Phase 3 config |
| Scheduler | CosineAnnealingWarmRestarts(T_0=10) | Phase 3 config |
| Label Smoothing | 0.1 | Phase 3 config |
| Mixup α | 0.2 (50% prob) | Phase 3 config |
| CutMix α | 1.0 (50% prob) | Phase 3 config |
| Class Balancing | WeightedRandomSampler | Phase 3 config |
| Seed | 42 (all RNGs) | Phase 3 config |
| Device | CUDA (RTX 3050 4GB) | Hardware |
| Loss | CrossEntropy + label_smoothing | Phase 3 config |

**Implementation**: `CODES/baselines/trainer.py` — `BaselineTrainer` class with `get_phase3_config()`

---

## 5. Evaluation Protocol: Two-Class Only (Conditional Binary)

### Decision: Conditional Binary Evaluation
Extract Healthy (class 0) vs Late (class 3) predictions from the **4-class model head**. This matches Phase 3 methodology and enables fair comparison with DMLPFFN's reported 95.64% test accuracy.

| Metric | Definition | Valid For |
|--------|------------|-----------|
| **Accuracy** | (TP₀ + TP₃) / N_test | ✅ Test |
| **Macro Precision/Recall/F1** | Mean of per-class (Healthy, Late) | ✅ Test, Val |
| **Weighted Precision/Recall/F1** | Support-weighted mean | ✅ Test, Val |
| **Per-class Precision/Recall/F1** | Binary for each class | ✅ Healthy, Late only |
| **Confusion Matrix** | 2×2 (Healthy vs Late) | ✅ Test |
| **AUC-ROC** | One-vs-Rest for each class | ✅ Test, Val |

### FORBIDDEN
- 4-class macro metrics (2 classes undefined)
- Reporting "Asymptomatic" or "Early" test metrics
- Retraining binary heads (breaks Phase 3 comparability)

### Rationale
- Phase 3 DMLPFFN used 4-class head with conditional binary evaluation
- Retraining binary heads would change feature learning and break fair comparison
- Group-wise split guarantees Asymptomatic/Early have 0 test support — mathematically impossible to evaluate

---

## 6. Statistical Rigor (Group-Aware)

| Requirement | Specification |
|-------------|---------------|
| **Seeds** | 5 seeds: [42, 123, 456, 789, 999] |
| **Confidence Intervals** | 95% CI via **group-aware bootstrap** (1000 resamples) on test set — resample test GROUPS (runs), not patches |
| **Significance Test** | McNemar's test for pairwise model comparison (α=0.05) — **group-aware contingency tables** |
| **Reporting** | Mean ± std across 5 seeds + 95% group-aware CI |
| **Group Structure** | Test set: 7 images (groups), 1,330 patches → bootstrap must resample at group level |

**Implementation**: `CODES/evaluation/statistical_tests.py` with:
- `group_bootstrap_ci(y_true, y_pred, groups, n_resamples=1000, confidence=0.95)`
- `mcnemar_group_aware(y_true, pred_a, pred_b, groups)`
- `compare_models_seeds(results_dict, groups)`

---

## 7. Compute Budget Estimation (RTX 3050 4GB VRAM)

| Baseline | Params | Est. VRAM | Est. Time/Epoch | Est. Total (250 ep) | Feasibility |
|----------|--------|-----------|-----------------|---------------------|-------------|
| SimpleCNN | 85K | 1.2 GB | ~15 sec | ~62 min | ✅ Easy |
| ResNet-18 | 11.2M | 3.5 GB | ~45 sec | ~3.1 hr | ⚠️ Tight |
| MLP-Mixer | 2.1M | 2.8 GB | ~35 sec | ~2.4 hr | ✅ OK |
| SVM | N/A | CPU | N/A | ~10 min | ✅ CPU |
| Random Forest | N/A | CPU | N/A | ~15 min | ✅ CPU |

**Total GPU Time**: ~6.1 hours (sequential)  
**Strategy**: Run SimpleCNN → MLP-Mixer → ResNet-18 (increasing VRAM). SVM/RF on CPU in parallel.

**Memory Optimizations if Needed**:
- Reduce batch size to 8 for ResNet-18 (doubles time, halves VRAM)
- Gradient accumulation (2 steps) to simulate batch 16
- Mixed precision (AMP) for all DL models

---

## 8. Comparison Matrix Template (Final Output)

| Model | Val Acc | Test Acc | Macro F1 | Weighted F1 | Params | Train/Val Gap | Inference (ms) |
|-------|---------|----------|----------|-------------|--------|---------------|----------------|
| DMLPFFN (Phase 3) | 78.75% | 95.64% | 0.9660 | 0.9658 | 448K | 11.08 pp | TBD |
| SimpleCNN | TBD | TBD | TBD | TBD | 85K | TBD | TBD |
| ResNet-18 | TBD | TBD | TBD | TBD | 11.2M | TBD | TBD |
| MLP-Mixer | TBD | TBD | TBD | TBD | 2.1M | TBD | TBD |
| SVM | N/A | TBD | TBD | TBD | N/A | N/A | TBD |
| Random Forest | N/A | TBD | TBD | TBD | N/A | N/A | TBD |

---

## 9. File Structure (Implemented)

```
CODES/
├── baselines/
│   ├── __init__.py
│   ├── simple_cnn.py           ✅ Validated
│   ├── resnet18_hsi.py         ✅ Validated
│   ├── mlp_mixer_hsi.py        ✅ Validated (corrected)
│   ├── classical_ml.py         ✅ Validated
│   └── trainer.py              ✅ Validated
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              ⏳ To implement
│   ├── statistical_tests.py    ⏳ To implement
│   └── compare.py              ⏳ To implement
├── run_baselines.py            ⏳ To implement
└── config_baselines.py         ⏳ To implement
```

---

## 10. Risk Mitigation Summary

| Risk | Mitigation |
|------|------------|
| ResNet-18 OOM on 4GB VRAM | Batch 8 + grad accum 2; AMP; fallback to ResNet-10 |
| Training time too long | Early stop patience=25; max 250 epochs; parallel CPU jobs |
| 4-class metrics temptation | Hard-code 2-class evaluation; assert test classes == [0, 3] |
| Split drift | Load split_manifest.json directly; never re-split |
| Preprocessing mismatch | Load sklearn artifacts directly; no refitting |

---

## 11. Approval Checklist

**This plan must be approved before ANY training begins.**

- [ ] Frozen split_manifest.json confirmed as immutable reference
- [ ] 2-class evaluation protocol accepted (Healthy vs Late only)
- [ ] 5 baseline architectures approved (SimpleCNN, ResNet-18, MLP-Mixer, SVM, RF)
- [ ] Phase 3 regularization config accepted for all DL baselines
- [ ] 5-seed statistical rigor accepted (group-aware bootstrap + McNemar)
- [ ] Compute budget feasible on RTX 3050 4GB
- [ ] Artifact output structure approved
- [ ] DMLPFFN included as reference model for comparison

---

## 12. Next Steps (Upon Approval)

1. **Implement** evaluation module: `CODES/evaluation/metrics.py`, `statistical_tests.py`, `compare.py`
2. **Implement** main orchestrator: `CODES/run_baselines.py`, `CODES/config_baselines.py`
3. **Run** SimpleCNN (5 seeds) → verify pipeline end-to-end
4. **Run** MLP-Mixer (5 seeds)
5. **Run** ResNet-18 (5 seeds) — monitor VRAM
6. **Run** SVM + RF (single run, deterministic)
7. **Generate** comparison tables + statistical analysis
8. **Produce** `PHASE5_BASELINE_RESULTS.md`

---

*Validation completed: 2026-11-14*  
*All models tested and validated. Awaiting explicit approval to begin training.*