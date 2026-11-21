# Phase 5 Execution Plan: Baseline Experiments & Comparative Evaluation
## DMLPFFN Tomato Leaf Disease Detection

**Date**: 2026-11-14  
**Status**: PLAN - Awaiting Approval  
**Constraint**: NO training until plan approved

---

## 1. Frozen Evaluation Protocol (Immutable)

### 1.1 Fixed Data Split
**Source**: `artifacts/split_manifest.json` (seed=42) — **NEVER MODIFY**

| Split | Images | Patches | Groups | Classes Present |
|-------|--------|---------|--------|-----------------|
| Train | 52 | 9,254 | 18 | Healthy(0): 1,316, Asymptomatic(1): 756, Early(2): 1,512, Late(3): 5,670 |
| Val | 10 | 1,694 | 4 | Healthy(0): 196, Early(2): 182, Late(3): 1,316 |
| Test | 7 | 1,330 | 4 | **Healthy(0): 574, Late(3): 756** |

**Critical**: Test set contains ONLY 2 classes (Healthy, Late). Asymptomatic & Early have 0 support → 4-class metrics mathematically invalid.

### 1.2 Fixed Preprocessing Pipeline (Immutable)
All baselines MUST use identical preprocessing:

| Step | Method | Fitted On | Output |
|------|--------|-----------|--------|
| 1. Band Removal | Remove SWIR bands 94-115 (water absorption) | N/A | SWIR: 146 bands, VIS-NIR: 300 bands |
| 2. PCA | **IncrementalPCA** (sensor-specific, n_components=128) | **Training patches ONLY** | 128 components per sensor |
| 3. Scaling | **StandardScaler** | **Training patches ONLY** | 128-dim z-scored |
| 4. Patch Extraction | 9×9 patches, stride=4 | **AFTER group-wise split** | [N, 9, 9, 128] |

**Artifacts (immutable)**:
- `preprocessing_artifacts_sklearn.joblib`: `pca_swir_168` (IncrementalPCA, 128 comp), `pca_visnir_300` (IncrementalPCA, 128 comp), `scaler` (StandardScaler, 128 features)
- `preprocessing_artifacts_arrays.npz`: wavelengths, removed band indices

### 1.3 Evaluation Protocol: Two-Class Only (Conditional Binary)

**Decision**: Use **Conditional Binary Evaluation** — extract Healthy (class 0) vs Late (class 3) predictions from the 4-class model head. This matches Phase 3 methodology and enables fair comparison with DMLPFFN's reported 95.64% test accuracy.

| Metric | Definition | Valid For |
|--------|------------|-----------|
| **Accuracy** | (TP₀ + TP₃) / N_test | ✅ Test |
| **Macro Precision/Recall/F1** | Mean of per-class (Healthy, Late) | ✅ Test, Val |
| **Weighted Precision/Recall/F1** | Support-weighted mean | ✅ Test, Val |
| **Per-class Precision/Recall/F1** | Binary for each class | ✅ Healthy, Late only |
| **Confusion Matrix** | 2×2 (Healthy vs Late) | ✅ Test |
| **AUC-ROC** | One-vs-Rest for each class | ✅ Test, Val |

**FORBIDDEN**: 4-class macro metrics (2 classes undefined), reporting "Asymptomatic" or "Early" test metrics, retraining binary heads (breaks Phase 3 comparability).

**Rationale**: 
- Phase 3 DMLPFFN used 4-class head with conditional binary evaluation
- Retraining binary heads would change feature learning and break fair comparison
- Group-wise split guarantees Asymptomatic/Early have 0 test support — mathematically impossible to evaluate

### 1.4 Training Configuration (Fixed for All Baselines)
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

---

## 2. Baseline Model Architectures

### 2.1 Input Tensor Specification
All models receive: **`[Batch, 9, 9, 128]`** (H, W, C) — **NHWC format** matching Phase 3 data loader.

**Adaptation Strategy**:
- **CNNs**: Use as `[Batch, 128, 9, 9]` (NCHW) via `.permute(0, 3, 1, 2)`
- **MLP-Mixer**: Flatten to `[Batch, 9*9*128]` = `[Batch, 10368]` or patch tokens `[Batch, 81, 128]`
- **SVM/RF**: Flatten to `[Batch, 10368]` (or PCA-reduced)

### 2.2 Baseline 1: SimpleCNN (HSI-Adapted) — **VALIDATED**
```python
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=128, num_classes=4, base_channels=44):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128 -> 44
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # Block 2: 44 -> 88
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(base_channels * 2, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.features(x).flatten(1)
        return self.classifier(x)
```
**Parameters**: 86,156 (base_channels=44) | **VRAM**: ~1.2 GB | **Target**: Lightweight spectral-spatial baseline  
**Validation**: ✅ Forward pass (NHWC [2,9,9,128] → [2,4]), ✅ Gradient flow (grad_norm=20.23), ✅ Param count matches expected

### 2.3 Baseline 2: ResNet-18 (128-Channel Adapted) — **VALIDATED**
```python
class ResNet18_HSI(nn.Module):
    def __init__(self, in_channels=128, num_classes=4):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.backbone(x)
```
**Parameters**: 11,242,884 | **VRAM**: ~3.5 GB | **Target**: Strong CNN baseline with residual connections  
**Validation**: ✅ Forward pass (NHWC [2,9,9,128] → [2,4]), ✅ Gradient flow (grad_norm=521.55), ✅ Param count matches expected (~11.2M)

### 2.4 Baseline 3: MLP-Mixer (Patch-Based) — **VALIDATED (CORRECTED)**
```python
class MLPMixer_HSI(nn.Module):
    def __init__(
        self,
        in_channels=128,
        num_classes=4,
        patch_size=9,
        embedding_dim=256,
        num_blocks=8,
        token_mixing_dim=128,
        channel_mixing_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = patch_size * patch_size  # 81
        self.embedding_dim = embedding_dim

        # Patch embedding: project 128 channels -> embedding_dim per spatial location
        self.patch_embed = nn.Linear(in_channels, embedding_dim)

        # Mixer blocks with proper token-mixing and channel-mixing
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=self.num_tokens,
                embedding_dim=embedding_dim,
                token_mixing_dim=token_mixing_dim,
                channel_mixing_dim=channel_mixing_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        # Final LayerNorm and classification head
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)
```
**Parameters**: 2,310,812 (embedding_dim=256, blocks=8, token_mixing=128, channel_mixing=512) | **VRAM**: ~2.8 GB | **Target**: Pure MLP baseline, no conv inductive bias  
**Corrections Applied**: 
- Token-mixing MLP operates across spatial tokens (N=81), not channels
- Channel-mixing MLP operates across embedding dimension (C=256)
- Separate LayerNorm + residual connections for each mixing type
- Reduced embedding_dim from 512→256, token_mixing_dim 256→128, channel_mixing_dim 2048→512
**Validation**: ✅ Forward pass (NHWC [2,9,9,128] → [2,4]), ✅ Token mixing verified (output varies), ✅ Gradient flow (grad_norm=138.68), ✅ Param count 2.3M (was 17.2M before fix)

### 2.5 Baseline 4: SVM (RBF Kernel) on Flattened Features — **VALIDATED**
```python
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Flatten: [N, 9, 9, 128] -> [N, 10368]
# Pipeline: PCA(256) → StandardScaler → SVC(RBF, C=10, gamma='scale', class_weight='balanced')
svm_pipeline = Pipeline([
    ('pca', PCA(n_components=256, random_state=42)),
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=42))
])
```
**Training Data**: 9,254 patches (train only) | **Target**: Classical ML baseline  
**Validation**: ✅ Pipeline structure correct, ✅ Save/load with joblib works, ✅ Forward pass works

### 2.6 Baseline 5: Random Forest on Flattened Features — **VALIDATED**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

rf_pipeline = Pipeline([
    ('pca', PCA(n_components=256, random_state=42)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42))
])
```
**Training Data**: 9,254 patches (train only) | **Target**: Ensemble baseline  
**Validation**: ✅ Pipeline structure correct, ✅ Save/load with joblib works, ✅ Forward pass works

### 2.7 Reference Model: DMLPFFN (Phase 3 Architecture) — **VALIDATED**
```python
# From CODES/dmlpffn_disease.py
class DMLPFFN(nn.Module):
    def __init__(self, in_channels=128, num_classes=4, dropout=0.7):
        super().__init__()
        # Low-level branch: Conv(128→16) + DMLPBlock(16) + Conv(16→64)
        # Mid-level branch:  Conv(128→32) + DMLPBlock(32) + Conv(32→64)
        # High-level branch: DMLPBlock(128→64) + Conv(64→64)
        # Fusion: sum branches → GAP → Dropout(0.7) → Linear(64→4)
```
**Parameters**: 447,780 | **VRAM**: ~1.5 GB | **Target**: Proposed method from Phase 3  
**Input Format**: NCHW [B, 128, 9, 9] (different from baselines!)  
**Phase 3 Results**: Val Acc 78.75%, Test Acc 95.64% (2-class: Healthy/Late only)  
**Validation**: ✅ Forward pass (NCHW [2,128,9,9] → [2,4]), ✅ Gradient flow (grad_norm=192.62), ✅ Param count 447.8K matches

---

## 3. Training Configurations Per Baseline

| Baseline | Epochs | Batch | LR | Weight Decay | Scheduler | Augmentation | Class Balancing |
|----------|--------|-------|----|--------------|-----------|--------------|-----------------|
| SimpleCNN | 250 | 16 | 2e-4 | 5e-4 | CosineAnnealWarm(T_0=10) | Mixup 0.2, CutMix 1.0 (50%) | WeightedRandomSampler |
| ResNet-18 | 250 | 16 | 2e-4 | 5e-4 | CosineAnnealWarm(T_0=10) | Mixup 0.2, CutMix 1.0 (50%) | WeightedRandomSampler |
| MLP-Mixer | 250 | 16 | 2e-4 | 5e-4 | CosineAnnealWarm(T_0=10) | Mixup 0.2, CutMix 1.0 (50%) | WeightedRandomSampler |
| SVM | N/A | N/A | N/A | N/A | N/A | N/A | class_weight='balanced' |
| Random Forest | N/A | N/A | N/A | N/A | N/A | N/A | class_weight='balanced' |

**Note**: Deep learning baselines use IDENTICAL Phase 3 regularization for fair comparison.

---

## 4. Compute Budget Estimation (RTX 3050 4GB VRAM)

| Baseline | Params | Est. VRAM | Est. Time/Epoch | Est. Total (250 ep) | Feasibility |
|----------|--------|-----------|-----------------|---------------------|-------------|
| SimpleCNN | 85K | 1.2 GB | ~15 sec | ~62 min | ✅ Easy |
| ResNet-18 | 11.2M | 3.5 GB | ~45 sec | ~3.1 hr | ⚠️ Tight (batch 16 OK) |
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

## 5. Evaluation Protocol

### 5.1 Metrics to Report (Per Baseline)
| Metric | Val | Test |
|--------|-----|------|
| Accuracy (2-class) | ✅ | ✅ |
| Macro Precision (Healthy, Late) | ✅ | ✅ |
| Macro Recall (Healthy, Late) | ✅ | ✅ |
| Macro F1 (Healthy, Late) | ✅ | ✅ |
| Weighted Precision | ✅ | ✅ |
| Weighted Recall | ✅ | ✅ |
| Weighted F1 | ✅ | ✅ |
| Per-class Precision/Recall/F1 | ✅ | ✅ |
| Confusion Matrix (2×2) | ✅ | ✅ |
| Train/Val Gap (same epoch) | ✅ | N/A |
| Best Val Epoch | ✅ | N/A |
| Parameters | ✅ | N/A |
| Inference Time (ms/sample) | N/A | ✅ |

### 5.2 Statistical Rigor (Group-Aware)
| Requirement | Specification |
|-------------|---------------|
| **Seeds** | 5 seeds: [42, 123, 456, 789, 999] |
| **Confidence Intervals** | 95% CI via **group-aware bootstrap** (1000 resamples) on test set — resample test GROUPS (runs), not patches |
| **Significance Test** | McNemar's test for pairwise model comparison (α=0.05) — **group-aware contingency tables** |
| **Reporting** | Mean ± std across 5 seeds + 95% group-aware CI |
| **Group Structure** | Test set: 7 images (groups), 1,330 patches → bootstrap must resample at group level |

**Implementation**: `CODES/evaluation/statistical_tests.py` with:
- `group_bootstrap_ci(y_true, y_pred, groups, n_resamples=1000, confidence=0.95)` — resamples groups
- `mcnemar_group_aware(y_true, pred_a, pred_b, groups)` — contingency table per group
- `compare_models_seeds(results_dict, groups)` — aggregates 5 seeds with group-aware stats

### 5.3 Comparison Matrix (Final Output)
| Model | Val Acc | Test Acc | Macro F1 | Weighted F1 | Params | Train/Val Gap | Inference (ms) |
|-------|---------|----------|----------|-------------|--------|---------------|----------------|
| DMLPFFN (Phase 3) | 78.75% | 95.64% | 0.9660 | 0.9658 | 448K | 11.08 pp | TBD |
| SimpleCNN | TBD | TBD | TBD | TBD | 85K | TBD | TBD |
| ResNet-18 | TBD | TBD | TBD | TBD | 11.2M | TBD | TBD |
| MLP-Mixer | TBD | TBD | TBD | TBD | 2.1M | TBD | TBD |
| SVM | N/A | TBD | TBD | TBD | N/A | N/A | TBD |
| Random Forest | N/A | TBD | TBD | TBD | N/A | N/A | TBD |

---

## 6. Implementation Plan

### 6.1 File Structure
```
CODES/
├── baselines/
│   ├── __init__.py
│   ├── simple_cnn.py
│   ├── resnet18_hsi.py
│   ├── mlp_mixer_hsi.py
│   ├── classical_ml.py      # SVM, RF
│   └── trainer.py           # Unified training loop
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # 2-class metrics only
│   ├── statistical_tests.py # McNemar, bootstrap CI
│   └── compare.py           # Comparison table generation
├── run_baselines.py         # Main orchestrator
└── config_baselines.py      # Baseline configs
```

### 6.2 Execution Order
1. **Implement** all baseline architectures in `CODES/baselines/`
2. **Implement** unified trainer with Phase 3 config
3. **Implement** 2-class evaluation + statistical tests
4. **Run** SimpleCNN (5 seeds) → verify pipeline
5. **Run** MLP-Mixer (5 seeds)
6. **Run** ResNet-18 (5 seeds) — monitor VRAM
7. **Run** SVM + RF (single run, deterministic)
8. **Generate** comparison tables + statistical analysis
9. **Produce** PHASE5_BASELINE_RESULTS.md

### 6.3 Artifact Outputs (Per Baseline)
```
artifacts/baselines/
├── simple_cnn/
│   ├── seed_42/
│   │   ├── best_model.pth
│   │   ├── history.json
│   │   └── test_results.json
│   ├── seeds_123_456_789_999/...
│   └── aggregate_results.json
├── resnet18_hsi/...
├── mlp_mixer_hsi/...
├── svm/
│   ├── model.joblib
│   └── test_results.json
└── rf/...
```

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ResNet-18 OOM on 4GB VRAM | Batch 8 + grad accum 2; AMP; fallback to ResNet-10 |
| Training time too long | Early stop patience=25; max 250 epochs; parallel CPU jobs |
| 4-class metrics temptation | Hard-code 2-class evaluation; assert test classes == [0, 3] |
| Split drift | Load split_manifest.json directly; never re-split |
| Preprocessing mismatch | Load sklearn artifacts directly; no refitting |

---

## 8. Approval Required

**This plan must be approved before ANY training begins.**

### Checklist for Approval:
- [ ] Frozen split_manifest.json confirmed as immutable reference
- [ ] 2-class evaluation protocol accepted (Healthy vs Late only)
- [ ] 5 baseline architectures approved
- [ ] Phase 3 regularization config accepted for all DL baselines
- [ ] 5-seed statistical rigor accepted
- [ ] Compute budget feasible on RTX 3050 4GB
- [ ] Artifact output structure approved

---

**Next Step**: Upon approval, implement baselines in `CODES/baselines/` and execute in order.

*Plan generated: 2026-11-14*