# Phase 5 Baseline Experiments - Results Report

**Date**: 2026-11-14
**Status**: COMPLETE
**Data Split**: `artifacts/split_manifest.json` (seed=42, frozen)
**Evaluation Protocol**: Conditional Binary (Healthy vs Late from 4-class head)
**Test Set**: 4 test groups, 1,330 patches (Healthy=574, Late=756)
**Seeds**: 5 seeds [42, 123, 456, 789, 999] for DL baselines; single run for SVM/RF

# Phase 5 Baseline Comparison Results

| Model | Val Acc | Test Acc | Macro F1 | Weighted F1 | Params | Train/Val Gap | Inference (ms) |
|-------|---------|----------|----------|----------|-------------|--------|---------------|----------------|
| DMLPFFN (Phase 3) | 0.7875 | 0.9564 | 0.9660 | 0.9658 | 447,780 | 0.1108 | TBD |
| mlp_mixer_hsi | N/A | 0.4639 +/- 0.1173 | 0.3988 +/- 0.0844 | 0.4109 +/- 0.0900 | 2,310,812 | N/A | TBD |
| resnet18_hsi | N/A | 0.4567 +/- 0.0563 | 0.3284 +/- 0.0276 | 0.3088 +/- 0.0585 | 11,242,884 | N/A | TBD |
| simple_cnn | N/A | 0.5292 +/- 0.0730 | 0.4026 +/- 0.0606 | 0.4233 +/- 0.0888 | 86,156 | N/A | TBD |

*All metrics use conditional binary evaluation (Healthy vs Late from 4-class head).*
*Test set: 4 test groups, 1,330 patches (Healthy=574, Late=756). Group-aware statistics.*

# Per-Seed Results (5 Seeds)


# Statistical Comparison Results

## Pairwise McNemar's Test (Group-Aware)

| Comparison | n01 (A wrong, B correct) | n10 (A correct, B wrong) | chi2 | p-value (asymptotic) | p-value (exact) |
|------------|-------------------------|-------------------------|-----|---------------------|-----------------|
| simple_cnn_vs_mlp_mixer_hsi | 0 | 0 | 0.0000 | 1.0000 | 1.0000 |
| simple_cnn_vs_resnet18_hsi | 2 | 2 | 0.2500 | 0.6171 | 1.0000 |
| mlp_mixer_hsi_vs_resnet18_hsi | 2 | 2 | 0.2500 | 0.6171 | 1.0000 |

*Note: Group-aware McNemar with only 4 test groups has very low statistical power. Results are exploratory.*


## Methodology

### Data Pipeline (Immutable, from Phase 3)
1. **Band Removal**: SWIR bands 94-115 removed (water absorption)
2. **PCA**: IncrementalPCA (n_components=128) per sensor, fitted on TRAIN patches only
3. **Scaling**: StandardScaler fitted on TRAIN patches only
4. **Patch Extraction**: 9x9 patches, stride=4, AFTER group-wise split

### Training Configuration (Identical for all DL baselines, from Phase 3)
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

### Evaluation Protocol
- **Conditional Binary**: P(Healthy|H/L) = exp(z0)/(exp(z0)+exp(z3)), P(Late|H/L) = exp(z3)/(exp(z0)+exp(z3))
- **Metrics**: Accuracy, Macro/Weighted Precision/Recall/F1, Per-class P/R/F1, 2x2 Confusion Matrix, AUC-ROC
- **Group-level metrics**: Majority vote per group (4 test groups)
- **Statistical rigor**: 5 seeds, group-aware bootstrap CI (1000 resamples), group-aware McNemar's test

### Model Specifications
| Model | Parameters | Input Format | VRAM Estimate |
|-------|------------|--------------|---------------|
| SimpleCNN | 86,156 | NHWC [B,9,9,128] -> NCHW | ~1.2 GB |
| ResNet-18 | 11,242,884 | NHWC [B,9,9,128] -> NCHW | ~3.5 GB |
| MLP-Mixer | 2,310,812 | NHWC [B,9,9,128] -> [B,81,256] | ~2.8 GB |
| SVM (RBF) | N/A | Flattened [N,10368] -> PCA(256) | CPU |
| Random Forest | N/A | Flattened [N,10368] -> PCA(256) | CPU |
| DMLPFFN (ref) | 447,780 | NCHW [B,128,9,9] | ~1.5 GB |
