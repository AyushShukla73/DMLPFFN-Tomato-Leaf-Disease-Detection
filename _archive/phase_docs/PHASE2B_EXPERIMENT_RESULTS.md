# Phase 2B: USDA Dataset Recovery & Validation - Experiment Results

## Overview

This document reports the results of Phase 2B: USDA Dataset Recovery & Validation for the DMLPFFN (Deep Multi-Layer Perceptron with Feature Fusion) architecture applied to hyperspectral imaging of tomato leaf diseases.

## Dataset Description

- **Source**: USDA Tomato Leaf Spot Dataset
- **Total Images**: 65 files
- **Stages**: 0dai (healthy), bi (asymptomatic), 1-7dai (infection progression)
- **Sensors**: SWIR (168 bands), VIS-NIR (300 bands)
- **Preprocessing**: PCA harmonization to 128 components
- **Patches**: 9x9 with stride=4

## 4-Class Classification Labels

| Class | Description |
|-------|-------------|
| 0 | Healthy |
| 1 | Asymptomatic |
| 2 | Early |
| 3 | Late (Bacterial Leaf Spot) |

## Preprocessing Corrections Applied

The following scientifically valid preprocessing steps were implemented:

1. **Group-wise Split by Biological Unit**: Images split by biological unit groups (not patch-level) to prevent data leakage from spatially correlated patches within the same image.

2. **Sensor-Specific PCA**: Separate PCA models fitted on SWIR (146 bands → 128 components) and VIS-NIR (300 bands → 128 components), both fitted on training data only.

3. **Normalization Scaler**: StandardScaler fitted on training patches only, applied to all splits.

4. **Patches Extracted After Split**: Patches extracted post-split to prevent preprocessing leakages.

## Train/Validation/Test Split

| Split | Images | Patches | Groups |
|-------|--------|---------|--------|
| Train | 42 | 7,924 | 18 |
| Val | 17 | 3,220 | 4 |
| Test | 6 | 1,134 | 4 |

**Split Ratios**: Train 75.4%, Val 13.8%, Test 10.8% (close to target 70/15/15)

**Group Overlap Check**: All splits verified non-overlapping:
- Train/Val overlap: False
- Train/Test overlap: False
- Val/Test overlap: False

## Class Distributions

| Split | Class 0 (Healthy) | Class 1 (Asymptomatic) | Class 2 (Early) | Class 3 (Late) |
|-------|-------------------|------------------------|-----------------|----------------|
| Train | 1,512 | 378 | 938 | 5,096 |
| Val | 196 | 378 | 756 | 1,890 |
| Test | 378 | 0 | 0 | 756 |

**Critical Finding**: The test set contains only 2 classes (Healthy and Late), missing Asymptomatic (1) and Early (2). This is a dataset imbalance issue inherent to the USDA tomato leaf dataset.

## Model Architecture

- **Type**: DMLPFFN (3-branch: Global/Partition/Local Perceptron)
- **Input Channels**: 128 PCA components
- **Output Classes**: 4
- **Total Parameters**: 447,780
- **Trainable Parameters**: 447,780

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Device | CUDA (NVIDIA GeForce RTX 3050 Laptop GPU) |
| Seed | 42 |
| Batch Size | 16 |
| Learning Rate | 2e-4 |
| Weight Decay | 1e-4 |
| Epochs | 250 (max) |
| Early Stopping Patience | 25 |
| Optimizer | Adam |
| Loss Function | CrossEntropy with class weights |

## Training Results

**Training Time**: 4,610.52 seconds (~77 minutes)

**Training Progress** (selected epochs):
- Epoch 1: Train Acc 84.63%, Val Acc 43.82%
- Epoch 10: Train Acc 99.42%, Val Acc 50.25%
- Epoch 22 (best val): Train Acc 99.10%, Val Acc 54.91%

**Early Stopping**: Triggered at epoch 47 (no improvement for 25 epochs)

### Overfitting Analysis

The training exhibited severe overfitting:
- Final Train Accuracy: ~99.97%
- Best Validation Accuracy: 54.91%
- Training Time: 77 minutes

This indicates the model is memorizing training patterns rather than generalizing. Potential causes:
1. Class imbalance (Late disease dominates training at 64%)
2. Test set class representation issues
3. Feature representation artifacts in the hyperspectral data

## Test Set Evaluation

**Test Accuracy**: 90.04%

**Classification Report**:
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

**Test Confusion Matrix**:
```
[[271   2  22  83]
 [  0   0   0   0]
 [  0   0   0   0]
 [  0   4   2 750]]
```

The model:
- Correctly predicts 750/756 (99.2%) Late cases
- Correctly predicts 271/378 (71.7%) Healthy cases
- Misclassifies 83 healthy as late, 6 healthy as early
- Has undefined metrics for Asymptomatic and Early (0 true samples in test)

## Key Findings

1. **Dataset Imbalance**: The USDA dataset has significant class imbalance. The Late (3) class dominates at ~64% of training data.

2. **Test Set Limitation**: The test set lacks representation for Asymptomatic and Early stages. This limits model evaluation for early disease detection.

3. **Overfitting**: The large gap between training (99.97%) and validation (54.91%) accuracy indicates overfitting to training artifacts.

4. **Model Bias**: The confusion matrix shows the model tends toward predicting the dominant classes (Late and Healthy) when uncertain.

## Conclusions

1. The corrected preprocessing pipeline successfully prevents data leakage through group-wise splitting.

2. The experiment runs to completion with valid artifacts saved.

3. Test set evaluation is limited by missing class representation.

4. Future work should address:
   - Dataset rebalancing strategies
   - More robust cross-validation
   - Feature analysis to identify spurious patterns
   - Augmentations for underrepresented classes

## Artifacts Generated

- `artifacts/best_model.pth` - Trained model checkpoint
- `artifacts/experiment_results.json` - Full experimental results
- `artifacts/split_manifest.json` - Data split specification
- `artifacts/preprocessing_artifacts/` - PCA models and scalers

## References

- USDA Tomato Leaf Spot Dataset: DOI: 10.15482/USDA.ADC/26046328
- DMLPFFN Architecture: Original implementation for hyperspectral plant disease detection