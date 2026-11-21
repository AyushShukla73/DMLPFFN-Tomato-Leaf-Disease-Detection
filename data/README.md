# Dataset Documentation

## USDA Hyperspectral Tomato Leaf Dataset

**DOI**: https://doi.org/10.15482/USDA.ADC/26046328

### Dataset Structure

The dataset contains hyperspectral cubes captured at multiple Days After Inoculation (DAI):

| Class | DAI | Description |
|-------|-----|-------------|
| 0 (Healthy) | 0 DAI | Disease-free control |
| 1 (Asymptomatic) | 1 DAI | Early symptoms, no visual lesions |
| 2 (Early Infection) | 2-3 DAI | Visible early lesions |
| 3 (Late Infection) | 4-7 DAI | Advanced disease symptoms |

### Sensor Configuration

- **SWIR Sensor**: 168 spectral bands
- **VIS-NIR Sensor**: 300 spectral bands
- Combined: 468 total bands

### Preprocessed Data (Pipeline Output)

After the standardized preprocessing pipeline (Phase 3):
- **PCA Components**: 128 components per sensor (separate PCA)
- **Patch Size**: 9×9 spatial × 128 spectral = [B, 9, 9, 128] (NHWC format)
- **Stride**: 4 for patch extraction

### Data Split Manifest

**File**: `artifacts/split_manifest_frozen.json`

**Split Details** (seed=42, group-wise by biological unit):

| Split | Images | Patches | Groups |
|-------|--------|---------|--------|
| Train | 39 | 9,254 | 18 |
| Val | 9 | 1,694 | 4 |
| Test | 7 | 1,330 | 4 |

**Test Class Distribution** (1,330 patches):
- Healthy (0): 574 patches
- Late (3): 756 patches
- Asymptomatic (1): 0 patches (excluded from test)
- Early (2): 0 patches (excluded from test)

**Group Labels**:
- Train groups: `_all_bc_run1`, `_2hr_bc_run2-1`, `_bi_bc_run2_1`, ..., `_7_bc_run1`
- Val groups: `_8_bc_run1`, `_bc_run2-2`, `_bi_bc_run2-1`, `_4_bc_run1`
- Test groups: `_bc_run2_sample1-2`, `_2hr_bc_run2-2`, `_5_bc_run1`, `_bi_bc_run2-2`

### Preprocessing Pipeline

1. **Band Removal**: SWIR bands 94-115 removed (water absorption regions)
2. **Sensor-Specific PCA**: IncrementalPCA (n_components=128) fitted per sensor on TRAIN patches only
3. **Normalization**: StandardScaler fitted on TRAIN patches only
4. **Patch Extraction**: 9×9 patches with stride=4, performed AFTER split to prevent leakage

### Artifacts

- `artifacts/split_manifest_frozen.json` - Frozen split definition with verified no train/val/test overlap
- `artifacts/preprocessing_artifacts_sklearn.joblib` - Fitted PCA and scaler objects
- `artifacts/preprocessing_artifacts_arrays.npz` - Training data arrays

### Loading Data

```python
from data_preprocessing_corrected import load_tomato_dataset_corrected

splits = load_tomato_dataset_corrected(
    root="tomato_leaf/",
    patch_size=9,
    stride=4,
    save_manifest_path="artifacts/split_manifest_frozen.json"
)

X_train, y_train = splits['train']  # Shape: (N, 9, 9, 128)
X_val, y_val = splits['val']
X_test, y_test = splits['test']
```

### Data Format

Input tensors are in NHWC format [Batch, Height, Width, Channels]:
- Height: 9 pixels
- Width: 9 pixels
- Channels: 128 PCA components

For PyTorch models expecting NCHW format, transpose:
```python
X_nchw = X.numpy().transpose(0, 3, 1, 2)  # NHWC -> NCHW
```

### Binary Evaluation Protocol

For conditional binary evaluation (Healthy vs Late):
- Binary mapping: CLASS_HEALTHY=0, CLASS_LATE=3 → BINARY_CLASS_MAP={0:0, 3:1}
- Only test samples with labels 0 or 3 are used
- Conditional probabilities computed from 4-class logits:
  - P(Healthy|H/L) = exp(z0)/(exp(z0)+exp(z3))
  - P(Late|H/L) = exp(z3)/(exp(z0)+exp(z3))