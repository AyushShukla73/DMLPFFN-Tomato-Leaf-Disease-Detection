# Dataset Download Instructions

## USDA Tomato Leaf Hyperspectral Dataset

**DOI:** 10.15482/USDA.ADC/26046328  
**Dataset:** Tomato bacterial leaf spot hyperspectral imaging data

### Download Steps

1. Visit the USDA Ag Data Commons: https://data.nal.usda.gov/dataset/tomato-bacterial-leaf-spot-hyperspectral-imaging-data
2. Download the dataset (65 hyperspectral files total)
3. Extract to `tomato_leaf/` directory in the project root:
   ```
   DMLPFFN-TomatoLeafSpot-Detection-main/
   ├── tomato_leaf/
   │   ├── 0dai_2hr_swir_bc_run1.hdr
   │   ├── 0dai_2hr_swir_bc_run1.img
   │   ├── ... (65 total files)
   ```

### Dataset Structure (from EDA.ipynb analysis)

**Two sensor configurations:**
- **SWIR sensor:** ~168 bands, ~320×500-700 spatial pixels (~30 files)
- **VIS-NIR sensor:** ~300 bands, ~900×400-1700 spatial pixels (~35 files)

**File naming pattern:** `{dai}dai_{time}hr_{sensor}_{bc/run}_run{num}.{hdr/img}`
- `dai`: Days After Inoculation (0, 1, 2, 3, 4, 5, 6, 7)
- `sensor`: `swir` or `visnir`
- Classes mapped from DAI:
  - Healthy: 0 DAI
  - Asymptomatic: 1 DAI
  - Early Infection: 2-3 DAI
  - Late Infection: 4-7 DAI

### Required File Format

Each hyperspectral sample consists of:
- `.hdr` - ENVI header file with metadata
- `.img` - Binary image data (band-interleaved-by-line or similar)

### After Download

Once the dataset is in place, run the corrected experiment:

```bash
cd CODES
python main_experiment_corrected.py
```

This will:
1. Apply corrected preprocessing (no data leakage)
2. Train DMLPFFN with reproducible settings
3. Evaluate on held-out test set
4. Save artifacts to `../artifacts/`
5. Produce scientifically valid test accuracy for comparison with original 0.9718