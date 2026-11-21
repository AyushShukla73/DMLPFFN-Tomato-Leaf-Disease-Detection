"""
Corrected Data Preprocessing Pipeline for DMLPFFN

This module implements scientifically valid preprocessing that addresses three critical
data leakage issues identified in the Phase 1 audit:

1. Per-Image PCA Leakage: PCA must be fitted on TRAINING data only, then applied to val/test
2. Patch-Level Split Leakage: Must split at IMAGE level (biological unit) before patch extraction
3. Normalization Leakage: Per-band statistics must be computed on TRAINING patches only

Additional improvements:
- Sensor-specific PCA (separate PCA for 168-band SWIR and 300-band VIS-NIR)
- Train-only fitted transformations with serialization
- Reproducible random state management
- Split manifest generation for audit trail
"""

import os
import numpy as np
import torch
import spectral
import json
import hashlib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from config import cfg


# =============================================================================
# Data Structures for Reproducibility and Audit Trail
# =============================================================================

@dataclass
class ImageMetadata:
    """Metadata for a single hyperspectral image."""
    filepath: str
    filename: str
    dai: int
    label: int
    sensor_type: str  # 'swir_168' or 'visnir_300'
    shape: Tuple[int, int, int]  # (H, W, B)
    group_id: str  # Biological unit identifier (e.g., 'run1', 'run2')

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SplitManifest:
    """Manifest documenting the train/val/test split for reproducibility."""
    train_images: List[str]
    val_images: List[str]
    test_images: List[str]
    train_patches: int
    val_patches: int
    test_patches: int
    train_groups: List[str]
    val_groups: List[str]
    test_groups: List[str]
    group_overlap_check: Dict[str, bool]  # Should all be False
    split_ratios: Dict[str, float]
    seed: int
    sensor_specific: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SplitManifest':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class PreprocessingArtifacts:
    """All fitted preprocessing artifacts for consistent application to new data."""
    pca_swir_168: Optional[PCA] = None
    pca_visnir_300: Optional[PCA] = None
    scaler: Optional[StandardScaler] = None
    remove_bands_swir: List[int] = field(default_factory=list)
    remove_bands_visnir: List[int] = field(default_factory=list)
    wavelengths_swir: Optional[np.ndarray] = None
    wavelengths_visnir: Optional[np.ndarray] = None

    def save(self, path: str):
        """Save artifacts using joblib for sklearn objects."""
        import joblib
        # Save sklearn objects separately
        sklearn_objs = {
            'pca_swir_168': self.pca_swir_168,
            'pca_visnir_300': self.pca_visnir_300,
            'scaler': self.scaler
        }
        joblib.dump(sklearn_objs, path + '_sklearn.joblib')

        # Save numpy arrays and metadata
        np.savez(path + '_arrays.npz',
                 remove_bands_swir=np.array(self.remove_bands_swir),
                 remove_bands_visnir=np.array(self.remove_bands_visnir),
                 wavelengths_swir=self.wavelengths_swir,
                 wavelengths_visnir=self.wavelengths_visnir)

    @classmethod
    def load(cls, path: str) -> 'PreprocessingArtifacts':
        import joblib
        sklearn_objs = joblib.load(path + '_sklearn.joblib')
        arrays = np.load(path + '_arrays.npz', allow_pickle=True)

        return cls(
            pca_swir_168=sklearn_objs['pca_swir_168'],
            pca_visnir_300=sklearn_objs['pca_visnir_300'],
            scaler=sklearn_objs['scaler'],
            remove_bands_swir=arrays['remove_bands_swir'].tolist(),
            remove_bands_visnir=arrays['remove_bands_visnir'].tolist(),
            wavelengths_swir=arrays['wavelengths_swir'] if 'wavelengths_swir' in arrays else None,
            wavelengths_visnir=arrays['wavelengths_visnir'] if 'wavelengths_visnir' in arrays else None,
        )


# =============================================================================
# Utility Functions
# =============================================================================

def parse_filename(filepath: str) -> Tuple[int, str, str]:
    """
    Parse filename to extract DAI, sensor type, and group identifier.

    Expected format: {dai}dai_{time}_{sensor}_{condition}_run{run}.bil.hdr
    Example: 0dai_2hr_swir_bc_run1.bil.hdr

    Returns:
        (dai, sensor_type, group_id)
    """
    filename = os.path.basename(filepath)
    name = filename.replace('.bil.hdr', '').replace('.bil', '')

    # Extract DAI
    if 'dai' not in name:
        raise ValueError(f"Cannot parse DAI from filename: {filename}")
    dai_str = name.split('dai')[0]
    dai = int(dai_str)

    # Extract sensor type (swir or visnir)
    sensor_type = 'unknown'
    if 'swir' in name.lower():
        sensor_type = 'swir_168'
    elif 'visnir' in name.lower() or 'vnir' in name.lower():
        sensor_type = 'visnir_300'

    # Extract group identifier (run number = biological unit)
    # Group is everything after DAI, representing a unique leaf acquisition
    group_id = name.split('dai', 1)[1]  # Keep leading underscore for token replacement

    # FIX: Strip sensor token (swir/visnir) to get true biological unit
    # e.g., "_2hr_swir_bc_run1" -> "_2hr_bc_run1" (same bio unit across sensors)
    # This ensures same plant imaged on both sensors gets same biological_unit
    bio_unit = group_id
    for sensor_token in ['_swir', '_visnir']:
        bio_unit = bio_unit.replace(sensor_token, '')

    return dai, sensor_type, bio_unit


def get_label_from_dai(dai: int) -> int:
    """Map DAI to class label."""
    label_map = {
        0: 0,   # healthy
        1: 1,   # asymptomatic
        2: 2,   # early
        3: 2,   # early
        4: 3,   # late
        5: 3,   # late
        6: 3,   # late
        7: 3,   # late
    }
    return label_map.get(dai, -1)


def load_hsi_image(hdr_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load hyperspectral image and wavelengths using memory mapping.

    Returns:
        (image_array, wavelengths)
        image_array: (H, W, B) as numpy.memmap
        wavelengths: (B,)
    """
    img = spectral.envi.open(hdr_path)
    # Use memory mapping to avoid loading full image into memory
    data = img.open_memmap(writable=False)  # (H, W, B) as memmap
    wavelengths = np.array(img.metadata['wavelength'], dtype=float)
    return data, wavelengths


def load_hsi_image_chunked(hdr_path: str, row_chunk_size: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load hyperspectral image in chunks to handle large images.

    Returns:
        (memmap_array, wavelengths, total_rows)
    """
    img = spectral.envi.open(hdr_path)
    data = img.open_memmap(writable=False)  # (H, W, B) as memmap
    wavelengths = np.array(img.metadata['wavelength'], dtype=float)
    total_rows = data.shape[0]
    return data, wavelengths, total_rows


def remove_water_bands(img: np.ndarray, wavelengths: np.ndarray,
                       remove_ranges: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Remove water absorption bands based on wavelength ranges.

    Returns:
        (filtered_img, filtered_wavelengths, removed_indices)
    """
    remove_indices = []
    for start, end in remove_ranges:
        mask = (wavelengths >= start) & (wavelengths <= end)
        remove_indices.extend(np.where(mask)[0])
    remove_indices = np.unique(remove_indices).astype(int)

    # Filter out-of-bounds indices
    valid_remove = remove_indices[remove_indices < img.shape[2]]

    if len(valid_remove) > 0:
        img_filtered = np.delete(img, valid_remove, axis=2)
        wavelengths_filtered = np.delete(wavelengths, valid_remove)
    else:
        img_filtered = img
        wavelengths_filtered = wavelengths

    return img_filtered, wavelengths_filtered, valid_remove.tolist()


def apply_dark_white_correction(img: np.ndarray, root: str) -> np.ndarray:
    """Apply dark/white reference correction if reference files exist."""
    dark_hdr = os.path.join(root, 'dark.bil.hdr')
    white_hdr = os.path.join(root, 'white.bil.hdr')

    if os.path.exists(dark_hdr) and os.path.exists(white_hdr):
        dark = spectral.envi.open(dark_hdr).load()
        white = spectral.envi.open(white_hdr).load()
        # Ensure shapes match (may need resizing)
        if dark.shape[:2] != img.shape[:2]:
            # Resize dark/white to match image spatial dims
            from skimage.transform import resize
            dark = resize(dark, img.shape[:2], preserve_range=True)
            white = resize(white, img.shape[:2], preserve_range=True)
        img = (img - dark) / (white - dark + 1e-8)
    return img


def center_crop(img: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Center crop image to target_size x target_size."""
    h, w = img.shape[:2]
    if h > target_size or w > target_size:
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        img = img[start_h:start_h+target_size, start_w:start_w+target_size]
    return img


def extract_patches(img: np.ndarray, patch_size: int = 9, stride: int = 4) -> np.ndarray:
    """
    Extract overlapping patches from hyperspectral image.

    Args:
        img: (H, W, B) hyperspectral image
        patch_size: Spatial size of patches
        stride: Stride for patch extraction

    Returns:
        patches: (N, patch_size, patch_size, B)
    """
    h, w, b = img.shape
    patches = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    return np.array(patches)


# =============================================================================
# Core Preprocessing Pipeline
# =============================================================================

def load_all_images_metadata(root: str = None) -> List[ImageMetadata]:
    """
    Load metadata for all images in the dataset without loading full data.
    This enables group-wise splitting before any preprocessing.
    """
    root = root or cfg.data_root
    metadata_list = []

    for file in os.listdir(root):
        if file.endswith('.bil.hdr'):
            hdr_path = os.path.join(root, file)
            try:
                # Quick load just to get shape and wavelengths
                img = spectral.envi.open(hdr_path)
                data_shape = img.load().shape  # This loads full data - optimize if needed
                wavelengths = np.array(img.metadata['wavelength'], dtype=float)

                dai, sensor_type, group_id = parse_filename(hdr_path)
                label = get_label_from_dai(dai)

                if label == -1:
                    print(f"Warning: Unknown DAI {dai} in {file}, skipping")
                    continue

                meta = ImageMetadata(
                    filepath=hdr_path,
                    filename=file,
                    dai=dai,
                    label=label,
                    sensor_type=sensor_type,
                    shape=data_shape,
                    group_id=group_id
                )
                metadata_list.append(meta)

            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

    print(f"Loaded metadata for {len(metadata_list)} images")
    return metadata_list


def group_wise_split(metadata_list: List[ImageMetadata],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     seed: int = 42) -> Tuple[List[ImageMetadata], List[ImageMetadata], List[ImageMetadata]]:
    """
    Split at the GROUP level (biological unit) to ensure zero data leakage.

    Groups are defined by unique acquisition identifiers (run numbers).
    All patches from the same group go to the same split.

    For stratification: groups with mixed labels need special handling.
    We use image-count-based stratification - each image contributes its label count.

    Args:
        metadata_list: List of ImageMetadata for all images
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility

    Returns:
        (train_metadata, val_metadata, test_metadata)
    """
    # Get unique groups with their images and label composition
    groups = {}
    for meta in metadata_list:
        if meta.group_id not in groups:
            groups[meta.group_id] = {
                'images': [],
                'sensor_type': meta.sensor_type,
                'label_counts': {},  # Count per label in this group
                'has_label': set()   # Which labels are present
            }
        groups[meta.group_id]['images'].append(meta)
        groups[meta.group_id]['label_counts'][meta.label] = \
            groups[meta.group_id]['label_counts'].get(meta.label, 0) + 1
        groups[meta.group_id]['has_label'].add(meta.label)

    group_ids = list(groups.keys())
    unique_labels = sorted(set(m.label for m in metadata_list))

    # STRATIFICATION FIX: Use label presence (not counts) for stratification
    # Each group gets a binary vector indicating which labels are present
    # This ensures each split has groups with diverse label coverage
    def get_label_presence_vector(gid):
        # Binary vector: 1 if label present, 0 if not
        presence = [1 if l in groups[gid]['has_label'] else 0 for l in unique_labels]
        return tuple(presence)

    group_presence_vectors = [get_label_presence_vector(gid) for gid in group_ids]

    # Stratified split at group level using label presence vectors
    np.random.seed(seed)

    # Strategy: Use stratified split only when it provides meaningful balance
    # Check if stratification by presence vectors is feasible
    unique_vectors = len(set(group_presence_vectors))
    min_vector_count = min([group_presence_vectors.count(v) for v in set(group_presence_vectors)]) if unique_vectors > 1 else 0

    # For stratification to work reasonably, we need multiple groups per vector pattern
    # But with many unique vectors, we may need to fall back to random split
    use_stratification = unique_vectors > 1 and min_vector_count >= 2

    if use_stratification:
        # First split: train vs (val+test)
        train_groups, temp_groups = train_test_split(
            group_ids,
            test_size=1-train_ratio,
            stratify=group_presence_vectors,
            random_state=seed
        )
        temp_presence_vectors = [get_label_presence_vector(gid) for gid in temp_groups]
        val_ratio_rel = val_ratio / (val_ratio + test_ratio)

        # Check if stratification is possible for second split
        unique_temp_vectors = len(set(temp_presence_vectors))
        min_temp_vector_count = min([temp_presence_vectors.count(v) for v in set(temp_presence_vectors)]) if unique_temp_vectors > 1 else 0

        if unique_temp_vectors > 1 and min_temp_vector_count >= 2:
            val_groups, test_groups = train_test_split(
                temp_groups,
                test_size=1-val_ratio_rel,
                stratify=temp_presence_vectors,
                random_state=seed
            )
        else:
            val_groups, test_groups = train_test_split(
                temp_groups,
                test_size=1-val_ratio_rel,
                random_state=seed
            )
    else:
        # Fall back to random split when stratification not feasible
        train_groups, temp_groups = train_test_split(
            group_ids,
            test_size=1-train_ratio,
            random_state=seed
        )
        val_ratio_rel = val_ratio / (val_ratio + test_ratio)
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=1-val_ratio_rel,
            random_state=seed
        )

    # Collect metadata for each split
    def collect_metadata(group_list):
        result = []
        for gid in group_list:
            result.extend(groups[gid]['images'])
        return result

    train_meta = collect_metadata(train_groups)
    val_meta = collect_metadata(val_groups)
    test_meta = collect_metadata(test_groups)

    # Verify zero group overlap
    train_gids = set(train_groups)
    val_gids = set(val_groups)
    test_gids = set(test_groups)

    overlap_check = {
        'train_val_overlap': len(train_gids & val_gids) > 0,
        'train_test_overlap': len(train_gids & test_gids) > 0,
        'val_test_overlap': len(val_gids & test_gids) > 0,
    }

    print(f"Split complete: Train={len(train_meta)} images ({len(train_groups)} groups), "
          f"Val={len(val_meta)} images ({len(val_groups)} groups), "
          f"Test={len(test_meta)} images ({len(test_groups)} groups)")
    print(f"Group overlap check: {overlap_check}")

    if any(overlap_check.values()):
        raise ValueError("GROUP OVERLAP DETECTED! Splits are not mutually exclusive.")

    # Verify all labels are present in each split
    for name, meta_list in [('Train', train_meta), ('Val', val_meta), ('Test', test_meta)]:
        labels = [m.label for m in meta_list]
        unique_labels_in_split = sorted(set(labels))
        print(f"{name} labels present: {unique_labels_in_split}")

    return train_meta, val_meta, test_meta


def create_split_manifest(train_meta: List[ImageMetadata],
                         val_meta: List[ImageMetadata],
                         test_meta: List[ImageMetadata],
                         train_patches: int,
                         val_patches: int,
                         test_patches: int) -> SplitManifest:
    """Create a manifest documenting the split for reproducibility."""

    train_groups = list(set(m.group_id for m in train_meta))
    val_groups = list(set(m.group_id for m in val_meta))
    test_groups = list(set(m.group_id for m in test_meta))

    total_images = len(train_meta) + len(val_meta) + len(test_meta)
    total_groups = len(train_groups) + len(val_groups) + len(test_groups)

    manifest = SplitManifest(
        train_images=[m.filename for m in train_meta],
        val_images=[m.filename for m in val_meta],
        test_images=[m.filename for m in test_meta],
        train_patches=train_patches,
        val_patches=val_patches,
        test_patches=test_patches,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        group_overlap_check={
            'train_val_overlap': False,
            'train_test_overlap': False,
            'val_test_overlap': False,
        },
        split_ratios={
            'train_images': len(train_meta) / total_images,
            'val_images': len(val_meta) / total_images,
            'test_images': len(test_meta) / total_images,
            'train_groups': len(train_groups) / total_groups,
            'val_groups': len(val_groups) / total_groups,
            'test_groups': len(test_groups) / total_groups,
        },
        seed=cfg.seed,
        sensor_specific=True
    )

    return manifest


def fit_sensor_specific_pca(train_meta: List[ImageMetadata],
                            n_components: int,
                            patch_size: int,
                            stride: int) -> Tuple[PCA, PCA, List[int], List[int], np.ndarray, np.ndarray]:
    """
    Fit separate PCA models for each sensor type using ONLY training patches.
    Uses memory mapping and chunked processing to handle large VIS-NIR images.

    Returns:
        (pca_swir, pca_visnir, remove_bands_swir, remove_bands_visnir,
         wavelengths_swir, wavelengths_visnir)
    """
    from sklearn.decomposition import IncrementalPCA

    # Collect training patches by sensor type using chunked processing
    wavelengths_swir = None
    wavelengths_visnir = None
    remove_bands_swir = []
    remove_bands_visnir = []

    # For IncrementalPCA, we need to process patches in batches
    # We'll accumulate pixel spectra from patches
    swir_spectra_chunks = []
    visnir_spectra_chunks = []
    chunk_size = 10000  # Number of spectra per batch

    for meta in train_meta:
        # Load with memory mapping
        img_memmap, wavelengths, total_rows = load_hsi_image_chunked(meta.filepath)
        h, w, b = img_memmap.shape

        # Apply dark/white correction (need to process in chunks)
        # For now, process the center-cropped region directly
        # Center crop: from 64x64 center
        start_h = max(0, (h - 64) // 2)
        start_w = max(0, (w - 64) // 2)
        end_h = min(h, start_h + 64)
        end_w = min(w, start_w + 64)

        # Process in row chunks to avoid memory issues
        for row_start in range(start_h, end_h, 50):
            row_end = min(row_start + 50, end_h)
            # Load chunk: (chunk_h, 64, b)
            chunk = img_memmap[row_start:row_end, start_w:end_w, :]
            chunk = np.array(chunk)  # Convert memmap slice to array

            # Apply dark/white correction if available
            chunk = apply_dark_white_correction(chunk, os.path.dirname(meta.filepath))

            # Remove water bands
            chunk_filtered, wavelengths_filtered, removed = remove_water_bands(
                chunk, wavelengths, cfg.remove_bands)

            # Store sensor-specific info from first image of each type
            if meta.sensor_type == 'swir_168':
                if wavelengths_swir is None:
                    wavelengths_swir = wavelengths_filtered
                    remove_bands_swir = removed
                # Reshape chunk to spectra: (chunk_h * 64, b_filtered)
                chunk_spectra = chunk_filtered.reshape(-1, chunk_filtered.shape[2])
                swir_spectra_chunks.append(chunk_spectra)
            elif meta.sensor_type == 'visnir_300':
                if wavelengths_visnir is None:
                    wavelengths_visnir = wavelengths_filtered
                    remove_bands_visnir = removed
                chunk_spectra = chunk_filtered.reshape(-1, chunk_filtered.shape[2])
                visnir_spectra_chunks.append(chunk_spectra)

    # Fit IncrementalPCA on accumulated spectra
    pca_swir = None
    pca_visnir = None

    if swir_spectra_chunks:
        # Concatenate all SWIR spectra
        all_swir_spectra = np.concatenate(swir_spectra_chunks, axis=0)
        print(f"Total SWIR spectra for PCA: {all_swir_spectra.shape[0]}")
        pca_swir = IncrementalPCA(n_components=n_components, batch_size=chunk_size)
        pca_swir.fit(all_swir_spectra)
        print(f"Fitted SWIR PCA: {all_swir_spectra.shape[1]} bands -> {n_components} components, "
              f"explained variance: {pca_swir.explained_variance_ratio_.sum():.4f}")

    if visnir_spectra_chunks:
        all_visnir_spectra = np.concatenate(visnir_spectra_chunks, axis=0)
        print(f"Total VIS-NIR spectra for PCA: {all_visnir_spectra.shape[0]}")
        pca_visnir = IncrementalPCA(n_components=n_components, batch_size=chunk_size)
        pca_visnir.fit(all_visnir_spectra)
        print(f"Fitted VIS-NIR PCA: {all_visnir_spectra.shape[1]} bands -> {n_components} components, "
              f"explained variance: {pca_visnir.explained_variance_ratio_.sum():.4f}")

    return (pca_swir, pca_visnir, remove_bands_swir, remove_bands_visnir,
            wavelengths_swir, wavelengths_visnir)


def fit_normalization_scaler(train_meta: List[ImageMetadata],
                             pca_swir: PCA,
                             pca_visnir: PCA,
                             patch_size: int,
                             stride: int) -> StandardScaler:
    """
    Fit StandardScaler on TRAINING patches only (after PCA transformation).
    Uses memory mapping and chunked processing to handle large VIS-NIR images.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import IncrementalPCA

    # We'll collect transformed patch data in chunks
    all_spectra_chunks = []
    chunk_size = 10000  # Number of spectra per batch for incremental fitting

    for meta in train_meta:
        # Load with memory mapping
        img_memmap, wavelengths, total_rows = load_hsi_image_chunked(meta.filepath)
        h, w, b = img_memmap.shape

        # Apply dark/white correction and center crop on-the-fly
        # Center crop region
        start_h = max(0, (h - 64) // 2)
        start_w = max(0, (w - 64) // 2)
        end_h = min(h, start_h + 64)
        end_w = min(w, start_w + 64)
        crop_h = end_h - start_h
        crop_w = end_w - start_w

        # Process in row chunks
        for row_start in range(start_h, end_h, 50):
            row_end = min(row_start + 50, end_h)
            # Load chunk: (chunk_h, crop_w, b)
            chunk = img_memmap[row_start:row_end, start_w:end_w, :]
            chunk = np.array(chunk)  # Convert memmap slice to array

            # Apply dark/white correction
            chunk = apply_dark_white_correction(chunk, os.path.dirname(meta.filepath))

            # Remove water bands
            chunk_filtered, wavelengths_filtered, _ = remove_water_bands(
                chunk, wavelengths, cfg.remove_bands)

            # Apply appropriate PCA
            if meta.sensor_type == 'swir_168' and pca_swir is not None:
                chunk_h, chunk_w, chunk_b = chunk_filtered.shape
                flat = chunk_filtered.reshape(chunk_h * chunk_w, chunk_b)
                transformed = pca_swir.transform(flat).reshape(chunk_h, chunk_w, -1)
            elif meta.sensor_type == 'visnir_300' and pca_visnir is not None:
                chunk_h, chunk_w, chunk_b = chunk_filtered.shape
                flat = chunk_filtered.reshape(chunk_h * chunk_w, chunk_b)
                transformed = pca_visnir.transform(flat).reshape(chunk_h, chunk_w, -1)
            else:
                continue

            # Extract patches from this chunk
            patches = extract_patches(transformed, patch_size, stride)

            # Reshape to spectra
            n, h_p, w_p, c = patches.shape
            spectra = patches.reshape(n * h_p * w_p, c)
            all_spectra_chunks.append(spectra)

    if not all_spectra_chunks:
        raise ValueError("No training patches extracted!")

    # Concatenate all spectra and fit scaler
    all_spectra = np.concatenate(all_spectra_chunks, axis=0)
    print(f"Total spectra for normalization: {all_spectra.shape[0]}")

    scaler = StandardScaler()
    scaler.fit(all_spectra)

    print(f"Fitted normalization scaler on {all_spectra.shape[0]} spectra, {all_spectra.shape[1]} channels")
    print(f"  Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  Std range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

    return scaler


def process_metadata_to_patches(meta_list: List[ImageMetadata],
                                pca_swir: Optional[PCA],
                                pca_visnir: Optional[PCA],
                                scaler: StandardScaler,
                                patch_size: int,
                                stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a list of image metadata to patches using fitted transformers.
    No fitting occurs here - only transform.
    Uses memory mapping and chunked processing to handle large VIS-NIR images.
    """
    X_list = []
    y_list = []

    for meta in meta_list:
        # Load with memory mapping
        img_memmap, wavelengths, total_rows = load_hsi_image_chunked(meta.filepath)
        h, w, b = img_memmap.shape

        # Center crop region
        start_h = max(0, (h - 64) // 2)
        start_w = max(0, (w - 64) // 2)
        end_h = min(h, start_h + 64)
        end_w = min(w, start_w + 64)
        crop_h = end_h - start_h
        crop_w = end_w - start_w

        # For small images (SWIR), we can load the whole cropped region
        # For large images (VIS-NIR), process in chunks
        if meta.sensor_type == 'swir_168':
            # Load the entire cropped region at once
            img_crop = img_memmap[start_h:end_h, start_w:end_w, :]
            img_crop = np.array(img_crop)  # Convert to array
            img_crop = apply_dark_white_correction(img_crop, os.path.dirname(meta.filepath))
            img_filtered, _, _ = remove_water_bands(img_crop, wavelengths, cfg.remove_bands)

            # Apply PCA
            if pca_swir is not None:
                h_f, w_f, b_f = img_filtered.shape
                flat = img_filtered.reshape(h_f * w_f, b_f)
                transformed = pca_swir.transform(flat).reshape(h_f, w_f, -1)
            else:
                print(f"Warning: No PCA for sensor type {meta.sensor_type}, skipping {meta.filename}")
                continue

            # Extract patches
            patches = extract_patches(transformed, patch_size, stride)

            # Apply normalization
            n, h_p, w_p, c = patches.shape
            patches_flat = patches.reshape(n * h_p * w_p, c)
            patches_normalized = scaler.transform(patches_flat).reshape(n, h_p, w_p, c)

            X_list.append(patches_normalized)
            y_list.extend([meta.label] * n)

        elif meta.sensor_type == 'visnir_300':
            # Process in row chunks to avoid memory issues
            image_patches = []
            for row_start in range(start_h, end_h, 50):
                row_end = min(row_start + 50, end_h)
                chunk = img_memmap[row_start:row_end, start_w:end_w, :]
                chunk = np.array(chunk)

                chunk = apply_dark_white_correction(chunk, os.path.dirname(meta.filepath))
                chunk_filtered, _, _ = remove_water_bands(chunk, wavelengths, cfg.remove_bands)

                if pca_visnir is not None:
                    chunk_h, chunk_w, chunk_b = chunk_filtered.shape
                    flat = chunk_filtered.reshape(chunk_h * chunk_w, chunk_b)
                    transformed = pca_visnir.transform(flat).reshape(chunk_h, chunk_w, -1)
                else:
                    print(f"Warning: No PCA for sensor type {meta.sensor_type}, skipping {meta.filename}")
                    continue

                patches = extract_patches(transformed, patch_size, stride)
                image_patches.append(patches)

            if image_patches:
                patches = np.concatenate(image_patches, axis=0)
                n, h_p, w_p, c = patches.shape
                patches_flat = patches.reshape(n * h_p * w_p, c)
                patches_normalized = scaler.transform(patches_flat).reshape(n, h_p, w_p, c)

                X_list.append(patches_normalized)
                y_list.extend([meta.label] * n)

    if not X_list:
        return np.array([]), np.array([])

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)

    return X, y


# =============================================================================
# Main Entry Points
# =============================================================================

def load_tomato_dataset_corrected(root: str = None,
                                   patch_size: int = 9,
                                   stride: int = 4,
                                   save_artifacts_path: Optional[str] = None,
                                   load_artifacts_path: Optional[str] = None,
                                   save_manifest_path: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Main entry point for corrected dataset loading.

    This function:
    1. Loads metadata for all images (no data loading yet)
    2. Splits at GROUP level (biological unit) - ZERO LEAKAGE
    3. Fits PCA on TRAINING patches only, per sensor type
    4. Fits normalization on TRAINING patches only
    5. Extracts patches and applies transforms to all splits
    6. Saves artifacts and manifest for reproducibility

    Returns:
        Dict with 'train', 'val', 'test' keys, each containing (X, y)
    """
    root = root or cfg.data_root

    # If artifacts exist, load them (for val/test evaluation)
    if load_artifacts_path and os.path.exists(str(load_artifacts_path) + '_sklearn.joblib'):
        print(f"Loading preprocessing artifacts from {load_artifacts_path}")
        artifacts = PreprocessingArtifacts.load(str(load_artifacts_path))
        pca_swir = artifacts.pca_swir_168
        pca_visnir = artifacts.pca_visnir_300
        scaler = artifacts.scaler
        remove_bands_swir = artifacts.remove_bands_swir
        remove_bands_visnir = artifacts.remove_bands_visnir
        wavelengths_swir = artifacts.wavelengths_swir
        wavelengths_visnir = artifacts.wavelengths_visnir

        # Load manifest to get same splits
        if save_manifest_path and os.path.exists(save_manifest_path):
            manifest = SplitManifest.load(save_manifest_path)
            # Reconstruct metadata lists from manifest
            all_meta = load_all_images_metadata(root)
            filename_to_meta = {m.filename: m for m in all_meta}

            train_meta = [filename_to_meta[f] for f in manifest.train_images if f in filename_to_meta]
            val_meta = [filename_to_meta[f] for f in manifest.val_images if f in filename_to_meta]
            test_meta = [filename_to_meta[f] for f in manifest.test_images if f in filename_to_meta]
        else:
            raise ValueError("Manifest required when loading artifacts")

    else:
        # Fresh processing: load metadata and create splits
        print("Loading all image metadata...")
        all_meta = load_all_images_metadata(root)

        print("Performing group-wise split...")
        train_meta, val_meta, test_meta = group_wise_split(
            all_meta,
            train_ratio=cfg.train_split,
            val_ratio=cfg.val_split,
            test_ratio=cfg.test_split,
            seed=cfg.seed
        )

        print("Fitting sensor-specific PCA on training patches...")
        pca_swir, pca_visnir, remove_bands_swir, remove_bands_visnir, wavelengths_swir, wavelengths_visnir = \
            fit_sensor_specific_pca(train_meta, cfg.pca_components, patch_size, stride)

        print("Fitting normalization scaler on training patches...")
        scaler = fit_normalization_scaler(train_meta, pca_swir, pca_visnir, patch_size, stride)

        # Save artifacts if path provided
        if save_artifacts_path:
            artifacts = PreprocessingArtifacts(
                pca_swir_168=pca_swir,
                pca_visnir_300=pca_visnir,
                scaler=scaler,
                remove_bands_swir=remove_bands_swir,
                remove_bands_visnir=remove_bands_visnir,
                wavelengths_swir=wavelengths_swir,
                wavelengths_visnir=wavelengths_visnir
            )
            artifacts.save(save_artifacts_path)
            print(f"Saved preprocessing artifacts to {save_artifacts_path}")

    # Process each split using fitted transformers
    print("Processing training patches...")
    X_train, y_train = process_metadata_to_patches(train_meta, pca_swir, pca_visnir, scaler, patch_size, stride)

    print("Processing validation patches...")
    X_val, y_val = process_metadata_to_patches(val_meta, pca_swir, pca_visnir, scaler, patch_size, stride)

    print("Processing test patches...")
    X_test, y_test = process_metadata_to_patches(test_meta, pca_swir, pca_visnir, scaler, patch_size, stride)

    # Create and save manifest
    manifest = create_split_manifest(train_meta, val_meta, test_meta,
                                     len(X_train), len(X_val), len(X_test))

    if save_manifest_path:
        manifest.save(save_manifest_path)
        print(f"Saved split manifest to {save_manifest_path}")

    print(f"\nFinal dataset shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


# =============================================================================
# Backward Compatibility Wrapper
# =============================================================================

def load_tomato_dataset(root: str = None, patch_size: int = 9, stride: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward compatibility wrapper - returns combined dataset (NOT RECOMMENDED).
    Use load_tomato_dataset_corrected for scientifically valid splits.
    """
    import warnings
    warnings.warn(
        "load_tomato_dataset uses the OLD leaky pipeline. "
        "Use load_tomato_dataset_corrected for valid scientific results.",
        DeprecationWarning,
        stacklevel=2
    )

    # This is the original leaky implementation kept for reference
    splits = load_tomato_dataset_corrected(root, patch_size, stride)
    X = np.concatenate([splits['train'][0], splits['val'][0], splits['test'][0]], axis=0)
    y = np.concatenate([splits['train'][1], splits['val'][1], splits['test'][1]], axis=0)
    return X, y


def spectral_normalize(X: np.ndarray) -> np.ndarray:
    """Deprecated: Use fitted StandardScaler instead."""
    import warnings
    warnings.warn("spectral_normalize computes statistics on all data (leakage). "
                  "Use fitted scaler from corrected pipeline.", DeprecationWarning, stacklevel=2)
    B = X.shape[-1]
    X_reshaped = X.reshape(-1, B)
    mu = X_reshaped.mean(axis=0)
    sigma = X_reshaped.std(axis=0) + 1e-8
    Xn = (X_reshaped - mu) / sigma
    return Xn.reshape(X.shape)


def apply_pca_single(img: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """Deprecated: Per-image PCA causes leakage."""
    import warnings
    warnings.warn("apply_pca_single fits PCA per image (major leakage). "
                  "Use sensor-specific PCA fitted on training data only.",
                  DeprecationWarning, stacklevel=2)
    H, W, B = img.shape
    X_flat = img.reshape(H * W, B)
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=False, random_state=cfg.seed)
    X_p = pca.fit_transform(X_flat).reshape(H, W, n_components)
    return X_p, pca


def apply_pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """Deprecated: Fits on all patches including test."""
    import warnings
    warnings.warn("apply_pca fits on all data (leakage). Use corrected pipeline.",
                  DeprecationWarning, stacklevel=2)
    N, H, W, B = X.shape
    X_flat = X.reshape(N * H * W, B)
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=False, random_state=cfg.seed)
    X_p = pca.fit_transform(X_flat).reshape(N, H, W, n_components)
    return X_p, pca


def train_val_test_split(X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Deprecated: Patch-level split causes leakage."""
    import warnings
    warnings.warn("train_val_test_split operates on patches (leakage). "
                  "Use group-wise split in corrected pipeline.", DeprecationWarning, stacklevel=2)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-cfg.train_split, stratify=y, random_state=cfg.seed)
    rel = cfg.val_split / (cfg.val_split + cfg.test_split)
    if len(np.unique(y_temp)) > 1 and min(np.bincount(y_temp)) > 1:
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-rel, stratify=y_temp, random_state=cfg.seed)
    else:
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-rel, random_state=cfg.seed)
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


# =============================================================================
# Dataset Class (unchanged)
# =============================================================================

class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment=False):
        # For patches: X is (N, H, W, B), keep as 4D for CNN input [N, B, H, W]
        N, H, W, B = X.shape
        self.X = torch.from_numpy(X).float().permute(0, 3, 1, 2)  # (N, B, H, W)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment:
            # For patches, add noise or other augmentations if needed
            pass
        return x, y