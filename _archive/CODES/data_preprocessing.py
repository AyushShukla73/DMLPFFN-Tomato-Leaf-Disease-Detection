import os
import numpy as np
import torch
import spectral
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
from config import cfg

# Placeholder: implement actual loaders according to USDA dataset structure

def extract_patches(img: np.ndarray, patch_size: int = 9, stride: int = 4) -> np.ndarray:
    """Extract overlapping patches from hyperspectral image."""
    h, w, b = img.shape
    patches = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    return np.array(patches)

def load_tomato_dataset(root: str = None, patch_size: int = 9, stride: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return X, y with patch extraction
    X: (N, patch_size, patch_size, B) hyperspectral patches
    y: (N,) labels in {0: healthy, 1: asymptomatic, 2: early, 3: late}
    """
    root = root or cfg.data_root
    # Label mapping based on days after inoculation (dai)
    label_map = {
        0: 0,  # healthy
        1: 1,  # asymptomatic
        2: 2,  # early
        3: 2,  # early
        4: 3,  # late
        5: 3,  # late
        6: 3,  # late
        7: 3   # late
    }
    X_list = []
    y_list = []
    for file in os.listdir(root):
        if file.endswith('.bil.hdr'):
            hdr_path = os.path.join(root, file)
            bil_path = hdr_path[:-4]  # remove .hdr
            # Parse dai from filename (e.g., '0dai_...' -> 0)
            dai_str = file.split('dai')[0]
            try:
                dai = int(dai_str)
            except ValueError:
                continue
            label = label_map.get(dai, -1)
            if label == -1:
                continue
            # Load hyperspectral cube
            img = spectral.envi.open(hdr_path).load()
            # img shape: (H, W, B)
            # Include all available bands
            # Patch to 64x64 if larger
            h, w = img.shape[:2]
            if h > 64 or w > 64:
                start_h = (h - 64) // 2
                start_w = (w - 64) // 2
                img = img[start_h:start_h+64, start_w:start_w+64]
            # Get wavelengths
            wavelengths = np.array(spectral.envi.open(hdr_path).metadata['wavelength'], dtype=float)
            # Find bands to remove (water absorption)
            remove_indices = []
            for start, end in cfg.remove_bands:
                mask = (wavelengths >= start) & (wavelengths <= end)
                remove_indices.extend(np.where(mask)[0])
            remove_indices = np.unique(remove_indices)
            # Remove bands if any valid indices
            if len(remove_indices) > 0 and remove_indices.max() < img.shape[2]:
                img = np.delete(img, remove_indices, axis=2)
            # Apply dark/white reference correction if reference files exist
            dark_hdr = os.path.join(root, 'dark.bil.hdr')
            white_hdr = os.path.join(root, 'white.bil.hdr')
            if os.path.exists(dark_hdr) and os.path.exists(white_hdr):
                dark = spectral.envi.open(dark_hdr).load()
                white = spectral.envi.open(white_hdr).load()
                img = (img - dark) / (white - dark + 1e-8)
            # Apply PCA to harmonize bands per image
            img_pca, _ = apply_pca_single(img, cfg.pca_components)
            # Extract patches
            patches = extract_patches(img_pca, patch_size, stride)
            # Assign label to all patches from this image
            X_list.extend(patches)
            y_list.extend([label] * len(patches))
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def spectral_normalize(X: np.ndarray) -> np.ndarray:
    # Per-band z-score normalization
    B = X.shape[-1]
    X_reshaped = X.reshape(-1, B)
    mu = X_reshaped.mean(axis=0)
    sigma = X_reshaped.std(axis=0) + 1e-8
    Xn = (X_reshaped - mu) / sigma
    return Xn.reshape(X.shape)


def apply_pca_single(img: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """Apply PCA to a single hyperspectral image."""
    H, W, B = img.shape
    X_flat = img.reshape(H * W, B)
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=False, random_state=cfg.seed)
    X_p = pca.fit_transform(X_flat).reshape(H, W, n_components)
    return X_p, pca

def apply_pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    N, H, W, B = X.shape
    X_flat = X.reshape(N * H * W, B)
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=False, random_state=cfg.seed)
    X_p = pca.fit_transform(X_flat).reshape(N, H, W, n_components)
    return X_p, pca


def train_val_test_split(X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-cfg.train_split, stratify=y, random_state=cfg.seed)
    rel = cfg.val_split / (cfg.val_split + cfg.test_split)
    # Remove stratify if classes are too imbalanced
    if len(np.unique(y_temp)) > 1 and min(np.bincount(y_temp)) > 1:
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-rel, stratify=y_temp, random_state=cfg.seed)
    else:
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-rel, random_state=cfg.seed)
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

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
