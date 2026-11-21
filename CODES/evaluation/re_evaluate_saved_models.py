#!/usr/bin/env python
"""
Unified Re-evaluation Script for Phase 5 Baseline Models
=========================================================
Loads saved model checkpoints, runs inference on frozen test set,
applies conditional binary evaluation, extracts group information,
and saves complete metrics + test artifacts for statistical analysis.

Usage:
    python re_evaluate_saved_models.py --model simple_cnn --seeds 42 123 456 789 999
    python re_evaluate_saved_models.py --model all --seeds 42 123 456 789 999
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # CODES directory for config import

from CODES.evaluation.metrics import evaluate_model_conditional_binary
from CODES.baselines.trainer import BaselineTrainer
# Import ACTUAL model classes from CODES/baselines/ (these match the trained checkpoints)
from CODES.baselines.simple_cnn import SimpleCNN
from CODES.baselines.mlp_mixer_hsi import MLPMixer_HSI
from CODES.baselines.resnet18_hsi import ResNet18_HSI
from CODES.config_baselines import EXPERIMENTS, SEEDS, get_experiment_config, get_model_config
from CODES.data_preprocessing_corrected import load_tomato_dataset_corrected, load_all_images_metadata


def load_frozen_manifest() -> Dict:
    """Load the frozen split manifest (source of truth)."""
    manifest_path = Path(__file__).parent.parent.parent / "artifacts" / "split_manifest_frozen.json"
    with open(manifest_path, 'r') as f:
        return json.load(f)


def load_preprocessing_artifacts(artifacts_dir: Path) -> Dict:
    """Load PCA models and scaler from preprocessing artifacts."""
    import joblib

    artifacts = {}

    # The artifacts are in the artifacts/ directory directly
    # Look for the sklearn joblib file that contains PCA and scaler
    sklearn_artifacts = artifacts_dir / "preprocessing_artifacts_sklearn.joblib"

    if sklearn_artifacts.exists():
        loaded = joblib.load(sklearn_artifacts)
        artifacts.update(loaded)

    return artifacts


def get_test_group_indices(manifest: Dict, test_labels: np.ndarray) -> np.ndarray:
    """
    Extract group indices for test patches from the frozen manifest.

    The manifest contains test_groups (biological units). We need to map each
    test patch to its group index.

    Since patches are extracted from images in order, and the test_images list
    in the manifest corresponds to the test patches in order, we can reconstruct
    the group for each patch by:
    1. Finding which image each patch came from
    2. Mapping that image to its biological group
    3. Assigning group index

    However, we don't have the patch-to-image mapping directly. We need to
    compute it by loading the data the same way as load_tomato_dataset_corrected().
    """
    # For now, use the fact that test_groups in manifest has 4 groups
    # We'll need to extract this from the dataset loading
    pass


def load_test_dataset_with_groups(config: Dict, manifest: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load test dataset and return X_test, y_test, split_manifest.

    The split_manifest contains test_images and test_groups needed to extract group indices.
    """
    # Use the corrected data loading function with the frozen manifest artifacts
    # Artifacts are directly in artifacts/ directory (preprocessing_artifacts_sklearn.joblib, etc.)
    artifacts_dir = str(Path(__file__).parent.parent.parent / "artifacts")
    manifest_path = str(Path(__file__).parent.parent.parent / "artifacts" / "split_manifest_frozen.json")

    # Load using the frozen manifest artifacts
    splits = load_tomato_dataset_corrected(
        root=config.get('data_root', '../data'),
        patch_size=config.get('patch_size', 9),
        stride=config.get('stride', 4),
        load_artifacts_path=artifacts_dir,
        save_artifacts_path=None,  # Don't overwrite!
        save_manifest_path=manifest_path  # Use frozen manifest
    )

    # splits is a dict with 'train', 'val', 'test' keys
    X_test, y_test = splits['test']
    split_manifest = manifest  # Use the frozen manifest we already loaded

    # For now, return what we have. We'll compute group indices in the caller.
    return X_test, y_test, split_manifest


def extract_group_indices_from_manifest(split_manifest: Dict, X_test: np.ndarray, data_config: Dict) -> np.ndarray:
    """
    Extract group index for each test patch.

    The split_manifest has:
    - test_images: list of test image filenames (in order)
    - test_groups: list of group identifiers for each test image

    The patches are extracted image by image in the same order as test_images.
    We need to compute how many patches came from each test image by replicating
    the preprocessing logic: center crop to 64x64 (hardcoded in process_metadata_to_patches),
    then extract patches with patch_size=9, stride=4.

    IMPORTANT: VIS-NIR images use chunked processing (50 rows at a time) which loses
    patches at chunk boundaries. SWIR images use full extraction.
    """
    test_images = split_manifest['test_images']
    test_groups = split_manifest['test_groups']

    # Map each image to its group index (unique groups in order of appearance)
    unique_groups = []
    for g in test_groups:
        if g not in unique_groups:
            unique_groups.append(g)
    image_to_group = {}
    for img, group in zip(test_images, test_groups):
        group_idx = unique_groups.index(group)
        image_to_group[img] = group_idx

    # Load image metadata to get original dimensions and sensor type
    root = data_config.get('data_root', '../data')
    all_meta = load_all_images_metadata(root)
    filename_to_meta = {m.filename: m for m in all_meta}

    patch_size = data_config.get('patch_size', 9)
    stride = data_config.get('stride', 4)
    crop_size = 64  # Hardcoded in process_metadata_to_patches

    # Count patches per test image using same logic as process_metadata_to_patches
    # SWIR: full extraction, VIS-NIR: chunked extraction (50 rows per chunk)
    patches_per_image = []
    total_patches = 0

    for img_name in test_images:
        if img_name in filename_to_meta:
            meta = filename_to_meta[img_name]
            h, w = meta.shape[0], meta.shape[1]  # Original image H, W
            sensor_type = meta.sensor_type

            # Replicate center crop logic from process_metadata_to_patches (lines 724-729)
            start_h = max(0, (h - crop_size) // 2)
            start_w = max(0, (w - crop_size) // 2)
            end_h = min(h, start_h + crop_size)
            end_w = min(w, start_w + crop_size)
            crop_h = end_h - start_h
            crop_w = end_w - start_w

            # Compute patches from cropped region
            if crop_h >= patch_size and crop_w >= patch_size:
                if sensor_type == 'visnir_300':
                    # VIS-NIR: chunked processing (50 rows at a time) - matches process_metadata_to_patches lines 763-779
                    total_patches_img = 0
                    for row_start in range(0, crop_h, 50):
                        row_end = min(row_start + 50, crop_h)
                        chunk_h = row_end - row_start
                        if chunk_h >= patch_size:
                            n_patches_h = (chunk_h - patch_size) // stride + 1
                            n_patches_w = (crop_w - patch_size) // stride + 1
                            total_patches_img += n_patches_h * n_patches_w
                    n_patches = total_patches_img
                else:
                    # SWIR: full extraction
                    n_patches_h = (crop_h - patch_size) // stride + 1
                    n_patches_w = (crop_w - patch_size) // stride + 1
                    n_patches = n_patches_h * n_patches_w
            else:
                n_patches = 0

            patches_per_image.append(n_patches)
            total_patches += n_patches
        else:
            print(f"Warning: {img_name} not found in metadata")
            patches_per_image.append(0)

    # Now assign group index to each patch
    group_indices = np.zeros(len(X_test), dtype=int)
    patch_idx = 0

    for img_idx, n_patches in enumerate(patches_per_image):
        if img_idx < len(test_images):
            img_name = test_images[img_idx]
            group_idx = image_to_group.get(img_name, 0)
            group_indices[patch_idx:patch_idx + n_patches] = group_idx
            patch_idx += n_patches

    # Verify we assigned all patches
    assert patch_idx == len(X_test), f"Assigned {patch_idx} patches but have {len(X_test)}"

    return group_indices


# Map original model names to actual model classes from CODES/baselines/
MODEL_CLASS_MAP = {
    'simple_cnn': SimpleCNN,
    'mlp_mixer_hsi': MLPMixer_HSI,
    'resnet18_hsi': ResNet18_HSI,
}

# Model-specific default configurations (must match training configs exactly)
MODEL_DEFAULT_CONFIGS = {
    'simple_cnn': {
        'in_channels': 128,
        'num_classes': 4,
        'base_channels': 44,
    },
    'mlp_mixer_hsi': {
        'in_channels': 128,
        'num_classes': 4,
        'patch_size': 9,
        'embedding_dim': 256,
        'num_blocks': 8,
        'token_mixing_dim': 128,
        'channel_mixing_dim': 512,
        'dropout': 0.1,
    },
    'resnet18_hsi': {
        'in_channels': 128,
        'num_classes': 4,
    },
}

def evaluate_model(
    model_name: str,
    model_name_original: str,
    seed: int,
    experiment_config: Dict,
    manifest: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    group_indices: np.ndarray,
    device: torch.device
) -> Dict:
    """
    Evaluate a single model checkpoint on the test set.

    Returns dictionary with all metrics and saves artifacts.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} seed {seed}")
    print(f"{'='*60}")

    # Use model-specific default configs that match the training architecture exactly
    model_kwargs = MODEL_DEFAULT_CONFIGS.get(model_name_original)
    if model_kwargs is None:
        raise ValueError(f"Unknown model: {model_name_original}. Available: {list(MODEL_DEFAULT_CONFIGS.keys())}")

    # Load ACTUAL model class that matches the trained checkpoint
    ModelClass = MODEL_CLASS_MAP.get(model_name_original)
    if ModelClass is None:
        raise ValueError(f"Unknown model class: {model_name_original}. Available: {list(MODEL_CLASS_MAP.keys())}")

    model = ModelClass(**model_kwargs)
    model = model.to(device)

    # Load checkpoint - checkpoint dir uses original model_name (e.g., simple_cnn)
    checkpoint_dir = Path(__file__).parent.parent.parent / "artifacts" / "baselines" / model_name_original / f"seed_{seed}"
    checkpoint_path = checkpoint_dir / "best_model.pth"

    if not checkpoint_path.exists():
        # Try root directory
        checkpoint_path = Path(__file__).parent.parent.parent / "artifacts" / "baselines" / model_name_original / "best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Checkpoint might be a state dict directly or wrapped in a dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Create data loader for test set
    from torch.utils.data import DataLoader, TensorDataset

    # Convert to NCHW format
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_test_tensor = torch.LongTensor(y_test)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Run inference
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(batch_y.numpy())

    logits = np.vstack(all_logits)
    targets = np.hstack(all_targets)

    # Verify targets match
    assert np.array_equal(targets, y_test), "Targets mismatch!"

    # Map targets: original labels are 0 (Healthy) and 3 (Late Blight)
    # Conditional binary evaluation expects 0 (Healthy) and 1 (Late)
    targets_binary = targets.copy()
    targets_binary[targets_binary == 3] = 1

    # Apply conditional binary evaluation
    results = evaluate_model_conditional_binary(logits, targets_binary, group_indices)

    # Add seed and model info
    results['seed'] = seed
    results['model'] = model_name
    results['num_parameters'] = sum(p.numel() for p in model.parameters())

    # Save artifacts
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save logits
    np.save(checkpoint_dir / "test_logits.npy", logits)
    np.save(checkpoint_dir / "test_targets.npy", targets)
    np.save(checkpoint_dir / "test_groups.npy", group_indices)

    # Save complete test results
    with open(checkpoint_dir / "test_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    print(f"Test Accuracy: {results['patch_level']['accuracy']:.4f}")
    print(f"Macro F1: {results['patch_level']['macro_f1']:.4f}")
    print(f"Weighted F1: {results['patch_level']['weighted_f1']:.4f}")
    if 'group_level' in results:
        print(f"Group Accuracy: {results['group_level']['group_accuracy']:.4f}")
        print(f"Num Groups: {results['group_level']['num_groups']}")

    return results


def compute_aggregate_results(model_name: str, seeds: List[int]) -> Dict:
    """Compute aggregate statistics across seeds for a model."""
    print(f"\n{'='*60}")
    print(f"Computing aggregates for {model_name}")
    print(f"{'='*60}")

    aggregate = {}
    baseline_dir = Path(__file__).parent.parent.parent / "artifacts" / "baselines" / model_name

    # Metrics to aggregate
    patch_metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                     'weighted_precision', 'weighted_recall', 'weighted_f1',
                     'auc_roc_macro', 'healthy_precision', 'healthy_recall', 'healthy_f1',
                     'late_precision', 'late_recall', 'late_f1']

    group_metrics = ['group_accuracy', 'group_macro_precision', 'group_macro_recall',
                     'group_macro_f1', 'group_healthy_precision', 'group_healthy_recall',
                     'group_healthy_f1', 'group_late_precision', 'group_late_recall',
                     'group_late_f1']

    for metric in patch_metrics:
        values = []
        for seed in seeds:
            result_path = baseline_dir / f"seed_{seed}" / "test_results.json"
            if result_path.exists():
                with open(result_path, 'r') as f:
                    data = json.load(f)
                if 'patch_level' in data and metric in data['patch_level']:
                    val = data['patch_level'][metric]
                    if val is not None:
                        values.append(val)

        if values:
            aggregate[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }

    for metric in group_metrics:
        values = []
        for seed in seeds:
            result_path = baseline_dir / f"seed_{seed}" / "test_results.json"
            if result_path.exists():
                with open(result_path, 'r') as f:
                    data = json.load(f)
                if 'group_level' in data and metric in data['group_level']:
                    val = data['group_level'][metric]
                    if val is not None:
                        values.append(val)

        if values:
            aggregate[f'group_{metric}'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }

    # Add metadata
    # Load model config to get parameter count
    model_kwargs = MODEL_DEFAULT_CONFIGS.get(model_name)
    if model_kwargs is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_DEFAULT_CONFIGS.keys())}")
    ModelClass = MODEL_CLASS_MAP.get(model_name)
    if ModelClass is None:
        raise ValueError(f"Unknown model class: {model_name}. Available: {list(MODEL_CLASS_MAP.keys())}")
    model = ModelClass(**model_kwargs)
    aggregate['parameters'] = sum(p.numel() for p in model.parameters())
    aggregate['seeds'] = seeds
    aggregate['num_seeds'] = len(seeds)

    # Save aggregate results
    with open(baseline_dir / "aggregate_results.json", 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f"Aggregate results saved to {baseline_dir / 'aggregate_results.json'}")
    for key, val in aggregate.items():
        if isinstance(val, dict) and 'mean' in val:
            print(f"  {key}: {val['mean']:.4f} ± {val['std']:.4f} ({val['values']})")

    return aggregate


# Model name mapping
MODEL_NAME_MAP = {
    'simple_cnn': 'simplecnn',
    'mlp_mixer_hsi': 'mlp_mixer',
    'resnet18_hsi': 'resnet18',
    'simplecnn': 'simplecnn',
    'mlp_mixer': 'mlp_mixer',
    'resnet18': 'resnet18',
}

def main():
    parser = argparse.ArgumentParser(description='Re-evaluate saved baseline models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['simple_cnn', 'mlp_mixer_hsi', 'resnet18_hsi', 'all'],
                        help='Model to evaluate')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 999],
                        help='Seeds to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation, only compute aggregates')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load frozen manifest
    manifest = load_frozen_manifest()
    print(f"Loaded frozen manifest: {manifest['test_patches']} test patches, "
          f"{len(manifest['test_groups'])} test groups")

    # Determine models to evaluate
    if args.model == 'all':
        models = ['simple_cnn', 'mlp_mixer_hsi', 'resnet18_hsi']
    else:
        models = [args.model]

    # Map model names for get_model function
    model_names_for_load = [MODEL_NAME_MAP[m] for m in models]

    # Load test dataset once (shared across models)
    print("\nLoading test dataset...")
    config = get_experiment_config('simple_cnn')  # Use simple_cnn config for data loading
    config['data_root'] = "C:/Users/ayush/OneDrive/Desktop/DMLPFFN/tomato_leaf"
    # seed is handled by the config module, not passed as parameter

    X_test, y_test, split_manifest = load_test_dataset_with_groups(config, manifest)
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Test labels: {np.unique(y_test, return_counts=True)}")

    # Extract group indices
    print("Extracting group indices...")
    group_indices = extract_group_indices_from_manifest(split_manifest, X_test, config)
    print(f"Group indices shape: {group_indices.shape}")
    print(f"Unique groups: {np.unique(group_indices)}")
    for g in np.unique(group_indices):
        print(f"  Group {g}: {np.sum(group_indices == g)} patches")

    # Evaluate each model
    for i, model_name in enumerate(models):
        load_model_name = model_names_for_load[i]
        print(f"\n{'#'*60}")
        print(f"# Model: {model_name}")
        print(f"{'#'*60}")

        experiment_config = get_experiment_config(model_name)
        # Add data_root to the experiment config for model loading
        experiment_config['data_root'] = config['data_root']

        if not args.skip_eval:
            for seed in args.seeds:
                try:
                    evaluate_model(load_model_name, model_name, seed, experiment_config, manifest,
                                 X_test, y_test, group_indices, device)
                except Exception as e:
                    print(f"Error evaluating {model_name} seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()

        # Compute aggregates
        try:
            compute_aggregate_results(model_name, args.seeds)
        except Exception as e:
            print(f"Error computing aggregates for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\nRe-evaluation complete!")


if __name__ == '__main__':
    main()