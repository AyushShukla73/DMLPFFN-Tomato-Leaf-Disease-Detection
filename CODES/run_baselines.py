"""
Phase 5 Baseline Experiments - Main Orchestrator
Runs all 5 baselines + DMLPFFN reference using frozen split and Phase 3 protocol.
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
import joblib

# Add CODES to path
sys.path.insert(0, str(Path(__file__).parent))

from config_baselines import (
    EXPERIMENTS,
    DMLPFFN_REFERENCE,
    SEEDS,
    DEVICE,
    USE_AMP,
    VRAM_LIMIT_GB,
    SPLIT_MANIFEST_PATH,
    CLASS_HEALTHY,
    CLASS_LATE,
    BINARY_CLASS_MAP,
    get_experiment_config,
    get_all_experiment_names,
    get_dl_experiment_names,
    get_classical_experiment_names,
    EVALUATION_CONFIG,
    ARTIFACT_STRUCTURE,
)

from baselines.trainer import BaselineTrainer, get_phase3_config, create_weighted_sampler
from torch.utils.data import DataLoader
from baselines.simple_cnn import SimpleCNN
from baselines.resnet18_hsi import ResNet18_HSI
from baselines.mlp_mixer_hsi import MLPMixer_HSI
from baselines.classical_ml import ClassicalMLBaselines

from data_preprocessing_corrected import load_tomato_dataset_corrected, HSIDataset

from evaluation.metrics import (
    evaluate_model_conditional_binary,
    conditional_binary_predictions,
    conditional_binary_probs,
)
from evaluation.statistical_tests import (
    group_bootstrap_ci,
    mcnemar_group_aware,
    compare_models_seeds,
    aggregate_seed_results,
)
from evaluation.compare import (
    load_baseline_results,
    load_dmlpffn_reference,
    format_comparison_table,
    format_statistical_comparison,
    format_per_seed_results,
    generate_full_report,
)

warnings.filterwarnings("ignore")


def load_preprocessing_artifacts():
    """Load frozen preprocessing artifacts from Phase 3."""
    # Artifacts are in parent directory (project root)
    project_root = Path(__file__).parent.parent
    pca_scaler_path = project_root / "artifacts" / "preprocessing_artifacts_sklearn.joblib"
    arrays_path = project_root / "artifacts" / "preprocessing_artifacts_arrays.npz"

    if not pca_scaler_path.exists():
        raise FileNotFoundError(f"Preprocessing artifacts not found at {pca_scaler_path}")

    artifacts = joblib.load(pca_scaler_path)
    arrays = np.load(arrays_path)

    return {
        'pca_swir': artifacts['pca_swir_168'],
        'pca_visnir': artifacts['pca_visnir_300'],
        'scaler': artifacts['scaler'],
        'wavelengths_swir': arrays['wavelengths_swir'],
        'wavelengths_visnir': arrays['wavelengths_visnir'],
        'removed_band_indices': np.concatenate([arrays['remove_bands_swir'], arrays['remove_bands_visnir']]),
    }


def load_split_manifest():
    """Load frozen split manifest (seed=42)."""
    project_root = Path(__file__).parent.parent
    manifest_path = project_root / SPLIT_MANIFEST_PATH
    with open(manifest_path) as f:
        return json.load(f)


def create_data_loaders(artifacts: Dict, split_manifest: Dict, config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test data loaders using frozen preprocessing artifacts and split manifest.
    Uses load_tomato_dataset_corrected to get pre-split data, then wraps in HSIDataset and DataLoader.
    """
    project_root = Path(__file__).parent.parent
    artifacts_path = project_root / "artifacts" / "preprocessing_artifacts"
    manifest_path = project_root / SPLIT_MANIFEST_PATH

    # Load pre-split data using corrected pipeline
    splits = load_tomato_dataset_corrected(
        patch_size=9,
        stride=4,
        load_artifacts_path=str(artifacts_path),
        save_manifest_path=str(manifest_path),
    )

    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    print(f"  Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Create datasets
    train_dataset = HSIDataset(X_train, y_train, augment=True)
    val_dataset = HSIDataset(X_val, y_val, augment=False)
    test_dataset = HSIDataset(X_test, y_test, augment=False)

    # Create weighted sampler for training
    # For CPU mode, pin_memory should be False
    use_weighted_sampler = getattr(config, 'use_weighted_sampler', True) if hasattr(config, '__dict__') else config.get('use_weighted_sampler', True)
    batch_size = getattr(config, 'batch_size', 16) if hasattr(config, '__dict__') else config.get('batch_size', 16)
    num_workers = 0  # Always use 0 workers to avoid multiprocessing issues
    pin_memory = False  # Always False for CPU mode to avoid memory issues

    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_dataset, num_classes=4)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def create_model(model_name: str, config) -> torch.nn.Module:
    """Create model instance from config."""
    if model_name == "simple_cnn":
        return SimpleCNN(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            num_classes=config.num_classes,
        )
    elif model_name == "resnet18_hsi":
        return ResNet18_HSI(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
        )
    elif model_name == "mlp_mixer_hsi":
        return MLPMixer_HSI(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            patch_size=config.patch_size,
            embedding_dim=config.embedding_dim,
            num_blocks=config.num_blocks,
            token_mixing_dim=config.token_mixing_dim,
            channel_mixing_dim=config.channel_mixing_dim,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown DL model: {model_name}")


def train_dl_baseline(
    model_name: str,
    seed: int,
    artifacts: Dict,
    split_manifest: Dict,
    output_dir: Path,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Train a single DL baseline with a specific seed."""
    exp_config = get_experiment_config(model_name)
    model_config = exp_config["config"]
    training_config = exp_config["training_config"]

    # Set all seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create model
    model = create_model(model_name, model_config)

    # Create data loaders using frozen artifacts and manifest
    train_loader, val_loader, test_loader = create_data_loaders(
        artifacts, split_manifest, training_config
    )

    # Initialize trainer with Phase 3 config and data loaders
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=training_config,
        device=device,
        output_dir=output_dir,
        seed=seed,
    )

    # Train - train() already evaluates on test set and returns results
    train_results = trainer.train()

    # Compute additional metrics from test predictions/targets
    from sklearn.metrics import precision_recall_fscore_support
    y_true = train_results['test_targets']
    y_pred = train_results['test_predictions']

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    return {
        "history": train_results['history'],
        "test_results": {
            'test_accuracy': train_results['test_acc'],
            'val_accuracy': train_results['best_val_acc'],
            'test_loss': train_results['test_loss'],
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'test_predictions': train_results['test_predictions'],
            'test_targets': train_results['test_targets'],
        },
        "best_epoch": train_results['best_epoch'],
        "best_val_acc": train_results['best_val_acc'],
    }


def run_classical_baseline(
    model_name: str,
    artifacts: Dict,
    split_manifest: Dict,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run classical ML baseline (SVM or RF) - single deterministic run."""
    exp_config = get_experiment_config(model_name)
    model_config = exp_config["config"]

    # Load data using the frozen preprocessing artifacts and split manifest
    project_root = Path(__file__).parent.parent
    artifacts_path = project_root / "artifacts" / "preprocessing_artifacts"
    manifest_path = project_root / SPLIT_MANIFEST_PATH

    splits = load_tomato_dataset_corrected(
        patch_size=9,
        stride=4,
        load_artifacts_path=str(artifacts_path),
        save_manifest_path=str(manifest_path),
    )

    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    print(f"  Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Prepare data for classical ML (flatten NHWC to [N, 10368])
    from baselines.classical_ml import prepare_classical_ml_data, train_svm, train_rf, save_classical_model

    X_train_flat, y_train_flat = prepare_classical_ml_data(X_train, y_train)
    X_val_flat, y_val_flat = prepare_classical_ml_data(X_val, y_val)
    X_test_flat, y_test_flat = prepare_classical_ml_data(X_test, y_test)

    if model_name == "svm":
        results = train_svm(
            X_train_flat, y_train_flat,
            X_val_flat, y_val_flat,
            pca_components=model_config.pca_components,
        )
        model = results['model']
    elif model_name == "random_forest":
        results = train_rf(
            X_train_flat, y_train_flat,
            X_val_flat, y_val_flat,
            pca_components=model_config.pca_components,
        )
        model = results['model']
    else:
        raise ValueError(f"Unknown classical model: {model_name}")

    # Evaluate on test set
    y_pred = model.predict(X_test_flat)
    y_proba = model.predict_proba(X_test_flat)

    test_accuracy = (y_pred == y_test_flat).mean()

    # For conditional binary evaluation, we need to extract Healthy/Late classes
    # Map 4-class to binary: Healthy(0)->0, Late(3)->1
    from evaluation.metrics import conditional_binary_predictions, evaluate_model_conditional_binary

    # Since classical ML outputs 4 classes, we need to map to binary
    # The predictions are 0,1,2,3 -> map to binary
    y_test_binary = np.where(y_test_flat == 0, 0, 1)  # Healthy=0, Late=1 (only these two in test)
    y_pred_binary = np.where(y_pred == 0, 0, 1)

    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test_binary, y_pred_binary, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test_binary, y_pred_binary, average='weighted', zero_division=0
    )

    # Save model
    save_classical_model(model, output_dir / "model.joblib")

    test_results = {
        'test_accuracy': float(test_accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'test_predictions': y_pred_binary.tolist(),
        'test_targets': y_test_binary.tolist(),
        'test_logits': y_proba.tolist(),  # probabilities
    }

    return {
        "train_results": results,
        "test_results": test_results,
    }


def save_seed_artifacts(
    model_name: str,
    seed: int,
    results: Dict,
    output_dir: Path,
    model: Optional[torch.nn.Module] = None,
):
    """Save per-seed artifacts."""
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Save test results
    with open(seed_dir / "test_results.json", "w") as f:
        json.dump(results["test_results"], f, indent=2, default=str)

    # Save history (DL only)
    if "history" in results:
        with open(seed_dir / "history.json", "w") as f:
            json.dump(results["history"], f, indent=2, default=str)

    # Save model weights (DL only)
    if model is not None:
        torch.save(model.state_dict(), seed_dir / "best_model.pth")

    # Save logits and targets for re-evaluation
    if "test_logits" in results["test_results"]:
        np.save(seed_dir / "test_logits.npy", results["test_results"]["test_logits"])
    if "test_targets" in results["test_results"]:
        np.save(seed_dir / "test_targets.npy", results["test_results"]["test_targets"])
    if "test_groups" in results["test_results"]:
        np.save(seed_dir / "test_groups.npy", results["test_results"]["test_groups"])


def compute_aggregate_results(model_name: str, output_dir: Path) -> Dict:
    """Compute aggregate results across seeds."""
    seed_dirs = sorted(output_dir.glob("seed_*"))
    if not seed_dirs:
        return {}

    all_test_results = []
    all_logits = []
    all_targets = []
    all_groups = []

    for seed_dir in seed_dirs:
        test_path = seed_dir / "test_results.json"
        if test_path.exists():
            with open(test_path) as f:
                test_results = json.load(f)
            all_test_results.append(test_results)

            # Load logits for re-evaluation
            logits_path = seed_dir / "test_logits.npy"
            if logits_path.exists():
                all_logits.append(np.load(logits_path))
            targets_path = seed_dir / "test_targets.npy"
            if targets_path.exists():
                all_targets.append(np.load(targets_path))
            groups_path = seed_dir / "test_groups.npy"
            if groups_path.exists():
                all_groups.append(np.load(groups_path))

    if not all_test_results:
        return {}

    # Aggregate metrics
    metric_keys = [
        "val_accuracy", "test_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1",
    ]

    aggregate = {}
    for key in metric_keys:
        values = [r.get(key) for r in all_test_results if key in r and r[key] is not None]
        if values:
            aggregate[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "values": values,
            }

    # Get parameters from first seed's model
    first_seed_dir = seed_dirs[0]
    model_path = first_seed_dir / "best_model.pth"
    if model_path.exists():
        # We'll get params from model config
        exp_config = get_experiment_config(model_name)
        aggregate["parameters"] = exp_config["config"].expected_params

    # Train/val gap
    if "val_accuracy" in aggregate and "test_accuracy" in aggregate:
        aggregate["train_val_gap"] = abs(
            aggregate["val_accuracy"]["mean"] - aggregate["test_accuracy"]["mean"]
        )

    # Group-aware bootstrap CI on test accuracy
    if all_logits and all_targets and all_groups:
        # Use first seed's logits for bootstrap (same test set across seeds)
        y_true = all_targets[0]
        logits = all_logits[0]
        groups = all_groups[0]

        y_pred = conditional_binary_predictions(logits)

        metric_fn = lambda yt, yp: np.mean(yt == yp)
        ci_result = group_bootstrap_ci(y_true, y_pred, groups, metric_fn, n_resamples=1000)
        aggregate["test_accuracy_ci"] = {
            "mean": ci_result["mean"],
            "std": ci_result["std"],
            "ci_lower": ci_result["ci_lower"],
            "ci_upper": ci_result["ci_upper"],
            "n_groups": ci_result["n_groups"],
            "n_resamples": ci_result["n_resamples"],
        }

    return aggregate


def save_aggregate_results(model_name: str, aggregate: Dict, output_dir: Path):
    """Save aggregate results."""
    agg_path = output_dir / "aggregate_results.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)


def run_single_experiment(
    model_name: str,
    artifacts: Dict,
    split_manifest: Dict,
    base_output_dir: Path,
) -> Dict[str, Any]:
    """Run a single experiment (all seeds for DL, single run for classical)."""
    exp_config = get_experiment_config(model_name)
    is_dl = exp_config["is_dl"]
    seeds = exp_config["seeds"]
    output_dir = base_output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {model_name} ({'DL' if is_dl else 'Classical'})")
    print(f"{'='*60}")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_dir}")

    if is_dl:
        # Deep Learning: 5 seeds
        all_seed_results = []

        for seed in seeds:
            print(f"\n  Seed {seed}...")
            seed_output_dir = output_dir
            try:
                results = train_dl_baseline(
                    model_name=model_name,
                    seed=seed,
                    artifacts=artifacts,
                    split_manifest=split_manifest,
                    output_dir=seed_output_dir,
                )
                all_seed_results.append(results)

                # Save per-seed artifacts
                model = create_model(model_name, exp_config["config"])
                # Load best weights
                best_model_path = seed_output_dir / f"seed_{seed}" / "best_model.pth"
                if best_model_path.exists():
                    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
                save_seed_artifacts(model_name, seed, results, output_dir, model)

                test_acc = results['test_results'].get('test_accuracy', 'N/A')
                macro_f1 = results['test_results'].get('macro_f1', 'N/A')
                print(f"    Test Acc: {test_acc:.4f}" if isinstance(test_acc, (int, float)) else f"    Test Acc: {test_acc}")
                print(f"    Macro F1: {macro_f1:.4f}" if isinstance(macro_f1, (int, float)) else f"    Macro F1: {macro_f1}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Compute and save aggregate
        aggregate = compute_aggregate_results(model_name, output_dir)
        save_aggregate_results(model_name, aggregate, output_dir)
        agg_test_acc = aggregate.get('test_accuracy', {})
        agg_mean = agg_test_acc.get('mean', 'N/A')
        agg_std = agg_test_acc.get('std', 'N/A')
        mean_str = f"{agg_mean:.4f}" if isinstance(agg_mean, (int, float)) else str(agg_mean)
        std_str = f"{agg_std:.4f}" if isinstance(agg_std, (int, float)) else str(agg_std)
        print(f"\n  Aggregate Test Acc: {mean_str} +/- {std_str}")

        return {"seeds": all_seed_results, "aggregate": aggregate}

    else:
        # Classical ML: single run
        print(f"  Training...")
        results = run_classical_baseline(
            model_name=model_name,
            artifacts=artifacts,
            split_manifest=split_manifest,
            output_dir=output_dir,
        )

        # Save test results
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(results["test_results"], f, indent=2, default=str)

        # Save logits/targets/groups for statistical tests
        if "test_logits" in results["test_results"]:
            np.save(output_dir / "test_logits.npy", results["test_results"]["test_logits"])
        if "test_targets" in results["test_results"]:
            np.save(output_dir / "test_targets.npy", results["test_results"]["test_targets"])
        if "test_groups" in results["test_results"]:
            np.save(output_dir / "test_groups.npy", results["test_results"]["test_groups"])

        # Create aggregate (single run)
        aggregate = {}
        test_results = results["test_results"]
        for key in ["test_accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "weighted_precision", "weighted_recall", "weighted_f1"]:
            if key in test_results:
                aggregate[key] = {"mean": test_results[key], "std": 0.0, "values": [test_results[key]]}

        save_aggregate_results(model_name, aggregate, output_dir)

        test_acc = test_results.get('test_accuracy', 'N/A')
        macro_f1 = test_results.get('macro_f1', 'N/A')
        print(f"  Test Acc: {test_acc:.4f}" if isinstance(test_acc, (int, float)) else f"  Test Acc: {test_acc}")
        print(f"  Macro F1: {macro_f1:.4f}" if isinstance(macro_f1, (int, float)) else f"  Macro F1: {macro_f1}")

        return {"seeds": [results], "aggregate": aggregate}


def run_statistical_analysis(results_dir: Path) -> Dict:
    """Run statistical comparison across all models."""
    print(f"\n{'='*60}")
    print("Running Statistical Analysis")
    print(f"{'='*60}")

    # Load all results
    all_results = load_baseline_results(str(results_dir))

    # Load test groups from split manifest
    split_manifest = load_split_manifest()
    test_groups = split_manifest["test_groups"]

    # Build group mapping for test patches
    # We need to load the actual groups from a results file
    first_model = next(iter(all_results))
    if "seeds" in all_results[first_model] and all_results[first_model]["seeds"]:
        first_seed = all_results[first_model]["seeds"][0]
        if "test_results" in first_seed and "test_groups" in first_seed["test_results"]:
            groups = np.array(first_seed["test_results"]["test_groups"])
        else:
            # Fallback: create group mapping from test_groups count
            groups = None

    if groups is not None:
        # Prepare results for statistical comparison
        model_data = {}
        for model_name, model_results in all_results.items():
            if "seeds" in model_results:
                seed_data = []
                for seed_result in model_results["seeds"]:
                    if "test_results" in seed_result:
                        tr = seed_result["test_results"]
                        if "targets" in tr and "predictions" in tr:
                            seed_data.append({
                                "targets": tr["targets"],
                                "predictions": tr["predictions"],
                            })
                if seed_data:
                    model_data[model_name] = seed_data

        if model_data:
            comparison = compare_models_seeds(model_data, groups, metric_name="accuracy")
            return comparison

    return {}


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Baseline Experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["simple_cnn", "mlp_mixer_hsi", "resnet18_hsi", "svm", "random_forest", "all"],
        default=["all"],
        help="Models to run",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only generate report from existing results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/baselines",
        help="Base output directory",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate final report",
    )
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine models to run
    if "all" in args.models:
        models_to_run = get_all_experiment_names()
    else:
        models_to_run = args.models

    print("="*60)
    print("PHASE 5 BASELINE EXPERIMENTS")
    print("="*60)
    print(f"Models: {models_to_run}")
    print(f"Output: {base_output_dir}")
    print(f"Device: {DEVICE}")
    print(f"Mixed Precision: {USE_AMP}")
    print(f"VRAM Limit: {VRAM_LIMIT_GB} GB")
    print(f"Seeds (DL): {SEEDS}")

    if args.report_only or args.skip_training:
        print("\nSkipping training - generating report only...")
    else:
        # Load frozen artifacts
        print("\nLoading preprocessing artifacts...")
        artifacts = load_preprocessing_artifacts()
        print("  PCA SWIR:", artifacts['pca_swir'].n_components_)
        print("  PCA VIS-NIR:", artifacts['pca_visnir'].n_components_)

        print("\nLoading split manifest...")
        split_manifest = load_split_manifest()
        print(f"  Train: {split_manifest['train_patches']} patches, {len(split_manifest['train_groups'])} groups")
        print(f"  Val: {split_manifest['val_patches']} patches, {len(split_manifest['val_groups'])} groups")
        print(f"  Test: {split_manifest['test_patches']} patches, {len(split_manifest['test_groups'])} groups")
        print(f"  Test groups: {split_manifest['test_groups']}")
        print(f"  Group overlap check: {split_manifest['group_overlap_check']}")

    all_results = {}

    if not args.report_only and not args.skip_training:
        # Run experiments in order
        for model_name in models_to_run:
            try:
                results = run_single_experiment(
                    model_name=model_name,
                    artifacts=artifacts,
                    split_manifest=split_manifest,
                    base_output_dir=base_output_dir,
                )
                all_results[model_name] = results
            except Exception as e:
                print(f"\nERROR running {model_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[model_name] = {"error": str(e)}

        # Run statistical analysis
        try:
            statistical_results = run_statistical_analysis(base_output_dir)
        except Exception as e:
            print(f"\nERROR in statistical analysis: {e}")
            statistical_results = {}
    else:
        # Load existing results
        all_results = load_baseline_results(str(base_output_dir))
        statistical_results = {}

    # Generate final report
    print(f"\n{'='*60}")
    print("Generating Final Report")
    print(f"{'='*60}")

    report_path = base_output_dir / "PHASE5_BASELINE_RESULTS.md"
    generate_full_report(
        results_dir=str(base_output_dir),
        output_path=str(report_path),
        statistical_results=statistical_results if statistical_results else None,
    )

    print(f"\nReport saved to: {report_path}")
    print("\nDONE!")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    dmlpffn = load_dmlpffn_reference()
    print(f"DMLPFFN (Phase 3): Test Acc = {dmlpffn['test_accuracy']:.4f}, Macro F1 = {dmlpffn['macro_f1']:.4f}")

    for model_name in models_to_run:
        if model_name in all_results and "aggregate" in all_results[model_name]:
            agg = all_results[model_name]["aggregate"]
            test_acc = agg.get("test_accuracy", {})
            macro_f1 = agg.get("macro_f1", {})
            if isinstance(test_acc, dict) and "mean" in test_acc:
                mean_val = test_acc['mean']
                std_val = test_acc.get('std', 0)
                f1_mean = macro_f1.get('mean', 0)
                f1_std = macro_f1.get('std', 0)
                mean_str = f"{mean_val:.4f}" if isinstance(mean_val, (int, float)) else str(mean_val)
                std_str = f"{std_val:.4f}" if isinstance(std_val, (int, float)) else str(std_val)
                f1_mean_str = f"{f1_mean:.4f}" if isinstance(f1_mean, (int, float)) else str(f1_mean)
                f1_std_str = f"{f1_std:.4f}" if isinstance(f1_std, (int, float)) else str(f1_std)
                print(f"{model_name}: Test Acc = {mean_str} +/- {std_str}, Macro F1 = {f1_mean_str} +/- {f1_std_str}")


if __name__ == "__main__":
    main()