"""
Phase 5: Baseline Comparisons & Ablation Studies

This script runs comprehensive baseline model comparisons and DMLPFFN ablation studies
using the corrected, scientifically valid pipeline (no data leakage).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Add CODES to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from dmlpffn_disease import DMLPFFN
from training_pipeline_corrected import train_model, evaluate_model
from CODES.baselines.factory import (
    get_baseline_model, count_parameters, get_ablation_model, ABLATION_CONFIGS
)

# Import data loading from corrected main
from main_experiment_corrected import load_tomato_dataset_corrected
from data_preprocessing_corrected import HSIDataset
from torch.utils.data import DataLoader


# =============================================================================
# Utility Functions
# =============================================================================

def compute_flops(model, input_shape=(1, 128, 9, 9)):
    """Estimate FLOPs for a model."""
    try:
        from thop import profile
        input_tensor = torch.randn(input_shape)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops, params
    except ImportError:
        # Fallback: rough estimation based on parameters
        total_params = sum(p.numel() for p in model.parameters())
        # Very rough estimate: 2 * params * spatial_size for conv
        return total_params * 2 * 9 * 9, total_params


def measure_inference_time(model, dataloader, device, num_batches=10):
    """Measure average inference time per batch."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (xb, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            xb = xb.to(device)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.perf_counter()
            _ = model(xb)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append(time.perf_counter() - start)
    return np.mean(times) * 1000  # ms per batch


def run_single_experiment(model, model_name, dataloaders, device, cfg, seed=42):
    """Run a single model training and evaluation."""
    print(f"\n{'='*60}")
    print(f"Running: {model_name}")
    print(f"{'='*60}")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train
    start_time = time.time()
    try:
        trained_model, history = train_model(
            dataloaders=dataloaders,
            in_ch=cfg.pca_components,
            num_classes=cfg.num_classes,
            optimizer_type=cfg.optimizer,
            loss_type=cfg.loss_function,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            patience=cfg.patience,
            scheduler_type=cfg.scheduler,
            scheduler_params=cfg.scheduler_params,
            device=device,
            use_weighted_sampler=True,
            label_smoothing=cfg.label_smoothing,
            use_mixup=cfg.use_mixup,
            mixup_alpha=cfg.mixup_alpha,
            use_cutmix=cfg.use_cutmix,
            cutmix_alpha=cfg.cutmix_alpha,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma
        )
        train_time = time.time() - start_time
    except Exception as e:
        print(f"Training failed for {model_name}: {e}")
        return None

    # Evaluate on test set
    report, cm = evaluate_model(trained_model, dataloaders['test'], device)

    # Compute metrics
    best_val_acc = max(history['val_acc'])
    best_val_epoch = np.argmax(history['val_acc']) + 1
    final_train_acc = history['train_acc'][-1]
    train_val_gap = final_train_acc - best_val_acc

    # Count parameters
    total_params, trainable_params = count_parameters(trained_model)

    # Estimate FLOPs
    flops, _ = compute_flops(trained_model)

    # Measure inference time
    inf_time = measure_inference_time(trained_model, dataloaders['test'], device)

    results = {
        'model_name': model_name,
        'seed': seed,
        'train_time_seconds': train_time,
        'best_val_acc': float(best_val_acc),
        'best_val_epoch': int(best_val_epoch),
        'final_train_acc': float(final_train_acc),
        'train_val_gap': float(train_val_gap),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'flops_estimate': float(flops),
        'inference_time_ms': float(inf_time),
        'history': history,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

    print(f"\n{model_name} Results:")
    print(f"  Best Val Acc: {best_val_acc:.4f} (epoch {best_val_epoch})")
    print(f"  Train/Val Gap: {train_val_gap:.4f}")
    print(f"  Params: {total_params:,}")
    print(f"  FLOPs: {flops:,.0f}")
    print(f"  Inference: {inf_time:.2f} ms/batch")
    print(f"  Train Time: {train_time:.1f}s")

    return results


def run_ablation_study(base_model_class, ablation_configs, dataloaders, device, cfg, seed=42):
    """Run ablation study with different model configurations."""
    results = []

    for config_name, config in ablation_configs.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {config_name}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = get_ablation_model(config, in_channels=cfg.pca_components,
                                   num_classes=cfg.num_classes, dropout=cfg.dropout)

        result = run_single_experiment(model, f"DMLPFFN_{config_name}", dataloaders, device, cfg, seed)
        if result:
            result['ablation_config'] = config_name
            result['ablation_details'] = config
            results.append(result)

    return results


# =============================================================================
# Main Phase 5 Experiment
# =============================================================================

def run_phase5_experiment():
    """Run complete Phase 5: Baseline Comparisons & Ablation Studies."""

    print("="*80)
    print("PHASE 5: BASELINE COMPARISONS & ABLATION STUDIES")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data with corrected pipeline
    print("\nLoading data with corrected pipeline (group-wise split, sensor-specific PCA)...")
    from main_experiment_corrected import load_tomato_dataset_corrected
    from data_preprocessing_corrected import HSIDataset
    from torch.utils.data import DataLoader

    splits = load_tomato_dataset_corrected(
        root=cfg.data_root,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        save_artifacts_path=None,
        save_manifest_path=None
    )

    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    train_dataset = HSIDataset(X_train, y_train, augment=True)
    val_dataset = HSIDataset(X_val, y_val, augment=False)
    test_dataset = HSIDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == 'cuda'))

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    print(f"Train: {len(dataloaders['train'].dataset)} samples")
    print(f"Val:   {len(dataloaders['val'].dataset)} samples")
    print(f"Test:  {len(dataloaders['test'].dataset)} samples")

    all_results = {
        'phase': 5,
        'experiment': 'baseline_comparisons_ablation',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'seed': cfg.seed,
        'dataset_info': {
            'train_samples': len(dataloaders['train'].dataset),
            'val_samples': len(dataloaders['val'].dataset),
            'test_samples': len(dataloaders['test'].dataset),
            'num_classes': cfg.num_classes,
            'pca_components': cfg.pca_components
        },
        'baselines': [],
        'ablations': []
    }

    # -------------------------------------------------------------------------
    # BASELINE MODEL COMPARISONS
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("BASELINE MODEL COMPARISONS")
    print("="*80)

    baseline_models = [
        ('SimpleCNN', 'simple_cnn'),
        ('ResNet-18', 'resnet18_hsi'),
        ('MLP-Mixer', 'mlp_mixer_hsi'),
        ('SVM (RBF)', 'svm'),
        ('Random Forest', 'random_forest'),
        ('DMLPFFN (Full)', 'dmlpffn_full'),  # We'll handle this specially
    ]

    # For DMLPFFN, use the original model class
    dmlpffn_model = DMLPFFN(in_channels=cfg.pca_components, num_classes=cfg.num_classes)

    for display_name, model_key in baseline_models:
        if model_key == 'dmlpffn_full':
            model = dmlpffn_model
        else:
            model = get_baseline_model(model_key, in_channels=cfg.pca_components,
                                       num_classes=cfg.num_classes, dropout=cfg.dropout)

        result = run_single_experiment(model, display_name, dataloaders, device, cfg, cfg.seed)
        if result:
            all_results['baselines'].append(result)

    # -------------------------------------------------------------------------
    # ABLATION STUDIES
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("DMLPFFN ABLATION STUDIES")
    print("="*80)

    ablation_results = run_ablation_study(
        DMLPFFN, ABLATION_CONFIGS, dataloaders, device, cfg, cfg.seed
    )
    all_results['ablations'] = ablation_results

    # -------------------------------------------------------------------------
    # SUMMARY & SAVE
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PHASE 5 SUMMARY")
    print("="*80)

    # Sort baselines by validation accuracy
    baselines_sorted = sorted(all_results['baselines'], key=lambda x: x['best_val_acc'], reverse=True)

    print("\nBaseline Model Rankings (by Val Accuracy):")
    print(f"{'Rank':<5} {'Model':<25} {'Val Acc':<10} {'Gap':<10} {'Params':<12} {'FLOPs':<15} {'Inf(ms)':<10}")
    print("-"*90)
    for i, r in enumerate(baselines_sorted):
        print(f"{i+1:<5} {r['model_name']:<25} {r['best_val_acc']:.4f}     "
              f"{r['train_val_gap']:.4f}     {r['total_params']:>10,}    "
              f"{r['flops_estimate']:>12,.0f}   {r['inference_time_ms']:>7.2f}")

    # Ablation summary
    print("\nAblation Study Results:")
    print(f"{'Config':<20} {'Val Acc':<10} {'Gap':<10} {'Params':<12} {'Notes'}")
    print("-"*70)
    for r in all_results['ablations']:
        config = r.get('ablation_config', 'unknown')
        print(f"{config:<20} {r['best_val_acc']:.4f}     {r['train_val_gap']:.4f}     "
              f"{r['total_params']:>10,}    {r.get('ablation_details', {})}")

    # Save results
    artifacts_dir = Path(__file__).parent.parent / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)

    output_file = artifacts_dir / f"phase5_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    all_results = convert(all_results)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\nPhase 5 Complete!")

    return all_results


if __name__ == '__main__':
    run_phase5_experiment()