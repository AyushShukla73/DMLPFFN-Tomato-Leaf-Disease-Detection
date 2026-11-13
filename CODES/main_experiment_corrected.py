"""
Corrected Main Experiment for DMLPFFN

This script uses the scientifically valid preprocessing pipeline that addresses:
1. Per-image PCA leakage (PCA fitted on training only)
2. Patch-level split leakage (group-wise split by biological unit)
3. Normalization leakage (scaler fitted on training patches only)

Also implements:
- Sensor-specific PCA (separate for 168-band SWIR and 300-band VIS-NIR)
- Reproducible training with logging
- CPU/GPU configuration
- Held-out test evaluation
- Artifact serialization for reproducibility
"""

import os
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from config import cfg
from data_preprocessing_corrected import (
    load_tomato_dataset_corrected,
    HSIDataset,
    PreprocessingArtifacts,
    SplitManifest
)
from training_pipeline_corrected import train_model
from evaluation_metrics import evaluate_model
from dmlpffn_disease import DMLPFFN


def set_reproducible_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """Get compute device based on config."""
    if cfg.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def run_corrected_experiment():
    """Run the corrected experiment with valid preprocessing."""
    # Set reproducibility
    set_reproducible_seed(cfg.seed)

    # Device configuration
    device = get_device()

    # Paths for artifacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    artifacts_path = os.path.join(artifacts_dir, 'preprocessing_artifacts')
    manifest_path = os.path.join(artifacts_dir, 'split_manifest.json')
    results_path = os.path.join(artifacts_dir, 'experiment_results.json')
    model_path = os.path.join(artifacts_dir, 'best_model.pth')

    print("=" * 70)
    print("CORRECTED DMLPFFN EXPERIMENT - SCIENTIFICALLY VALID PIPELINE")
    print("=" * 70)
    print(f"Seed: {cfg.seed}")
    print(f"Device: {device}")
    print(f"Artifacts dir: {artifacts_dir}")
    print()

    # =====================================================================
    # Step 1: Load data with corrected preprocessing
    # =====================================================================
    print("Step 1: Loading dataset with corrected preprocessing...")
    print("-" * 70)

    splits = load_tomato_dataset_corrected(
        root=cfg.data_root,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        save_artifacts_path=artifacts_path,
        save_manifest_path=manifest_path
    )

    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    print(f"\nDataset shapes:")
    print(f"  Train: {X_train.shape} ({len(np.unique(y_train))} classes)")
    print(f"  Val:   {X_val.shape} ({len(np.unique(y_val))} classes)")
    print(f"  Test:  {X_test.shape} ({len(np.unique(y_test))} classes)")

    # Check class distributions
    for name, (_, y) in splits.items():
        unique, counts = np.unique(y, return_counts=True)
        print(f"  {name} class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # =====================================================================
    # Step 2: Create DataLoaders
    # =====================================================================
    print("\nStep 2: Creating DataLoaders...")
    print("-" * 70)

    train_dataset = HSIDataset(X_train, y_train, augment=True)
    val_dataset = HSIDataset(X_val, y_val, augment=False)
    test_dataset = HSIDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == 'cuda'))

    dls = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # =====================================================================
    # Step 3: Initialize Model
    # =====================================================================
    print("\nStep 3: Initializing DMLPFFN model...")
    print("-" * 70)

    in_channels = X_train.shape[3]  # PCA components (should be 128)
    model = DMLPFFN(in_channels=in_channels, num_classes=cfg.num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input channels (PCA components): {in_channels}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # =====================================================================
    # Step 4: Train Model
    # =====================================================================
    print("\nStep 4: Training model...")
    print("-" * 70)

    start_time = time.time()

    model, history = train_model(
        dls=dls,
        in_ch=in_channels,
        num_classes=cfg.num_classes,
        optimizer_type=cfg.optimizer,
        loss_type=cfg.loss_function,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        patience=cfg.patience,
        scheduler_type=cfg.scheduler,
        scheduler_params=cfg.scheduler_params,
        device=device
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # =====================================================================
    # Step 5: Evaluate on Held-Out Test Set
    # =====================================================================
    print("\nStep 5: Evaluating on held-out test set...")
    print("-" * 70)

    report, cm = evaluate_model(model, test_loader, device=device)

    print("\nTest Set Classification Report:")
    print(report)

    print("\nTest Set Confusion Matrix:")
    print(cm)

    # =====================================================================
    # Step 6: Save Results
    # =====================================================================
    print("\nStep 6: Saving results and artifacts...")
    print("-" * 70)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_channels': in_channels,
        'num_classes': cfg.num_classes,
        'config': cfg.__dict__,
        'history': history,
        'test_report': report,
        'test_cm': cm.tolist() if hasattr(cm, 'tolist') else cm,
    }, model_path)
    print(f"Model saved to {model_path}")

    # Save experiment results
    results = {
        'experiment': 'DMLPFFN_corrected',
        'seed': cfg.seed,
        'device': str(device),
        'training_time_seconds': training_time,
        'dataset_shapes': {
            'train': list(X_train.shape),
            'val': list(X_val.shape),
            'test': list(X_test.shape),
        },
        'class_distributions': {
            'train': dict(zip(*np.unique(y_train, return_counts=True))),
            'val': dict(zip(*np.unique(y_val, return_counts=True))),
            'test': dict(zip(*np.unique(y_test, return_counts=True))),
        },
        'model_params': {
            'total': total_params,
            'trainable': trainable_params,
        },
        'best_val_acc': max(history['val_acc']) if history['val_acc'] else 0,
        'best_val_epoch': np.argmax(history['val_acc']) + 1 if history['val_acc'] else 0,
        'final_test_report': report,
        'final_test_cm': cm.tolist() if hasattr(cm, 'tolist') else cm,
        'history': {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc'],
        },
        'corrections_applied': [
            'group_wise_split_by_biological_unit',
            'sensor_specific_pca_fitted_on_train_only',
            'normalization_scaler_fitted_on_train_only',
            'patches_extracted_after_split'
        ]
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # =====================================================================
    # Step 7: Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE - CORRECTED PIPELINE")
    print("=" * 70)
    print(f"Best validation accuracy: {max(history['val_acc']):.4f} (epoch {np.argmax(history['val_acc'])+1})")
    print(f"Test accuracy: {report.split()[1] if isinstance(report, str) else 'N/A'}")
    print(f"Artifacts saved to: {artifacts_dir}")
    print("=" * 70)

    return model, history, report, cm


if __name__ == '__main__':
    # Check if dataset exists
    if not os.path.exists(cfg.data_root):
        print(f"ERROR: Dataset not found at {cfg.data_root}")
        print("Please download the USDA tomato leaf dataset (DOI: 10.15482/USDA.ADC/26046328)")
        print("and extract it to the tomato_leaf/ directory.")
        exit(1)

    model, history, report, cm = run_corrected_experiment()