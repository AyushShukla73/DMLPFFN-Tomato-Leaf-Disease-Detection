#!/usr/bin/env python
"""
Run DMLPFFN inference on test set (frozen manifest) to get conditional binary logits and confusion matrix.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

# Add CODES to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CODES'))

from config import cfg
from dmlpffn_disease import DMLPFFN
from data_preprocessing_corrected import load_tomato_dataset_corrected, SplitManifest, PreprocessingArtifacts

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the FROZEN split manifest (correct one with 1330 test patches)
    manifest_path = os.path.join(os.path.dirname(__file__), 'artifacts', 'split_manifest_frozen.json')
    manifest = SplitManifest.load(manifest_path)
    print(f"Loaded frozen manifest: {manifest.test_patches} test patches")
    print(f"Test groups: {manifest.test_groups}")
    print(f"Test images: {manifest.test_images}")

    # Load preprocessing artifacts
    artifacts_path = os.path.join(os.path.dirname(__file__), 'artifacts', 'preprocessing_artifacts')
    artifacts = PreprocessingArtifacts.load(artifacts_path)
    print(f"Loaded preprocessing artifacts")

    # Load test data using the corrected pipeline with FROZEN manifest
    data_root = cfg.data_root
    print(f"Loading test data from: {data_root}")

    splits = load_tomato_dataset_corrected(
        root=data_root,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        load_artifacts_path=artifacts_path,
        save_manifest_path=manifest_path
    )

    X_test, y_test = splits['test']
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Test labels distribution: {np.bincount(y_test)}")

    # The test set only has Healthy (0) and Late (3) labels
    y_test_binary = (y_test == 3).astype(int)
    print(f"Conditional binary labels: Healthy={np.sum(y_test_binary==0)}, Late={np.sum(y_test_binary==1)}")

    # Create model
    model = DMLPFFN(
        in_channels=cfg.pca_components,  # 128
        num_classes=4,                   # 4 classes
        dropout=0.7
    ).to(device)

    # Load best model checkpoint
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'artifacts', 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best val acc: {checkpoint.get('best_val_acc', 'unknown')}")

    # Run inference
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float().permute(0, 3, 1, 2),  # NHWC -> NCHW
        torch.from_numpy(y_test).long()
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_logits = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            all_logits.append(logits.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y_batch.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\nAll logits shape: {all_logits.shape}")
    print(f"All labels shape: {all_labels.shape}")
    print(f"Label distribution: {np.bincount(all_labels)}")

    # 4-class confusion matrix
    from sklearn.metrics import confusion_matrix
    cm_4class = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    print(f"\n4-class Confusion Matrix:")
    print(cm_4class)

    # Conditional binary from logits (mathematically exact marginalization)
    # P(Healthy|H/L) = exp(z0) / (exp(z0) + exp(z3))
    # P(Late|H/L) = exp(z3) / (exp(z0) + exp(z3))
    logits_healthy = all_logits[:, 0]
    logits_late = all_logits[:, 3]

    # Compute conditional binary probabilities
    log_sum_exp = np.log(np.exp(logits_healthy) + np.exp(logits_late))
    p_healthy_given_hl = np.exp(logits_healthy - log_sum_exp)
    p_late_given_hl = np.exp(logits_late - log_sum_exp)

    # Predictions: threshold at 0.5
    pred_binary = (p_late_given_hl > 0.5).astype(int)

    # True labels: 0=Healthy, 1=Late
    true_binary = (all_labels == 3).astype(int)

    cm_binary = confusion_matrix(true_binary, pred_binary)
    print(f"\nConditional Binary Confusion Matrix (from logits):")
    print(cm_binary)
    print(f"  TN (Healthy->Healthy): {cm_binary[0,0]}")
    print(f"  FP (Healthy->Late):    {cm_binary[0,1]}")
    print(f"  FN (Late->Healthy):    {cm_binary[1,0]}")
    print(f"  TP (Late->Late):       {cm_binary[1,1]}")

    # Also compute from 4-class argmax for comparison
    pred_4class_binary = (all_preds == 3).astype(int)
    cm_4class_binary = confusion_matrix(true_binary, pred_4class_binary)
    print(f"\nConditional Binary Confusion Matrix (from 4-class argmax):")
    print(cm_4class_binary)

    # Metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    acc = accuracy_score(true_binary, pred_binary)
    prec, rec, f1, _ = precision_recall_fscore_support(true_binary, pred_binary, average=None)
    macro_f1 = f1.mean()
    weighted_f1 = np.average(f1, weights=np.bincount(true_binary))

    # AUC
    auc = roc_auc_score(true_binary, p_late_given_hl)

    print(f"\nConditional Binary Metrics (from logits):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: Healthy={prec[0]:.4f}, Late={prec[1]:.4f}")
    print(f"  Recall:    Healthy={rec[0]:.4f}, Late={rec[1]:.4f}")
    print(f"  F1:        Healthy={f1[0]:.4f}, Late={f1[1]:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

    # Per-group metrics
    # Need to know which group each patch belongs to
    # For now, use the manifest info
    print(f"\nTest groups: {manifest.test_groups}")
    print(f"Test images: {manifest.test_images}")

    # Save results
    results = {
        'logits': all_logits.tolist(),
        'predictions_4class': all_preds.tolist(),
        'true_labels': all_labels.tolist(),
        'confusion_matrix_4class': cm_4class.tolist(),
        'confusion_matrix_conditional_binary_logits': cm_binary.tolist(),
        'confusion_matrix_conditional_binary_argmax': cm_4class_binary.tolist(),
        'conditional_binary_metrics': {
            'accuracy': float(acc),
            'precision_healthy': float(prec[0]),
            'precision_late': float(prec[1]),
            'recall_healthy': float(rec[0]),
            'recall_late': float(rec[1]),
            'f1_healthy': float(f1[0]),
            'f1_late': float(f1[1]),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'auc_roc': float(auc)
        }
    }

    output_path = os.path.join(os.path.dirname(__file__), 'artifacts', 'dmlpffn_test_logits_frozen.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    return results

if __name__ == '__main__':
    run_inference()