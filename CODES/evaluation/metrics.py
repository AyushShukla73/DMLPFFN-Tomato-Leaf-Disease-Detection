"""
Evaluation Metrics for Phase 5 Baseline Experiments
Conditional Binary Evaluation Protocol: Healthy vs Late from 4-class model head
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import json


# Class indices in the 4-class head
CLASS_HEALTHY = 0
CLASS_ASYMPTOMATIC = 1
CLASS_EARLY = 2
CLASS_LATE = 3

# Binary evaluation classes
BINARY_CLASS_HEALTHY = 0
BINARY_CLASS_LATE = 1

BINARY_CLASS_NAMES = ['Healthy', 'Late']
FOUR_CLASS_NAMES = ['Healthy', 'Asymptomatic', 'Early', 'Late']


def conditional_binary_probs(logits: np.ndarray) -> np.ndarray:
    """
    Compute conditional binary probabilities from 4-class logits.

    P(Healthy | Healthy or Late, x) = exp(z0) / (exp(z0) + exp(z3))
    P(Late | Healthy or Late, x) = exp(z3) / (exp(z0) + exp(z3))

    Args:
        logits: [N, 4] raw logits from model (Healthy, Asymptomatic, Early, Late)

    Returns:
        probs_binary: [N, 2] conditional probabilities (Healthy, Late)
    """
    z0 = logits[:, CLASS_HEALTHY]
    z3 = logits[:, CLASS_LATE]

    exp_z0 = np.exp(z0 - np.maximum(z0, z3))  # numerical stability
    exp_z3 = np.exp(z3 - np.maximum(z0, z3))

    denom = exp_z0 + exp_z3
    p_healthy = exp_z0 / denom
    p_late = exp_z3 / denom

    return np.column_stack([p_healthy, p_late])


def conditional_binary_predictions(logits: np.ndarray) -> np.ndarray:
    """
    Predict binary class using conditional probabilities.

    Args:
        logits: [N, 4] raw logits from model

    Returns:
        preds_binary: [N] predictions (0=Healthy, 1=Late)
    """
    probs = conditional_binary_probs(logits)
    return np.argmax(probs, axis=1)


def conditional_binary_predictions_with_confidence(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict binary class with confidence scores.

    Args:
        logits: [N, 4] raw logits from model

    Returns:
        preds_binary: [N] predictions (0=Healthy, 1=Late)
        confidences: [N] max conditional probability
    """
    probs = conditional_binary_probs(logits)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    return preds, confidences


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: List[str] = BINARY_CLASS_NAMES
) -> Dict:
    """
    Compute comprehensive binary classification metrics.

    Args:
        y_true: [N] ground truth (0=Healthy, 1=Late)
        y_pred: [N] predictions (0=Healthy, 1=Late)
        y_probs: [N, 2] optional conditional probabilities for AUC
        class_names: List of class names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Overall accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Per-class metrics
    for i, name in enumerate(class_names):
        metrics[f'{name.lower()}_precision'] = precision_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]
        metrics[f'{name.lower()}_recall'] = recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]
        metrics[f'{name.lower()}_f1'] = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

    # Macro averages
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted averages (support-weighted)
    metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics['confusion_matrix'] = cm.tolist()
    metrics['confusion_matrix_labels'] = class_names

    # AUC-ROC (if probabilities provided)
    if y_probs is not None and y_probs.shape[1] == 2:
        try:
            metrics['auc_roc_healthy'] = roc_auc_score(y_true == 0, y_probs[:, 0])
            metrics['auc_roc_late'] = roc_auc_score(y_true == 1, y_probs[:, 1])
            metrics['auc_roc_macro'] = (metrics['auc_roc_healthy'] + metrics['auc_roc_late']) / 2
        except ValueError:
            metrics['auc_roc_healthy'] = None
            metrics['auc_roc_late'] = None
            metrics['auc_roc_macro'] = None

    return metrics


def compute_group_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    class_names: List[str] = BINARY_CLASS_NAMES
) -> Dict:
    """
    Compute metrics at group (image) level by majority voting within each group.

    Args:
        y_true: [N] ground truth
        y_pred: [N] predictions
        groups: [N] group identifiers (image indices)
        class_names: List of class names

    Returns:
        Dictionary of group-level metrics
    """
    unique_groups = np.unique(groups)
    group_true = []
    group_pred = []

    for g in unique_groups:
        mask = groups == g
        # Majority vote for true label (should be unanimous in test set)
        true_label = np.bincount(y_true[mask]).argmax()
        pred_label = np.bincount(y_pred[mask]).argmax()
        group_true.append(true_label)
        group_pred.append(pred_label)

    group_true = np.array(group_true)
    group_pred = np.array(group_pred)

    metrics = {}
    metrics['group_accuracy'] = accuracy_score(group_true, group_pred)
    metrics['num_groups'] = len(unique_groups)
    metrics['group_confusion_matrix'] = confusion_matrix(group_true, group_pred, labels=[0, 1]).tolist()
    metrics['group_confusion_matrix_labels'] = class_names

    # Per-class group metrics
    for i, name in enumerate(class_names):
        metrics[f'group_{name.lower()}_precision'] = precision_score(group_true, group_pred, labels=[i], average=None, zero_division=0)[0]
        metrics[f'group_{name.lower()}_recall'] = recall_score(group_true, group_pred, labels=[i], average=None, zero_division=0)[0]
        metrics[f'group_{name.lower()}_f1'] = f1_score(group_true, group_pred, labels=[i], average=None, zero_division=0)[0]

    metrics['group_macro_precision'] = precision_score(group_true, group_pred, average='macro', zero_division=0)
    metrics['group_macro_recall'] = recall_score(group_true, group_pred, average='macro', zero_division=0)
    metrics['group_macro_f1'] = f1_score(group_true, group_pred, average='macro', zero_division=0)

    return metrics


def evaluate_model_conditional_binary(
    logits: np.ndarray,
    y_true: np.ndarray,
    groups: Optional[np.ndarray] = None,
    class_names: List[str] = BINARY_CLASS_NAMES
) -> Dict:
    """
    Full conditional binary evaluation from 4-class logits.

    This is the MAIN evaluation function for Phase 5.
    Uses conditional probabilities: P(Healthy|Healthy/Late) = exp(z0)/(exp(z0)+exp(z3))

    Args:
        logits: [N, 4] raw logits from 4-class model head
        y_true: [N] ground truth (0=Healthy, 1=Late) - binary labels
        groups: [N] optional group identifiers for group-level metrics
        class_names: List of class names

    Returns:
        Dictionary with all metrics
    """
    # Convert 4-class logits to conditional binary predictions
    y_pred = conditional_binary_predictions(logits)
    y_probs = conditional_binary_probs(logits)

    # Verify y_true only contains binary classes
    unique_true = np.unique(y_true)
    assert set(unique_true).issubset({0, 1}), f"y_true must only contain 0 (Healthy) and 1 (Late), got {unique_true}"

    results = {}
    results['patch_level'] = compute_binary_metrics(y_true, y_pred, y_probs, class_names)

    if groups is not None:
        results['group_level'] = compute_group_level_metrics(y_true, y_pred, groups, class_names)

    # Add metadata
    results['evaluation_protocol'] = 'conditional_binary'
    results['num_samples'] = len(y_true)
    results['class_names'] = class_names
    results['note'] = 'Conditional binary evaluation: P(Healthy|Healthy/Late) = exp(z0)/(exp(z0)+exp(z3))'

    return results


def evaluate_model_from_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    groups: Optional[np.ndarray] = None,
    evaluation_mode: str = 'conditional_binary'
) -> Dict:
    """
    Evaluate model from 4-class logits with specified protocol.

    Args:
        logits: [N, 4] raw logits
        y_true: [N] ground truth
        groups: [N] optional group identifiers
        evaluation_mode: 'conditional_binary' (only supported mode)

    Returns:
        Evaluation results dictionary
    """
    if evaluation_mode != 'conditional_binary':
        raise ValueError(f"Only 'conditional_binary' evaluation is supported. Got: {evaluation_mode}")

    return evaluate_model_conditional_binary(logits, y_true, groups)


# Backwards compatibility for loading Phase 3 DMLPFFN results
def load_phase3_dmlpffn_results(results_path: str) -> Dict:
    """Load Phase 3 DMLPFFN test logits and re-evaluate with conditional binary protocol."""
    data = np.load(results_path, allow_pickle=True)
    logits = data['logits'] if 'logits' in data else data['test_logits']
    y_true = data['targets'] if 'targets' in data else data['test_targets']
    groups = data['groups'] if 'groups' in data else None

    # Map 4-class labels to binary (Healthy=0, Late=1)
    # Original: 0=Healthy, 1=Asymptomatic, 2=Early, 3=Late
    y_binary = np.where(y_true == CLASS_HEALTHY, BINARY_CLASS_HEALTHY,
               np.where(y_true == CLASS_LATE, BINARY_CLASS_LATE, -1))

    # Filter to only Healthy/Late samples (should be all in test set)
    valid_mask = y_binary != -1
    logits = logits[valid_mask]
    y_binary = y_binary[valid_mask]
    if groups is not None:
        groups = groups[valid_mask]

    return evaluate_model_conditional_binary(logits, y_binary, groups)


if __name__ == "__main__":
    # Quick test
    print("Testing conditional binary evaluation...")

    # Simulate 4-class logits
    np.random.seed(42)
    logits = np.random.randn(100, 4)
    # Make Healthy and Late more separable
    logits[:50, 0] += 2  # Healthy
    logits[50:, 3] += 2  # Late

    y_true = np.array([0]*50 + [1]*50)
    groups = np.array([i//10 for i in range(100)])  # 10 groups

    results = evaluate_model_conditional_binary(logits, y_true, groups)

    print(f"Patch Accuracy: {results['patch_level']['accuracy']:.4f}")
    print(f"Macro F1: {results['patch_level']['macro_f1']:.4f}")
    print(f"Weighted F1: {results['patch_level']['weighted_f1']:.4f}")
    if 'group_level' in results:
        print(f"Group Accuracy: {results['group_level']['group_accuracy']:.4f}")
        print(f"Num Groups: {results['group_level']['num_groups']}")

    print("\nConfusion Matrix (patch):")
    print(np.array(results['patch_level']['confusion_matrix']))

    print("\nAll tests passed!")