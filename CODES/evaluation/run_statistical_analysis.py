#!/usr/bin/env python
"""
Run statistical analysis on re-evaluated model results.
Computes group-aware bootstrap CIs and pairwise McNemar's tests.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add CODES to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.statistical_tests import (
    group_bootstrap_ci,
    mcnemar_group_aware,
    compare_models_seeds,
    aggregate_seed_results,
    format_comparison_table,
)
from evaluation.metrics import evaluate_model_conditional_binary


def load_model_results(model_name: str, seeds: List[int], base_dir: Path) -> Dict[str, List[Dict]]:
    """Load test results and predictions for a model across seeds."""
    results = {"predictions": [], "targets": []}
    seed_results = []

    for seed in seeds:
        seed_dir = base_dir / model_name / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"Warning: {seed_dir} does not exist")
            continue

        logits = np.load(seed_dir / "test_logits.npy")
        targets = np.load(seed_dir / "test_targets.npy")
        groups = np.load(seed_dir / "test_groups.npy")

        # Compute conditional binary predictions from 4-class logits
        # Extract logits for Healthy (0) and Late (3)
        healthy_logits = logits[:, 0]
        late_logits = logits[:, 3]
        binary_logits = np.stack([healthy_logits, late_logits], axis=1)

        # Softmax and argmax
        exp_logits = np.exp(binary_logits - binary_logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        predictions = probs.argmax(axis=1)  # 0=Healthy, 1=Late

        # Map targets: 0->0 (Healthy), 3->1 (Late)
        binary_targets = np.where(targets == 0, 0, 1)

        seed_results.append({
            "predictions": predictions,
            "targets": binary_targets,
            "groups": groups,
            "logits": logits,
        })

    return {model_name: seed_results}


def compute_seed_metrics(seed_result: Dict) -> Dict[str, float]:
    """Compute all metrics for a single seed result."""
    logits = seed_result["logits"]
    targets = seed_result["targets"]
    groups = seed_result["groups"]

    # Map targets: 0->0 (Healthy), 3->1 (Late)
    binary_targets = np.where(targets == 0, 0, 1)

    # Use the proper evaluation function from metrics.py
    results = evaluate_model_conditional_binary(logits, binary_targets, groups)

    # Flatten metrics for easier access
    metrics = {}
    patch = results["patch_level"]
    metrics["accuracy"] = patch["accuracy"]
    metrics["macro_f1"] = patch["macro_f1"]
    metrics["weighted_f1"] = patch["weighted_f1"]
    metrics["macro_precision"] = patch["macro_precision"]
    metrics["macro_recall"] = patch["macro_recall"]
    metrics["weighted_precision"] = patch["weighted_precision"]
    metrics["weighted_recall"] = patch["weighted_recall"]

    if "group_level" in results:
        group = results["group_level"]
        metrics["group_accuracy"] = group["group_accuracy"]
        metrics["group_macro_f1"] = group["group_macro_f1"]
        metrics["group_macro_precision"] = group["group_macro_precision"]
        metrics["group_macro_recall"] = group["group_macro_recall"]

    return metrics


def main():
    # Configuration
    base_dir = Path("artifacts/baselines")
    dl_models = ["simple_cnn", "mlp_mixer_hsi", "resnet18_hsi"]
    seeds = [42, 123, 456, 789, 999]

    print("=" * 60)
    print("STATISTICAL ANALYSIS - PHASE 5 BASELINES")
    print("=" * 60)

    # Load all model results
    all_results = {}
    all_groups = None

    for model_name in dl_models:
        print(f"\nLoading {model_name}...")
        model_results = load_model_results(model_name, seeds, base_dir)
        if model_results[model_name]:
            all_results.update(model_results)
            if all_groups is None:
                all_groups = model_results[model_name][0]["groups"]

    if not all_results:
        print("ERROR: No results found!")
        return

    print(f"\nLoaded {len(all_results)} models with {len(seeds)} seeds each")
    print(f"Test groups: {np.unique(all_groups)} ({len(np.unique(all_groups))} groups)")

    # Compute per-seed metrics
    print("\nComputing per-seed metrics...")
    for model_name, seed_results in all_results.items():
        print(f"\n{model_name}:")
        for i, seed_result in enumerate(seed_results):
            metrics = compute_seed_metrics(seed_result)
            print(f"  Seed {seeds[i]}: Acc={metrics['accuracy']:.4f}, Macro F1={metrics['macro_f1']:.4f}, "
                  f"Weighted F1={metrics['weighted_f1']:.4f}, Group Acc={metrics['group_accuracy']:.4f}")
            seed_result["metrics"] = metrics

    # Run statistical comparison
    print("\n" + "=" * 60)
    print("GROUP-AWARE BOOTSTRAP CONFIDENCE INTERVALS (1000 resamples)")
    print("=" * 60)

    comparison = compare_models_seeds(all_results, all_groups, metric_name="accuracy")

    print(format_comparison_table(comparison))

    # Also for macro_f1
    print("\n" + "=" * 60)
    print("GROUP-AWARE BOOTSTRAP CONFIDENCE INTERVALS - MACRO F1")
    print("=" * 60)

    comparison_f1 = compare_models_seeds(all_results, all_groups, metric_name="macro_f1")
    print(format_comparison_table(comparison_f1))

    # Pairwise McNemar tests
    print("\n" + "=" * 60)
    print("PAIRWISE MCNEMAR'S TEST (GROUP-AWARE)")
    print("=" * 60)

    for comp_name, test_result in comparison["pairwise_mcnemar"].items():
        print(f"\n{comp_name}:")
        print(f"  n01 (A wrong, B correct): {test_result['n01']}")
        print(f"  n10 (A correct, B wrong): {test_result['n10']}")
        print(f"  Chi2: {test_result['chi2_statistic']:.4f}")
        print(f"  p-value (asymptotic): {test_result['p_value_asymptotic']:.4f}")
        print(f"  p-value (exact): {test_result['p_value_exact']:.4f}")
        print(f"  Contingency: {test_result['contingency_table']}")

    print(f"\n{test_result['warning']}")

    # Aggregate results across seeds
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (MEAN ± STD ACROSS 5 SEEDS)")
    print("=" * 60)

    for model_name, seed_results in all_results.items():
        metrics_keys = ["accuracy", "macro_f1", "weighted_f1", "group_accuracy"]
        agg = aggregate_seed_results([sr["metrics"] for sr in seed_results], metrics_keys)

        print(f"\n{model_name}:")
        for key in metrics_keys:
            if key in agg and agg[key]["mean"] is not None:
                print(f"  {key}: {agg[key]['mean']:.4f} ± {agg[key]['std']:.4f}")

    # Save statistical results
    output = {
        "bootstrap_cis_accuracy": comparison["bootstrap_cis"],
        "bootstrap_cis_macro_f1": comparison_f1["bootstrap_cis"],
        "pairwise_mcnemar": comparison["pairwise_mcnemar"],
        "per_seed_metrics": {
            model: [sr["metrics"] for sr in seed_results]
            for model, seed_results in all_results.items()
        },
        "aggregate_metrics": {
            model: aggregate_seed_results(
                [sr["metrics"] for sr in seed_results],
                ["accuracy", "macro_f1", "weighted_f1", "group_accuracy"]
            )
            for model, seed_results in all_results.items()
        },
    }

    output_path = base_dir / "statistical_analysis.json"
    with open(output_path, "w") as f:
        # Convert numpy types to native Python
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(output), f, indent=2)

    print(f"\nStatistical analysis saved to: {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()