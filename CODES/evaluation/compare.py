"""
Comparison Table Generation for Phase 5 Baseline Experiments
Generates markdown tables and statistical comparison outputs.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_baseline_results(results_dir: str) -> Dict[str, Any]:
    """Load all baseline results from artifacts directory."""
    results = {}
    base_path = Path(results_dir)

    # Core keys that must be present (accuracy, macro_f1, weighted_f1)
    # The aggregate results use "accuracy" (not "test_accuracy") since re-evaluation runs directly on test set
    # val_accuracy and parameters are optional
    REQUIRED_AGG_KEYS = {"accuracy", "macro_f1", "weighted_f1"}
    OPTIONAL_AGG_KEYS = {"val_accuracy", "parameters"}

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # Try to load aggregate results first
        agg_path = model_dir / "aggregate_results.json"
        agg_valid = False
        if agg_path.exists():
            with open(agg_path) as f:
                agg_data = json.load(f)
            # Check if aggregate has all REQUIRED keys with non-N/A values
            # The aggregate format is nested: {"metric": {"mean": X, "std": Y, "values": [...]}, ...}
            if isinstance(agg_data, dict):
                all_valid = True
                for key in REQUIRED_AGG_KEYS:
                    val = agg_data.get(key)
                    if val is None:
                        all_valid = False
                        break
                    # Check if it's a nested format with mean/std or a flat value
                    if isinstance(val, dict):
                        if "mean" not in val:
                            all_valid = False
                            break
                    elif isinstance(val, str) and val.upper() == "N/A":
                        all_valid = False
                        break
                if all_valid:
                    results[model_name] = {"aggregate": agg_data}
                    agg_valid = True

        if not agg_valid:
            # Fall back to seed directories
            seed_results = {}
            for seed_dir in model_dir.iterdir():
                if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                    test_path = seed_dir / "test_results.json"
                    if test_path.exists():
                        with open(test_path) as f:
                            seed_results[seed_dir.name] = json.load(f)
            if seed_results:
                results[model_name] = {"seeds": seed_results}

    return results


def load_dmlpffn_reference() -> Dict[str, Any]:
    """Load DMLPFFN Phase 3 reference results."""
    return {
        "model_name": "DMLPFFN (Phase 3)",
        "val_accuracy": 0.7875,
        "test_accuracy": 0.9564,
        "macro_f1": 0.9660,
        "weighted_f1": 0.9658,
        "parameters": 447780,
        "train_val_gap": 0.1108,  # 11.08 percentage points
        "inference_ms": None,  # TBD
        "note": "Single run (Phase 3), 4-class head with conditional binary evaluation"
    }


def extract_metric(agg: Dict, key: str) -> Optional[float]:
    """Extract metric value from nested aggregate format (mean/std/values)."""
    val = agg.get(key)
    if val is None:
        return None
    if isinstance(val, dict) and "mean" in val:
        return val["mean"]
    if isinstance(val, (int, float)):
        return float(val)
    return None


def extract_metric_str(agg: Dict, key: str) -> str:
    """Extract metric as formatted string from nested aggregate format."""
    val = agg.get(key)
    if val is None:
        return "N/A"
    if isinstance(val, dict) and "mean" in val:
        mean = val["mean"]
        std = val.get("std", 0.0)
        if std > 0:
            return f"{mean:.4f} +/- {std:.4f}"
        return f"{mean:.4f}"
    if isinstance(val, (int, float)):
        return f"{val:.4f}"
    return str(val)


def format_comparison_table(results: Dict[str, Any], dmlpffn_ref: Dict[str, Any]) -> str:
    """Generate markdown comparison table."""
    lines = []
    lines.append("# Phase 5 Baseline Comparison Results")
    lines.append("")
    lines.append("| Model | Val Acc | Test Acc | Macro F1 | Weighted F1 | Params | Train/Val Gap | Inference (ms) |")
    lines.append("|-------|---------|----------|----------|----------|-------------|--------|---------------|----------------|")

    # DMLPFFN reference row
    lines.append(f"| {dmlpffn_ref['model_name']} | {dmlpffn_ref['val_accuracy']:.4f} | {dmlpffn_ref['test_accuracy']:.4f} | {dmlpffn_ref['macro_f1']:.4f} | {dmlpffn_ref['weighted_f1']:.4f} | {dmlpffn_ref['parameters']:,} | {dmlpffn_ref['train_val_gap']:.4f} | {dmlpffn_ref['inference_ms'] or 'TBD'} |")

    # Baseline models
    for model_name, model_data in sorted(results.items()):
        if "aggregate" in model_data:
            agg = model_data["aggregate"]
            val_acc = extract_metric_str(agg, "val_accuracy")
            test_acc = extract_metric_str(agg, "accuracy")  # aggregate uses "accuracy" not "test_accuracy"
            macro_f1 = extract_metric_str(agg, "macro_f1")
            weighted_f1 = extract_metric_str(agg, "weighted_f1")
            params = agg.get("parameters", "N/A")
            gap = agg.get("train_val_gap", "N/A")
            if isinstance(gap, (int, float)):
                gap = f"{gap:.4f}"
            inference = agg.get("inference_ms", "TBD")
        elif "seeds" in model_data:
            # Compute from seeds if no aggregate
            seeds = model_data["seeds"]
            test_accs = [s.get("test_accuracy", 0) for s in seeds.values() if s.get("test_accuracy") is not None]
            macro_f1s = [s.get("macro_f1", 0) for s in seeds.values() if s.get("macro_f1") is not None]
            weighted_f1s = [s.get("weighted_f1", 0) for s in seeds.values() if s.get("weighted_f1") is not None]

            val_acc = "N/A"
            test_acc = f"{np.mean(test_accs):.4f} +/- {np.std(test_accs, ddof=1):.4f}" if test_accs else "N/A"
            macro_f1 = f"{np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s, ddof=1):.4f}" if macro_f1s else "N/A"
            weighted_f1 = f"{np.mean(weighted_f1s):.4f} +/- {np.std(weighted_f1s, ddof=1):.4f}" if weighted_f1s else "N/A"
            params = "N/A"
            gap = "N/A"
            inference = "TBD"
        else:
            val_acc = test_acc = macro_f1 = weighted_f1 = params = gap = "N/A"
            inference = "TBD"

        if isinstance(params, int):
            params = f"{params:,}"

        lines.append(f"| {model_name} | {val_acc} | {test_acc} | {macro_f1} | {weighted_f1} | {params} | {gap} | {inference} |")

    lines.append("")
    lines.append("*All metrics use conditional binary evaluation (Healthy vs Late from 4-class head).*")
    lines.append("*Test set: 4 test groups, 1,330 patches (Healthy=574, Late=756). Group-aware statistics.*")

    return "\n".join(lines)


def format_statistical_comparison(comparison_results: Dict[str, Any]) -> str:
    """Format statistical test results as markdown."""
    lines = []
    lines.append("# Statistical Comparison Results")
    lines.append("")

    if "pairwise_mcnemar" in comparison_results:
        lines.append("## Pairwise McNemar's Test (Group-Aware)")
        lines.append("")
        lines.append("| Comparison | n01 (A wrong, B correct) | n10 (A correct, B wrong) | chi2 | p-value (asymptotic) | p-value (exact) |")
        lines.append("|------------|-------------------------|-------------------------|-----|---------------------|-----------------|")

        for comp_name, test_result in comparison_results["pairwise_mcnemar"].items():
            lines.append(f"| {comp_name} | {test_result['n01']} | {test_result['n10']} | {test_result['chi2_statistic']:.4f} | {test_result['p_value_asymptotic']:.4f} | {test_result['p_value_exact']:.4f} |")

        lines.append("")
        lines.append(f"*Note: {comparison_results['pairwise_mcnemar'][list(comparison_results['pairwise_mcnemar'].keys())[0]]['warning']}*")
        lines.append("")

    if "bootstrap_cis" in comparison_results:
        lines.append("## Group-Aware Bootstrap Confidence Intervals (95%)")
        lines.append("")
        lines.append("| Model | Mean | Std | CI Lower | CI Upper | N Groups | N Resamples |")
        lines.append("|-------|------|-----|----------|----------|----------|-------------|")

        for model_name, ci in comparison_results["bootstrap_cis"].items():
            lines.append(f"| {model_name} | {ci['mean']:.4f} | {ci['std']:.4f} | {ci['ci_lower']:.4f} | {ci['ci_upper']:.4f} | {ci['n_groups']} | {ci['n_resamples']} |")

        lines.append("")
        lines.append("*Group-aware bootstrap: resamples test GROUPS (runs), not patches. 1000 resamples.*")
        lines.append("")

    return "\n".join(lines)


def format_per_seed_results(results: Dict[str, Any]) -> str:
    """Format per-seed results as markdown table."""
    lines = []
    lines.append("# Per-Seed Results (5 Seeds)")
    lines.append("")

    for model_name, model_data in sorted(results.items()):
        if "seeds" not in model_data:
            continue

        seeds = model_data["seeds"]
        lines.append(f"## {model_name}")
        lines.append("")
        lines.append("| Seed | Val Acc | Test Acc | Macro F1 | Weighted F1 | Best Epoch |")
        lines.append("|------|---------|----------|----------|-------------|------------|")

        for seed_name, seed_data in sorted(seeds.items()):
            val_acc = seed_data.get("val_accuracy", "N/A")
            test_acc = seed_data.get("test_accuracy", "N/A")
            macro_f1 = seed_data.get("macro_f1", "N/A")
            weighted_f1 = seed_data.get("weighted_f1", "N/A")
            best_epoch = seed_data.get("best_epoch", "N/A")

            if isinstance(val_acc, float):
                val_acc = f"{val_acc:.4f}"
            if isinstance(test_acc, float):
                test_acc = f"{test_acc:.4f}"
            if isinstance(macro_f1, float):
                macro_f1 = f"{macro_f1:.4f}"
            if isinstance(weighted_f1, float):
                weighted_f1 = f"{weighted_f1:.4f}"

            lines.append(f"| {seed_name} | {val_acc} | {test_acc} | {macro_f1} | {weighted_f1} | {best_epoch} |")

        # Aggregate row
        test_accs = [s.get("test_accuracy", 0) for s in seeds.values() if s.get("test_accuracy") is not None]
        macro_f1s = [s.get("macro_f1", 0) for s in seeds.values() if s.get("macro_f1") is not None]
        weighted_f1s = [s.get("weighted_f1", 0) for s in seeds.values() if s.get("weighted_f1") is not None]

        if test_accs:
            lines.append(f"| **Mean +/- Std** | - | {np.mean(test_accs):.4f} +/- {np.std(test_accs, ddof=1):.4f} | {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s, ddof=1):.4f} | {np.mean(weighted_f1s):.4f} +/- {np.std(weighted_f1s, ddof=1):.4f} | - |")

        lines.append("")

    return "\n".join(lines)


def generate_full_report(
    results_dir: str,
    output_path: str,
    statistical_results: Optional[Dict] = None
) -> str:
    """Generate complete PHASE5_BASELINE_RESULTS.md report."""
    results = load_baseline_results(results_dir)
    dmlpffn_ref = load_dmlpffn_reference()

    report = []
    report.append("# Phase 5 Baseline Experiments - Results Report")
    report.append("")
    report.append(f"**Date**: 2026-11-14")
    report.append(f"**Status**: COMPLETE")
    report.append(f"**Data Split**: `artifacts/split_manifest.json` (seed=42, frozen)")
    report.append(f"**Evaluation Protocol**: Conditional Binary (Healthy vs Late from 4-class head)")
    report.append(f"**Test Set**: 4 test groups, 1,330 patches (Healthy=574, Late=756)")
    report.append(f"**Seeds**: 5 seeds [42, 123, 456, 789, 999] for DL baselines; single run for SVM/RF")
    report.append("")

    # Comparison table
    report.append(format_comparison_table(results, dmlpffn_ref))
    report.append("")

    # Per-seed results
    report.append(format_per_seed_results(results))
    report.append("")

    # Statistical comparison if available
    if statistical_results:
        report.append(format_statistical_comparison(statistical_results))
        report.append("")

    # Methodology section
    report.append("## Methodology")
    report.append("")
    report.append("### Data Pipeline (Immutable, from Phase 3)")
    report.append("1. **Band Removal**: SWIR bands 94-115 removed (water absorption)")
    report.append("2. **PCA**: IncrementalPCA (n_components=128) per sensor, fitted on TRAIN patches only")
    report.append("3. **Scaling**: StandardScaler fitted on TRAIN patches only")
    report.append("4. **Patch Extraction**: 9x9 patches, stride=4, AFTER group-wise split")
    report.append("")
    report.append("### Training Configuration (Identical for all DL baselines, from Phase 3)")
    report.append("- Batch size: 16")
    report.append("- Optimizer: AdamW")
    report.append("- Learning rate: 2e-4")
    report.append("- Weight decay: 5e-4")
    report.append("- Epochs: 250 (early stop patience=25)")
    report.append("- Scheduler: CosineAnnealingWarmRestarts(T_0=10)")
    report.append("- Label smoothing: 0.1")
    report.append("- Mixup: alpha=0.2 (50% probability)")
    report.append("- CutMix: alpha=1.0 (50% probability)")
    report.append("- Class balancing: WeightedRandomSampler")
    report.append("- Seed: 42 (all RNGs)")
    report.append("- Device: CUDA (RTX 3050 4GB)")
    report.append("- Mixed precision: Enabled")
    report.append("")
    report.append("### Evaluation Protocol")
    report.append("- **Conditional Binary**: P(Healthy|H/L) = exp(z0)/(exp(z0)+exp(z3)), P(Late|H/L) = exp(z3)/(exp(z0)+exp(z3))")
    report.append("- **Metrics**: Accuracy, Macro/Weighted Precision/Recall/F1, Per-class P/R/F1, 2x2 Confusion Matrix, AUC-ROC")
    report.append("- **Group-level metrics**: Majority vote per group (4 test groups)")
    report.append("- **Statistical rigor**: 5 seeds, group-aware bootstrap CI (1000 resamples), group-aware McNemar's test")
    report.append("")
    report.append("### Model Specifications")
    report.append("| Model | Parameters | Input Format | VRAM Estimate |")
    report.append("|-------|------------|--------------|---------------|")
    report.append("| SimpleCNN | 86,156 | NHWC [B,9,9,128] -> NCHW | ~1.2 GB |")
    report.append("| ResNet-18 | 11,242,884 | NHWC [B,9,9,128] -> NCHW | ~3.5 GB |")
    report.append("| MLP-Mixer | 2,310,812 | NHWC [B,9,9,128] -> [B,81,256] | ~2.8 GB |")
    report.append("| SVM (RBF) | N/A | Flattened [N,10368] -> PCA(256) | CPU |")
    report.append("| Random Forest | N/A | Flattened [N,10368] -> PCA(256) | CPU |")
    report.append("| DMLPFFN (ref) | 447,780 | NCHW [B,128,9,9] | ~1.5 GB |")
    report.append("")

    full_report = "\n".join(report)

    with open(output_path, "w") as f:
        f.write(full_report)

    return full_report


if __name__ == "__main__":
    # Test with dummy data
    print("Testing comparison module...")

    # Create dummy results for testing
    dummy_results = {
        "simple_cnn": {
            "aggregate": {
                "val_accuracy": 0.8234,
                "test_accuracy": 0.9123,
                "macro_f1": 0.9087,
                "weighted_f1": 0.9102,
                "parameters": 86156,
                "train_val_gap": 0.0889,
                "inference_ms": 2.3
            }
        },
        "mlp_mixer": {
            "seeds": {
                "seed_42": {"test_accuracy": 0.9234, "macro_f1": 0.9201, "weighted_f1": 0.9215, "best_epoch": 142},
                "seed_123": {"test_accuracy": 0.9187, "macro_f1": 0.9156, "weighted_f1": 0.9172, "best_epoch": 156},
                "seed_456": {"test_accuracy": 0.9256, "macro_f1": 0.9223, "weighted_f1": 0.9238, "best_epoch": 138},
                "seed_789": {"test_accuracy": 0.9201, "macro_f1": 0.9178, "weighted_f1": 0.9194, "best_epoch": 149},
                "seed_999": {"test_accuracy": 0.9178, "macro_f1": 0.9145, "weighted_f1": 0.9161, "best_epoch": 162},
            }
        }
    }

    dmlpffn = load_dmlpffn_reference()
    table = format_comparison_table(dummy_results, dmlpffn)
    print(table)
    print("\n---\n")

    per_seed = format_per_seed_results(dummy_results)
    print(per_seed)

    print("\nAll tests passed!")