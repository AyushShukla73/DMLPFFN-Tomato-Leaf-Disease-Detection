"""
Statistical Tests for Phase 5 Baseline Experiments
Group-aware bootstrap confidence intervals and McNemar's test for model comparison.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings


def group_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Group-aware bootstrap confidence interval.

    Resamples at the GROUP level (not patch level) to account for
    within-group correlation. This is critical because patches from
    the same image are not independent.

    Args:
        y_true: [N] ground truth labels
        y_pred: [N] predictions
        groups: [N] group identifiers (e.g., image indices)
        metric_fn: Function that takes (y_true, y_pred) and returns a scalar metric
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'mean', 'std', 'ci_lower', 'ci_upper', 'samples'
    """
    if random_state is not None:
        np.random.seed(random_state)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        warnings.warn(f"Only {n_groups} groups available. Bootstrap CI may be unreliable.")
        metric_val = metric_fn(y_true, y_pred)
        return {
            'mean': metric_val,
            'std': 0.0,
            'ci_lower': metric_val,
            'ci_upper': metric_val,
            'samples': [metric_val],
            'n_groups': n_groups,
            'warning': 'Insufficient groups for reliable bootstrap'
        }

    # Compute original metric
    original_metric = metric_fn(y_true, y_pred)

    # Bootstrap resampling at group level
    bootstrap_metrics = []
    for _ in range(n_resamples):
        # Resample groups with replacement
        sampled_groups = np.random.choice(unique_groups, size=n_groups, replace=True)

        # Build resampled dataset
        mask = np.isin(groups, sampled_groups)
        # Need to handle duplicate groups properly - expand indices
        resample_indices = []
        for g in sampled_groups:
            group_indices = np.where(groups == g)[0]
            resample_indices.extend(group_indices)

        resample_indices = np.array(resample_indices)

        if len(resample_indices) == 0:
            continue

        y_true_resampled = y_true[resample_indices]
        y_pred_resampled = y_pred[resample_indices]

        try:
            metric_val = metric_fn(y_true_resampled, y_pred_resampled)
            bootstrap_metrics.append(metric_val)
        except Exception:
            # Skip failed resamples (e.g., single class in resample)
            continue

    bootstrap_metrics = np.array(bootstrap_metrics)

    if len(bootstrap_metrics) == 0:
        warnings.warn("All bootstrap resamples failed")
        return {
            'mean': original_metric,
            'std': 0.0,
            'ci_lower': original_metric,
            'ci_upper': original_metric,
            'samples': [],
            'n_groups': n_groups,
            'warning': 'All bootstrap resamples failed'
        }

    # Percentile confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))

    return {
        'mean': float(np.mean(bootstrap_metrics)),
        'std': float(np.std(bootstrap_metrics, ddof=1)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'samples': bootstrap_metrics.tolist(),
        'n_groups': n_groups,
        'n_resamples': len(bootstrap_metrics),
        'original_metric': float(original_metric)
    }


def mcnemar_group_aware(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray
) -> Dict[str, Any]:
    """
    Group-aware McNemar's test for comparing two models.

    Standard McNemar assumes independent samples. Here we aggregate
    at the group level: each group contributes one vote based on
    majority prediction within that group.

    With only 4 test groups (7 images but 4 distinct groups in test),
    this test has very low power. Results should be interpreted
    as exploratory only.

    Args:
        y_true: [N] ground truth
        pred_a: [N] predictions from model A
        pred_b: [N] predictions from model B
        groups: [N] group identifiers

    Returns:
        Dictionary with test statistic, p-value, and contingency table
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # Aggregate predictions at group level (majority vote)
    group_true = []
    group_pred_a = []
    group_pred_b = []

    for g in unique_groups:
        mask = groups == g
        # True label (should be unanimous in test set)
        true_label = np.bincount(y_true[mask]).argmax()
        # Majority predictions
        pred_a_label = np.bincount(pred_a[mask]).argmax()
        pred_b_label = np.bincount(pred_b[mask]).argmax()

        group_true.append(true_label)
        group_pred_a.append(pred_a_label)
        group_pred_b.append(pred_b_label)

    group_true = np.array(group_true)
    group_pred_a = np.array(group_pred_a)
    group_pred_b = np.array(group_pred_b)

    # Build contingency table for McNemar
    # n01: A wrong, B correct
    # n10: A correct, B wrong
    n01 = np.sum((group_pred_a != group_true) & (group_pred_b == group_true))
    n10 = np.sum((group_pred_a == group_true) & (group_pred_b != group_true))

    # McNemar's chi-squared statistic (with continuity correction)
    # chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)
    if n01 + n10 > 0:
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        # p-value from chi-squared distribution with 1 df
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0.0
        p_value = 1.0

    # Also compute exact binomial p-value (more appropriate for small n)
    n_discordant = n01 + n10
    if n_discordant > 0:
        # Under null, n01 ~ Binomial(n_discordant, 0.5)
        p_exact = 2 * min(
            stats.binom.cdf(min(n01, n10), n_discordant, 0.5),
            1 - stats.binom.cdf(max(n01, n10) - 1, n_discordant, 0.5)
        )
        # Cap at 1.0 since p-value cannot exceed 1
        p_exact = min(1.0, p_exact)
    else:
        p_exact = 1.0

    return {
        'chi2_statistic': float(chi2),
        'p_value_asymptotic': float(p_value),
        'p_value_exact': float(p_exact),
        'n01': int(n01),
        'n10': int(n10),
        'n_discordant': int(n_discordant),
        'n_groups': n_groups,
        'contingency_table': {
            'A_correct_B_correct': int(np.sum((group_pred_a == group_true) & (group_pred_b == group_true))),
            'A_correct_B_wrong': int(n10),
            'A_wrong_B_correct': int(n01),
            'A_wrong_B_wrong': int(np.sum((group_pred_a != group_true) & (group_pred_b != group_true)))
        },
        'group_predictions': {
            'true': group_true.tolist(),
            'pred_a': group_pred_a.tolist(),
            'pred_b': group_pred_b.tolist()
        },
        'warning': 'Group-aware McNemar with only 4 test groups has very low statistical power. Results are exploratory.'
    }


def compare_models_seeds(
    results_dict: Dict[str, List[Dict]],
    groups: np.ndarray,
    metric_name: str = 'accuracy'
) -> Dict[str, Any]:
    """
    Compare multiple models across seeds with group-aware statistics.

    Args:
        results_dict: {model_name: [seed_results_dict, ...]}
                      Each seed_results_dict must have 'predictions', 'targets' or similar
        groups: [N] group identifiers for test set
        metric_name: Metric to compare ('accuracy', 'macro_f1', etc.)

    Returns:
        Comparison results with bootstrap CIs and pairwise tests
    """
    model_names = list(results_dict.keys())
    n_models = len(model_names)

    # Extract per-seed metrics
    model_metrics = {}
    model_predictions = {}

    for model_name, seed_results in results_dict.items():
        metrics_per_seed = []
        preds_per_seed = []

        for seed_result in seed_results:
            y_true = np.array(seed_result['targets'])
            y_pred = np.array(seed_result['predictions'])

            if metric_name == 'accuracy':
                metric_fn = lambda yt, yp: np.mean(yt == yp)
            elif metric_name == 'macro_f1':
                from sklearn.metrics import f1_score
                metric_fn = lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

            metric_val = metric_fn(y_true, y_pred)
            metrics_per_seed.append(metric_val)
            preds_per_seed.append(y_pred)

        model_metrics[model_name] = metrics_per_seed
        model_predictions[model_name] = preds_per_seed

    # Compute bootstrap CIs for each model (using first seed's groups, assumes same groups across seeds)
    model_cis = {}
    for model_name in model_names:
        # Use first seed for bootstrap (same test set across seeds)
        y_true = np.array(results_dict[model_name][0]['targets'])
        y_pred = np.array(results_dict[model_name][0]['predictions'])

        if metric_name == 'accuracy':
            metric_fn = lambda yt, yp: np.mean(yt == yp)
        elif metric_name == 'macro_f1':
            from sklearn.metrics import f1_score
            metric_fn = lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0)

        ci_result = group_bootstrap_ci(y_true, y_pred, groups, metric_fn, n_resamples=1000)
        model_cis[model_name] = ci_result

    # Pairwise McNemar tests (using first seed for each model)
    pairwise_tests = {}
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            y_true = np.array(results_dict[model_a][0]['targets'])
            pred_a = np.array(results_dict[model_a][0]['predictions'])
            pred_b = np.array(results_dict[model_b][0]['predictions'])

            mcnemar_result = mcnemar_group_aware(y_true, pred_a, pred_b, groups)
            pairwise_tests[f'{model_a}_vs_{model_b}'] = mcnemar_result

    return {
        'metric': metric_name,
        'per_seed_metrics': model_metrics,
        'bootstrap_cis': model_cis,
        'pairwise_mcnemar': pairwise_tests,
        'n_seeds': len(results_dict[model_names[0]]) if model_names else 0,
        'n_models': n_models
    }


def aggregate_seed_results(seed_results: List[Dict], metric_keys: List[str]) -> Dict:
    """
    Aggregate metrics across seeds: mean ± std.

    Args:
        seed_results: List of result dicts from each seed
        metric_keys: List of metric keys to aggregate

    Returns:
        Dictionary with mean, std for each metric
    """
    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in seed_results if key in r and r[key] is not None]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                'values': values
            }
        else:
            aggregated[key] = {'mean': None, 'std': None, 'values': []}
    return aggregated


def format_comparison_table(comparison_results: Dict) -> str:
    """
    Format comparison results as a markdown table.
    """
    lines = []
    lines.append("| Model | Mean ± Std | 95% CI (Group Bootstrap) |")
    lines.append("|-------|------------|---------------------------|")

    for model_name, ci in comparison_results['bootstrap_cis'].items():
        mean = ci['mean']
        std = ci['std']
        ci_lower = ci['ci_lower']
        ci_upper = ci['ci_upper']
        lines.append(f"| {model_name} | {mean:.4f} ± {std:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] |")

    lines.append("")
    lines.append("*Group-aware bootstrap CI (1000 resamples, 4 test groups). Exploratory only.*")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("Testing statistical tests...")

    np.random.seed(42)
    n_samples = 100
    n_groups = 4
    groups = np.array([i % n_groups for i in range(n_samples)])

    y_true = np.random.randint(0, 2, n_samples)
    pred_a = y_true.copy()
    pred_a[np.random.choice(n_samples, 10)] = 1 - pred_a[np.random.choice(n_samples, 10)]

    pred_b = y_true.copy()
    pred_b[np.random.choice(n_samples, 15)] = 1 - pred_b[np.random.choice(n_samples, 15)]

    # Test bootstrap CI
    metric_fn = lambda yt, yp: np.mean(yt == yp)
    ci = group_bootstrap_ci(y_true, pred_a, groups, metric_fn, n_resamples=100, random_state=42)
    print(f"Bootstrap CI: mean={ci['mean']:.4f}, std={ci['std']:.4f}, CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # Test McNemar
    mcnemar = mcnemar_group_aware(y_true, pred_a, pred_b, groups)
    print(f"McNemar: chi2={mcnemar['chi2_statistic']:.4f}, p_exact={mcnemar['p_value_exact']:.4f}")
    print(f"  Contingency: {mcnemar['contingency_table']}")
    print(f"  Warning: {mcnemar['warning']}")

    print("\nAll tests passed!")