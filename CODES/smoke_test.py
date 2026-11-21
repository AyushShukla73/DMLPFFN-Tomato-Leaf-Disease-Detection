#!/usr/bin/env python
"""
Smoke test for Phase 5 baseline pipeline.
Verifies all imports work and models can be instantiated.
Uses ASCII output for Windows console compatibility.
"""
import sys
import traceback
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")

    # Core dependencies
    import torch
    import numpy as np
    import sklearn
    print(f"  [OK] PyTorch {torch.__version__}")
    print(f"  [OK] NumPy {np.__version__}")
    print(f"  [OK] scikit-learn {sklearn.__version__}")
    print(f"  [OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")

    # Baseline models
    from CODES.baselines.simple_cnn import SimpleCNN
    print("  [OK] SimpleCNN")

    from CODES.baselines.resnet18_hsi import ResNet18_HSI
    print("  [OK] ResNet18_HSI")

    from CODES.baselines.mlp_mixer_hsi import MLPMixer_HSI
    print("  [OK] MLPMixer_HSI")

    from CODES.baselines.classical_ml import ClassicalMLBaselines
    print("  [OK] ClassicalMLBaselines (get_svm_pipeline, get_rf_pipeline)")

    # Trainer
    from CODES.baselines.trainer import BaselineTrainer
    print("  [OK] BaselineTrainer")

    # Evaluation
    from CODES.evaluation.metrics import (
        conditional_binary_probs,
        conditional_binary_predictions,
        compute_binary_metrics,
        compute_group_level_metrics,
        evaluate_model_conditional_binary
    )
    print("  [OK] evaluation.metrics")

    from CODES.evaluation.statistical_tests import (
        group_bootstrap_ci,
        mcnemar_group_aware,
        compare_models_seeds
    )
    print("  [OK] evaluation.statistical_tests")

    from CODES.evaluation.compare import (
        load_baseline_results,
        load_dmlpffn_reference,
        format_comparison_table,
        format_statistical_comparison,
        format_per_seed_results,
        generate_full_report
    )
    print("  [OK] evaluation.compare")

    # Config
    from CODES.config_baselines import (
        EXPERIMENTS,
        DMLPFFN_REFERENCE,
        EVALUATION_CONFIG,
        SEEDS,
        TrainingConfig,
        get_all_experiment_names
    )
    print("  [OK] config_baselines")

    # Run orchestrator
    import CODES.run_baselines as run_baselines
    print("  [OK] run_baselines")

    return True


def test_model_instantiation():
    """Test that all models can be instantiated and forward pass works."""
    print("\nTesting model instantiation...")
    import torch
    from CODES.baselines.simple_cnn import SimpleCNN
    from CODES.baselines.resnet18_hsi import ResNet18_HSI
    from CODES.baselines.mlp_mixer_hsi import MLPMixer_HSI

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    # NCHW format: [B, C, H, W] = [2, 128, 9, 9] - PyTorch Conv2d expectation
    dummy_input = torch.randn(batch_size, 128, 9, 9).to(device)

    # SimpleCNN
    model = SimpleCNN(num_classes=4, base_channels=44).to(device)
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (batch_size, 4), f"SimpleCNN output shape: {out.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] SimpleCNN: {n_params:,} params, output {tuple(out.shape)}")

    # ResNet18
    model = ResNet18_HSI(num_classes=4).to(device)
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (batch_size, 4), f"ResNet18 output shape: {out.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] ResNet18_HSI: {n_params:,} params, output {tuple(out.shape)}")

    # MLP-Mixer
    model = MLPMixer_HSI(num_classes=4).to(device)
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (batch_size, 4), f"MLP-Mixer output shape: {out.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] MLPMixer_HSI: {n_params:,} params, output {tuple(out.shape)}")

    return True


def test_evaluation_pipeline():
    """Test conditional binary evaluation with dummy data."""
    print("\nTesting evaluation pipeline...")
    import numpy as np
    from CODES.evaluation.metrics import evaluate_model_conditional_binary

    np.random.seed(42)
    # Simulate 4-class logits (batch=100)
    logits = np.random.randn(100, 4)
    logits[:50, 0] += 2  # Healthy class
    logits[50:, 3] += 2  # Late class

    y_true = np.array([0]*50 + [1]*50)  # Binary: 0=Healthy, 1=Late
    groups = np.array([i//10 for i in range(100)])  # 10 groups

    results = evaluate_model_conditional_binary(logits, y_true, groups)

    print(f"  [OK] Patch Accuracy: {results['patch_level']['accuracy']:.4f}")
    print(f"  [OK] Macro F1: {results['patch_level']['macro_f1']:.4f}")
    print(f"  [OK] Weighted F1: {results['patch_level']['weighted_f1']:.4f}")
    print(f"  [OK] Group Accuracy: {results['group_level']['group_accuracy']:.4f}")
    print(f"  [OK] Num Groups: {results['group_level']['num_groups']}")
    print(f"  [OK] Protocol: {results['evaluation_protocol']}")

    return True


def test_statistical_functions():
    """Test statistical test functions."""
    print("\nTesting statistical functions...")
    import numpy as np
    from CODES.evaluation.statistical_tests import group_bootstrap_ci, mcnemar_group_aware

    np.random.seed(42)
    n_samples = 100
    n_groups = 4
    groups = np.array([i % n_groups for i in range(n_samples)])
    y_true = np.random.randint(0, 2, n_samples)
    pred_a = y_true.copy()
    pred_a[np.random.choice(n_samples, 10)] = 1 - pred_a[np.random.choice(n_samples, 10)]
    pred_b = y_true.copy()
    pred_b[np.random.choice(n_samples, 15)] = 1 - pred_b[np.random.choice(n_samples, 15)]

    metric_fn = lambda yt, yp: np.mean(yt == yp)
    ci = group_bootstrap_ci(y_true, pred_a, groups, metric_fn, n_resamples=100, random_state=42)
    print(f"  [OK] Bootstrap CI: mean={ci['mean']:.4f}, CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    mcnemar = mcnemar_group_aware(y_true, pred_a, pred_b, groups)
    print(f"  [OK] McNemar: chi2={mcnemar['chi2_statistic']:.4f}, p_exact={mcnemar['p_value_exact']:.4f}")

    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    from CODES.config_baselines import EXPERIMENTS, SEEDS, get_all_experiment_names

    exp_names = get_all_experiment_names()
    print(f"  [OK] Experiment order: {exp_names}")
    print(f"  [OK] DL Seeds: {SEEDS}")

    for name in exp_names:
        exp = EXPERIMENTS[name]
        cfg = exp["config"]
        params_str = f"~{cfg.expected_params:,} params" if hasattr(cfg, 'expected_params') else "params N/A (classical ML)"
        print(f"  [OK] {name}: {exp['model_class']} from {exp['module']}, {params_str}")

    return True


def test_artifacts():
    """Verify artifacts exist."""
    print("\nTesting artifacts...")
    import os
    from pathlib import Path

    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    required = [
        "split_manifest_frozen.json",
        "preprocessing_artifacts_sklearn.joblib",
        "preprocessing_artifacts_arrays.npz",
    ]

    for f in required:
        path = artifacts_dir / f
        if path.exists():
            size_mb = path.stat().st_size / (1024*1024)
            print(f"  [OK] {f} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {f}")
            return False

    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("PHASE 5 BASELINE PIPELINE SMOKE TEST")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Model Instantiation", test_model_instantiation),
        ("Evaluation Pipeline", test_evaluation_pipeline),
        ("Statistical Functions", test_statistical_functions),
        ("Configuration", test_config),
        ("Artifacts", test_artifacts),
    ]

    all_passed = True
    for name, test_fn in tests:
        try:
            result = test_fn()
            if not result:
                print(f"  [FAIL] {name}")
                all_passed = False
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SMOKE TESTS PASSED!")
        print("Pipeline ready for baseline experiments.")
    else:
        print("SOME TESTS FAILED - Fix before running experiments.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())