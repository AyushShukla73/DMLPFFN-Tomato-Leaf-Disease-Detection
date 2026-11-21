"""Run only missing baseline seeds for a specific model."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config_baselines import get_experiment_config, DEVICE
from run_baselines import (
    load_preprocessing_artifacts,
    load_split_manifest,
    create_model,
    train_dl_baseline,
    save_seed_artifacts,
)
import torch

MODEL = sys.argv[1]
SEEDS = [int(s) for s in sys.argv[2].split(",")]
OUTPUT_DIR = Path("artifacts/baselines") / MODEL

print(f"Running {MODEL} seeds {SEEDS}")

artifacts = load_preprocessing_artifacts()
split_manifest = load_split_manifest()
exp_config = get_experiment_config(MODEL)

for seed in SEEDS:
    print(f"\n=== Seed {seed} ===")
    results = train_dl_baseline(
        model_name=MODEL,
        seed=seed,
        artifacts=artifacts,
        split_manifest=split_manifest,
        output_dir=OUTPUT_DIR,
        device='cuda',  # Use GPU for faster training
    )
    model = create_model(MODEL, exp_config["config"])
    best_path = OUTPUT_DIR / f"seed_{seed}" / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=False))
    save_seed_artifacts(MODEL, seed, results, OUTPUT_DIR, model)
    print(f"  Done. Test acc: {results['test_results'].get('test_accuracy', 'N/A')}")

print("\nALL DONE")