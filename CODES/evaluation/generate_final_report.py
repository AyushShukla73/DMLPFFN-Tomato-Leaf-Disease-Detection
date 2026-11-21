#!/usr/bin/env python
"""
Generate the final PHASE5_BASELINE_RESULTS.md report.
Combines baseline comparison table with statistical analysis results.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.compare import generate_full_report


def main():
    results_dir = "artifacts/baselines"
    statistical_path = Path(results_dir) / "statistical_analysis.json"
    output_path = "PHASE5_BASELINE_RESULTS.md"

    # Load statistical results
    statistical_results = None
    if statistical_path.exists():
        with open(statistical_path) as f:
            statistical_results = json.load(f)
        print(f"Loaded statistical results from {statistical_path}")
    else:
        print(f"Warning: {statistical_path} not found, generating report without statistical tests")

    # Generate full report
    report = generate_full_report(results_dir, output_path, statistical_results)
    print(f"\nReport generated: {output_path}")

    # Print the report
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)


if __name__ == "__main__":
    main()