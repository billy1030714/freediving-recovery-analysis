#!/usr/bin/env python
"""
Regression Baseline Comparison - Compare current model performance against historical baselines to prevent performance degradation
"""
import json
import sys
from pathlib import Path
import argparse

def load_card(filepath: Path) -> dict:
    """Load dataset_card.json file"""
    if not filepath.exists():
        print(f"⚠️ Baseline file not found: {filepath}. This might be the first run.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Compare current model performance against a baseline dataset_card.")
    parser.add_argument("--current-card", type=str, required=True, help="Path to the current dataset_card.json")
    parser.add_argument("--baseline-card", type=str, required=True, help="Path to the baseline dataset_card.json")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Acceptable R² degradation percentage (e.g., 0.05 for 5%)")
    args = parser.parse_args()

    print("=" * 60)
    print("Running Regression Baseline Comparison")
    print("=" * 60)

    current_card_path = Path(args.current_card)
    baseline_card_path = Path(args.baseline_card)

    # Load current results
    current_card = load_card(current_card_path)
    if not current_card:
        print(f"❌ Current dataset card not found at {current_card_path}!")
        sys.exit(1)
        
    current_metrics = current_card.get("evaluation_metrics", {})
    current_r2 = current_metrics.get("r2")

    # Load baseline results
    baseline_card = load_card(baseline_card_path)
    if not baseline_card:
        print("✅ No baseline found. Setting current results as the first baseline.")
        baseline_card_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_card_path, 'w') as f:
            json.dump(current_card, f, indent=2)
        sys.exit(0)

    baseline_metrics = baseline_card.get("evaluation_metrics", {})
    baseline_r2 = baseline_metrics.get("r2")

    # Compare
    if current_r2 is None or baseline_r2 is None:
        print("❌ Could not find 'r2' score in one of the dataset cards.")
        sys.exit(1)

    print(f"Current R²:    {current_r2:.4f} (from {current_card_path.name})")
    print(f"Baseline R²:   {baseline_r2:.4f} (from {baseline_card_path.name})")

    degradation = baseline_r2 - current_r2
    
    # Special handling for Track B: we expect R² to stay low
    task_type = current_card.get("task_type", "unknown")
    if task_type == "long_term":
        if current_r2 > 0.05:
            print("❌ FAIL: Track B (long_term) R² score is too high, indicating potential data leakage.")
            sys.exit(1)
        else:
            print("✅ PASS: Track B (long_term) R² score remains near zero as expected.")
            sys.exit(0)

    # Track A handling
    if degradation > (abs(baseline_r2) * args.tolerance):
        print(f"❌ FAIL: Performance degraded by {degradation:.4f} ({degradation/abs(baseline_r2):.2%}), which is over the {args.tolerance:.0%} tolerance.")
        sys.exit(1)
    else:
        print(f"✅ PASS: Performance is stable or has improved.")
        # If performance improved, you may choose to update the baseline (implemented in CI/CD yml)
        sys.exit(0)

if __name__ == "__main__":
    main()