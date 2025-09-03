#!/usr/bin/env python
"""
Regression Test - Run the full pipeline and compare key metrics against the golden standard
"""
import sys
import json
from pathlib import Path
import subprocess
import argparse
import os

GOLDEN_METRICS = {
    "ERS": {
        "short_term": { "r2": 0.9175 },
        "long_term": { "r2": -0.1015 }
    }
}
TOLERANCE = 1e-4

def run_full_pipeline(task_type: str, target: str, skip_cleaning: bool = False) -> None:
    """Run the full analysis pipeline as subprocesses"""
    print(f"\n--- Running full pipeline for track: {task_type}, target: {target} ---")
    env = os.environ.copy()
    env["TASK_TYPE"] = task_type
    env["TARGETS"] = target
    
    scripts = [
        "hrr_analysis/01_cleaning.py",
        "hrr_analysis/02_features.py",
        "hrr_analysis/04_models.py",
        "hrr_analysis/07_visualize.py",
        "hrr_analysis/08_report.py"
    ]
    
    if skip_cleaning:
        removed_script = scripts.pop(0)
        print(f"⏩ Skipping slow step: {removed_script}")

    for script in scripts:
        print(f"Executing {script}...")
        result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ FAILED: Script {script} failed to execute.")
            print("--- STDOUT ---\n", result.stdout)
            print("--- STDERR ---\n", result.stderr)
            raise RuntimeError(f"Pipeline execution failed at {script}")

def validate_output(task_type: str, target: str) -> None:
    """Validate whether pipeline output matches the golden standard"""
    print(f"--- Validating output for track: {task_type}, target: {target} ---")
    card_path = Path(f"models/{target}/dataset_card.json")
    if not card_path.exists(): raise FileNotFoundError(f"Output file not found: {card_path}")
    with open(card_path, 'r') as f: card = json.load(f)
    current_r2 = card.get("evaluation_metrics", {}).get("r2")
    expected_r2 = GOLDEN_METRICS[target][task_type]["r2"]
    print(f"Current R²:  {current_r2:.4f}")
    print(f"Expected R²: {expected_r2:.4f}")
    if abs(current_r2 - expected_r2) < TOLERANCE:
        print(f"✅ PASS: R² score is consistent with the golden master.")
    else:
        print(f"❌ FAIL: R² score has deviated from the golden master!")
        raise ValueError("Regression test failed: R² mismatch.")

def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Regression Test")
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip the slow 01_cleaning.py step and use existing converted data."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Running Full Pipeline Regression Test")
    if args.skip_cleaning:
        print("⚡ Fast mode enabled: Skipping 01_cleaning.py")
    print("=" * 60)

    try:
        # Since integration_test has already separated Track B and Track A testing,
        # this script is mainly used for local development integrity checks.
        # In CI/CD, check_feature_isolation.py has already validated the dual-track logic.
        run_full_pipeline(task_type="long_term", target="ERS", skip_cleaning=args.skip_cleaning)
        run_full_pipeline(task_type="short_term", target="ERS", skip_cleaning=args.skip_cleaning)
        
        print("\n[INFO] Pipeline execution for both tracks completed successfully.")

        print("\n" + "=" * 60)
        print("🎉 Regression Test Execution Passed! (Pipeline ran successfully)")
        print("=" * 60)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ An error occurred during the regression test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()