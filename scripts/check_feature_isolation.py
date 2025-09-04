#!/usr/bin/env python
"""
Feature Isolation Check - Verify that the feature sets of Track A (short_term) and Track B (long_term) meet expectations (revised logic)
"""
import sys
import json
from pathlib import Path
import os
import subprocess

# Direct constants (no config.py dependency)
ERS_COMPONENTS = {"recovery_ratio_60s", "recovery_ratio_90s", "normalized_slope"}
ALL_POSSIBLE_TARGETS = {"ERS", "rmssd_post", "sdnn_post", "pnn50_post", "mean_rr_post", "hrr60"}

sys.path.append(str(Path(__file__).resolve().parent.parent))

def run_pipeline_for_track(task_type: str, targets: str) -> bool:
    """Run model training for the specified track to generate schema"""
    print(f"\n--- Running pipeline for [{task_type}] track ---")
    env = os.environ.copy()
    env["TASK_TYPE"] = task_type
    env["TARGETS"] = targets

    result = subprocess.run(
        [sys.executable, "hrr_analysis/04_models.py"],
        env=env,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Pipeline run for [{task_type}] track failed!")
        print(result.stderr)
        return False
    print(f"✅ Pipeline run for [{task_type}] track successful.")
    return True

def get_features_from_schema(target: str) -> set:
    """Read feature set from feature_schema.json"""
    schema_path = Path(f"models/{target}/feature_schema.json")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found for target '{target}' at {schema_path}")
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return set(schema['features'])

def main():
    print("=" * 60)
    print("Verifying Feature Isolation in Dual-Track Models")
    print("=" * 60)

    try:
        # 1. Run Track B (long_term) and *immediately* read its feature set
        if not run_pipeline_for_track("long_term", "ERS"): sys.exit(1)
        # MEMORY-BASED COMPARISON: Capture feature sets immediately after each run
        # to avoid file overwrite issues during dual-track validation
        features_b_track = get_features_from_schema("ERS")
        print(f"[Track B - long_term] captured {len(features_b_track)} features.")

        # 2. Run Track A (short_term) and *immediately* read its feature set
        # Track A will overwrite the same files, but we've already captured Track B in memory
        if not run_pipeline_for_track("short_term", "ERS"): sys.exit(1)
        features_a_track = get_features_from_schema("ERS")
        print(f"[Track A - short_term] captured {len(features_a_track)} features.")

        # 3. Now compare the two *different* feature sets in memory
        print("\n--- Performing Isolation Checks on Captured Features ---")
        
        # Check 1: Does Track A include the ERS components as expected?
        if ERS_COMPONENTS.issubset(features_a_track):
            print("✅ PASS: Track A (short_term) correctly includes ERS components.")
        else:
            print("❌ FAIL: Track A (short_term) is missing ERS components!")
            print(f"   Missing: {ERS_COMPONENTS - features_a_track}")
            sys.exit(1)

        # Check 2: Does Track B successfully exclude all recovery-related features?
        leakage_features = (ERS_COMPONENTS | ALL_POSSIBLE_TARGETS) & features_b_track
        
        if not leakage_features:
            print("✅ PASS: Track B (long_term) correctly excludes all post-dive and target features.")
        else:
            print("❌ FAIL: Track B (long_term) has data leakage!")
            print(f"   Leaked features found: {leakage_features}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("🎉 Feature Isolation Check Passed! Dual-track logic is correctly implemented.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()