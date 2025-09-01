#!/usr/bin/env python
"""
特徵隔離檢查 - 驗證 A軌(short_term) 和 B軌(long_term) 的特徵集符合預期 (邏輯修正版)
"""
import sys
import json
from pathlib import Path
import os
import subprocess

sys.path.append(str(Path(__file__).resolve().parent.parent))
from hrr_analysis.config import ERS_COMPONENTS, ALL_POSSIBLE_TARGETS

def run_pipeline_for_track(task_type: str, targets: str) -> bool:
    """為指定的軌道運行模型訓練，以生成 schema"""
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
    """從 feature_schema.json 讀取特徵集"""
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
        # --- 修正後的邏輯 ---
        # 1. 運行 B 軌 (long_term) 並 *立刻* 讀取其特徵集
        if not run_pipeline_for_track("long_term", "ERS"): sys.exit(1)
        features_b_track = get_features_from_schema("ERS")
        print(f"[Track B - long_term] captured {len(features_b_track)} features.")

        # 2. 運行 A 軌 (short_term) 並 *立刻* 讀取其特徵集 (這會覆蓋文件，但沒關係)
        if not run_pipeline_for_track("short_term", "ERS"): sys.exit(1)
        features_a_track = get_features_from_schema("ERS")
        print(f"[Track A - short_term] captured {len(features_a_track)} features.")

        # 3. 現在，在記憶體中對兩個 *不同* 的特徵集進行比較
        print("\n--- Performing Isolation Checks on Captured Features ---")
        
        # 檢查 1: A 軌是否如預期包含了 ERS 組件？
        if ERS_COMPONENTS.issubset(features_a_track):
            print("✅ PASS: Track A (short_term) correctly includes ERS components.")
        else:
            print("❌ FAIL: Track A (short_term) is missing ERS components!")
            print(f"   Missing: {ERS_COMPONENTS - features_a_track}")
            sys.exit(1)

        # 檢查 2: B 軌是否成功排除了所有恢復期特徵？
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