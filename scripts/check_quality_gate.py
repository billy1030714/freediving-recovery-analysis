#!/usr/bin/env python
"""
模型品質門檻檢查 - 驗證模型是否通過在 04_models.py 中定義的固定品質門檻
"""
import json
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))
from hrr_analysis.config import QUALITY_GATE_THRESHOLD

def check_gate(target: str) -> None:
    """檢查指定目標的品質門檻"""
    card_path = Path(f"models/{target}/dataset_card.json")
    
    if not card_path.exists():
        print(f"❌ FAIL: Dataset card for target '{target}' not found at {card_path}")
        sys.exit(1)
    
    with open(card_path, 'r') as f:
        card = json.load(f)
        
    passed = card.get("passed_quality_gate", False)
    r2_score = card.get("evaluation_metrics", {}).get("r2")

    print(f"\n--- Checking Quality Gate for Target: {target} ---")
    print(f"R² Score: {r2_score:.4f}" if r2_score is not None else "R² Score: N/A")
    print(f"Quality Gate Threshold: > {QUALITY_GATE_THRESHOLD}")
    
    if passed:
        print(f"✅ PASS: Model for '{target}' passed the quality gate.")
    else:
        print(f"❌ FAIL: Model for '{target}' did NOT pass the quality gate.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Check model quality gate from dataset_card.json")
    parser.add_argument("--targets", type=str, required=True, help="Comma-separated list of targets to check (e.g., 'ERS,rmssd_post')")
    args = parser.parse_args()
    
    targets_to_check = [t.strip() for t in args.targets.split(',') if t.strip()]
    
    print("=" * 60)
    print("Running Model Quality Gate Checks")
    print("=" * 60)
    
    if not targets_to_check:
        print("⚠️ No targets specified. Skipping check.")
        sys.exit(0)

    for target in targets_to_check:
        try:
            check_gate(target)
        except Exception as e:
            print(f"\nAn error occurred while checking target '{target}': {e}")
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print("🎉 All specified targets passed their quality gates!")
    print("=" * 60)

if __name__ == "__main__":
    main()