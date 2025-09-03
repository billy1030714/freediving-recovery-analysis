#!/usr/bin/env python
"""
Data Quality Check Script - Validate final data quality after feature engineering (optimized logic version)
"""
import sys
import pandas as pd
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from hrr_analysis.config import CRITICAL_FEATURE_COLS, FEATURE_EXPECTED_RANGES
except ImportError:
    print("❌ Unable to import config.py. Please make sure hrr_analysis/config.py exists and the path is correct.")
    sys.exit(1)

def check_missing_values(df: pd.DataFrame, critical_columns: list[str], threshold_pct: float = 10.0) -> dict:
    """Check for missing values in critical columns and determine status based on threshold"""
    results = {"status": "PASSED", "messages": []}
    is_fail = False
    is_warn = False
    
    missing_info = df[critical_columns].isnull().sum()
    
    for col, count in missing_info.items():
        if count > 0:
            pct = (count / len(df)) * 100
            msg = f"{col}: {count} missing ({pct:.2f}%)"
            results["messages"].append(msg)
            
            if pct > threshold_pct:
                is_fail = True
            is_warn = True # mark as warning if any missing values exist

    if is_fail:
        results["status"] = "FAILED"
        print(f"❌ FAIL: Missing rate in critical columns exceeds {threshold_pct}%: {results['messages']}")
    elif is_warn:
        results["status"] = "WARNING"
        print(f"⚠️ WARN: Minor missing values found in critical columns (model will use Imputer): {results['messages']}")
    else:
        print("✅ PASS: No missing values in critical columns.")
        
    return results

def check_feature_ranges(df: pd.DataFrame, expected_ranges: dict[str, tuple]) -> dict:
    """Check if feature value ranges are reasonable"""
    results = {"status": "PASSED", "invalid_ranges": []}
    for col, (min_val, max_val) in expected_ranges.items():
        if col in df.columns and df[col].notna().any():
            actual_min, actual_max = df[col].min(), df[col].max()
            if actual_min < min_val or actual_max > max_val:
                msg = (f"{col}: Range [{actual_min:.2f}, {actual_max:.2f}] "
                       f"exceeds expected [{min_val}, {max_val}]")
                results["invalid_ranges"].append(msg)
    if results["invalid_ranges"]:
        results["status"] = "WARNING"
        print(f"⚠️ WARN: Some feature values exceed expected ranges: {results['invalid_ranges']}")
    else:
        print("✅ PASS: All feature values are within expected ranges.")
    return results

def check_data_types(df: pd.DataFrame) -> dict:
    """Check if time column is in datetime format"""
    results = {"status": "PASSED", "message": ""}
    time_col = 'end_apnea_time'
    if time_col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            print("✅ PASS: Time column 'end_apnea_time' has correct datetime format.")
        else:
            results["status"] = "FAILED"
            results["message"] = f"Time column '{time_col}' is not in datetime format."
            print(f"❌ FAIL: {results['message']}")
    return results

def main():
    print("=" * 60)
    print("Running Data Quality Checks on Engineered Features")
    print("=" * 60)
    
    data_path = Path("features/features_ml.parquet")
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        sys.exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns from {data_path}")

    if len(df) == 0 or len(df.columns) == 0:
        print("⚠️ WARN: Empty dataset detected (likely from sample data with no apnea events)")
        print("🎯 CI Mode: This is expected behavior for smoke testing - exiting gracefully")
        sys.exit(0)

    checks = []
    
    print("\n--- 1. Check Missing Values in Critical Columns ---")
    checks.append(check_missing_values(df, CRITICAL_FEATURE_COLS))
    
    print("\n--- 2. Check Feature Value Ranges ---")
    checks.append(check_feature_ranges(df, FEATURE_EXPECTED_RANGES))
    
    print("\n--- 3. Check Data Types ---")
    checks.append(check_data_types(df))
    
    statuses = [c["status"] for c in checks]
    if "FAILED" in statuses:
        overall_status = "FAILED"
    elif "WARNING" in statuses:
        overall_status = "WARNING"
    else:
        overall_status = "PASSED"
        
    print("\n" + "=" * 60)
    if overall_status == "FAILED":
        print("❌ Data Quality Check FAILED. Pipeline cannot proceed.")
        sys.exit(1)
    elif overall_status == "WARNING":
        print("⚠️ Data Quality Check PASSED with warnings. Please review.")
        sys.exit(0)
    else:
        print("🎉 Data Quality Check PASSED.")
        sys.exit(0)

if __name__ == "__main__":
    main()