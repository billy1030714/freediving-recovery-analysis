#!/usr/bin/env python
"""
數據品質檢查腳本 - 驗證 feature engineering 後的最終數據品質 (邏輯優化版)
"""
import sys
import pandas as pd
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from hrr_analysis.config import CRITICAL_FEATURE_COLS, FEATURE_EXPECTED_RANGES
except ImportError:
    print("❌ 無法導入 config.py。請確保 hrr_analysis/config.py 存在且路徑正確。")
    sys.exit(1)

def check_missing_values(df: pd.DataFrame, critical_columns: list[str], threshold_pct: float = 10.0) -> dict:
    """檢查關鍵欄位的缺失值，並根據閾值決定狀態"""
    results = {"status": "PASSED", "messages": []}
    is_fail = False
    is_warn = False
    
    missing_info = df[critical_columns].isnull().sum()
    
    for col, count in missing_info.items():
        if count > 0:
            pct = (count / len(df)) * 100
            msg = f"{col}: {count} 筆缺失 ({pct:.2f}%)"
            results["messages"].append(msg)
            
            if pct > threshold_pct:
                is_fail = True
            is_warn = True # 只要有缺失就標記為警告

    if is_fail:
        results["status"] = "FAILED"
        print(f"❌ FAIL: 關鍵欄位缺失率超過 {threshold_pct}%: {results['messages']}")
    elif is_warn:
        results["status"] = "WARNING"
        print(f"⚠️ WARN: 關鍵欄位發現少量缺失值 (模型將使用 Imputer 處理): {results['messages']}")
    else:
        print("✅ PASS: 所有關鍵欄位無缺失值。")
        
    return results

# ... (check_feature_ranges 和 check_data_types 函數保持不變) ...
def check_feature_ranges(df: pd.DataFrame, expected_ranges: dict[str, tuple]) -> dict:
    """檢查特徵值範圍是否合理"""
    results = {"status": "PASSED", "invalid_ranges": []}
    for col, (min_val, max_val) in expected_ranges.items():
        if col in df.columns and df[col].notna().any():
            actual_min, actual_max = df[col].min(), df[col].max()
            if actual_min < min_val or actual_max > max_val:
                msg = (f"{col}: 範圍 [{actual_min:.2f}, {actual_max:.2f}] "
                       f"超出預期 [{min_val}, {max_val}]")
                results["invalid_ranges"].append(msg)
    if results["invalid_ranges"]:
        results["status"] = "WARNING"
        print(f"⚠️ WARN: 部分特徵值超出預期範圍: {results['invalid_ranges']}")
    else:
        print("✅ PASS: 所有特徵值均在預期範圍內。")
    return results

def check_data_types(df: pd.DataFrame) -> dict:
    """檢查時間欄位是否為 datetime 格式"""
    results = {"status": "PASSED", "message": ""}
    time_col = 'end_apnea_time'
    if time_col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            print("✅ PASS: 時間欄位 'end_apnea_time' 格式正確。")
        else:
            results["status"] = "FAILED"
            results["message"] = f"時間欄位 '{time_col}' 不是 datetime 格式。"
            print(f"❌ FAIL: {results['message']}")
    return results

def main():
    print("=" * 60)
    print("Running Data Quality Checks on Engineered Features")
    print("=" * 60)
    
    data_path = Path("features/features_ml.parquet")
    if not data_path.exists():
        print(f"❌ 找不到數據檔案: {data_path}")
        sys.exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"載入數據: {len(df)} 筆, {len(df.columns)} 個欄位 from {data_path}")

    checks = []
    
    print("\n--- 1. 檢查關鍵欄位缺失值 ---")
    checks.append(check_missing_values(df, CRITICAL_FEATURE_COLS))
    
    print("\n--- 2. 檢查特徵值範圍 ---")
    checks.append(check_feature_ranges(df, FEATURE_EXPECTED_RANGES))
    
    print("\n--- 3. 檢查資料類型 ---")
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