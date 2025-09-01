# hrr_analysis/config.py
# 請將此檔案放在 hrr_analysis/ 資料夾下，以便統一管理

from typing import List, Set

# --- Global Settings ---
RANDOM_SEED: int = 42
QUALITY_GATE_THRESHOLD: float = 0.05 # R² > 0.05 才算通過品質門檻

# --- Feature Groups (Based on 04_models.py logic) ---
# 這些是真實 Pipeline 在進行特徵工程和模型訓練時依賴的核心設定

# 所有可能的目標變數，用於特徵排除
ALL_POSSIBLE_TARGETS: Set[str] = {
    "ERS", "rmssd_post", "sdnn_post", "pnn50_post", "mean_rr_post", "hrr60"
}

# 元數據欄位，在任何模型中都應被排除
METADATA_COLS: Set[str] = {
    "row_id", "date", "end_apnea_time"
}

# ERS 的核心組成特徵。在 B 軌 (long_term) 中必須被排除
ERS_COMPONENTS: Set[str] = {
    "recovery_ratio_60s", "recovery_ratio_90s", "normalized_slope"
}

# --- Data Quality Settings ---
# 用於 data_quality_check.py，基於 01_cleaning.py 和 02_features.py 的真實邏輯

# 原始數據中的心率有效範圍
HR_VALID_RANGE: tuple[int, int] = (30, 200)

# 特徵工程後，應檢查的關鍵欄位
CRITICAL_FEATURE_COLS: List[str] = [
    "ERS", "hrr60", "rmssd_post", "HRbaseline", "HRpeak", "end_apnea_time"
]

# 部分特徵的合理數值範圍，用於異常偵測
FEATURE_EXPECTED_RANGES: dict[str, tuple[float, float]] = {
    'HRbaseline': (40.0, 100.0),
    'HRpeak': (60.0, 200.0),
    'ERS': (0.0, 1.0),
    'recovery_ratio_60s': (0.0, 1.0),
    'recovery_ratio_90s': (0.0, 1.0),
    'baseline_quality_score': (0.0, 1.0),
}