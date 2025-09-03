# hrr_analysis/config.py
# Please place this file under the hrr_analysis/ folder for centralized management

from typing import List, Set

# --- Global Settings ---
RANDOM_SEED: int = 42
QUALITY_GATE_THRESHOLD: float = 0.05 # R² > 0.05 is considered passing the quality gate

# --- Feature Groups (Based on 04_models.py logic) ---
# These are the core settings used by the real pipeline during feature engineering and model training

# All possible target variables, used for feature exclusion
ALL_POSSIBLE_TARGETS: Set[str] = {
    "ERS", "rmssd_post", "sdnn_post", "pnn50_post", "mean_rr_post", "hrr60"
}

# Metadata columns that should always be excluded from any model
METADATA_COLS: Set[str] = {
    "row_id", "date", "end_apnea_time"
}

# Core components of ERS. Must be excluded in Track B (long_term)
ERS_COMPONENTS: Set[str] = {
    "recovery_ratio_60s", "recovery_ratio_90s", "normalized_slope"
}

# --- Data Quality Settings ---
# Used in data_quality_check.py, based on the actual logic of 01_cleaning.py and 02_features.py

# Valid heart rate range in raw data
HR_VALID_RANGE: tuple[int, int] = (30, 200)

# Critical columns to check after feature engineering
CRITICAL_FEATURE_COLS: List[str] = [
    "ERS", "hrr60", "rmssd_post", "HRbaseline", "HRpeak", "end_apnea_time"
]

# Reasonable value ranges for certain features, used for anomaly detection
FEATURE_EXPECTED_RANGES: dict[str, tuple[float, float]] = {
    'HRbaseline': (40.0, 100.0),
    'HRpeak': (60.0, 200.0),
    'ERS': (0.0, 1.0),
    'recovery_ratio_60s': (0.0, 1.0),
    'recovery_ratio_90s': (0.0, 1.0),
    'baseline_quality_score': (0.0, 1.0),
}