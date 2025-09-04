"""
02_features.py - Heart Rate Recovery Feature Engineering Pipeline
Extracts physiological recovery metrics from Apple Watch heart rate data.
"""

# --- Library Imports ---
import os
import yaml
import logging
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

# --- Local Modules ---
from paths import DIR_CONVERTED, DIR_FEATURES, get_daily_path

# --- Constants and Settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
DATE_RE = re.compile(r"(\d{8})") # Matches YYYYMMDD date format in filenames
FINAL_COLUMNS: List[str] = [
    # Event Identifiers
    "row_id", "date", "end_apnea_time",
    # Core HR Metrics
    "HRbaseline", "HRpeak",
    # Raw & Composite Recovery Metrics
    "hrr60", "recovery_ratio_60s", "recovery_ratio_90s", "normalized_slope", 
    "ERS", "ers_feature_count", 
    # HRV Proxy Features
    "rmssd_post", "sdnn_post", "pnn50_post", "mean_rr_post",
    # Contextual Features
    "personal_baseline_28d", "baseline_diff", 
    "time_since_last_apnea", "baseline_quality_score"
]

# --- Data Structure Definitions ---
@dataclass(frozen=True)
class DayFiles:
    date_str: str
    hr_path: Path
    apnea_path: Path

@dataclass(frozen=True)
class LoadedDay:
    date_str: str
    hr_df: pd.DataFrame
    apnea_df: pd.DataFrame

# --- Core Functions ---
def _get_hr_with_tolerance(df: pd.DataFrame, target_time: pd.Timestamp, tol: int = 3) -> Optional[float]:
    """
    TEMPORAL TOLERANCE MATCHING:
    Apple Watch sampling is irregular (~1-5 second intervals), so we search
    within ±3 second window and select the closest timestamp match.
    """
    df_win = df[(df["Time"] >= target_time - timedelta(seconds=tol)) & (df["Time"] <= target_time + timedelta(seconds=tol))]
    if df_win.empty: return None
    idx = (df_win["Time"] - target_time).abs().idxmin()
    return None if pd.isna(idx) else df_win.loc[idx, "HR"]

def _lin_slope_with_tolerance(df: pd.DataFrame, ts: pd.Timestamp, te: pd.Timestamp, tol: int = 3) -> Optional[float]:
    win_df = df[(df["Time"] >= ts - timedelta(seconds=tol)) & (df["Time"] <= te + timedelta(seconds=tol))]
    if len(win_df) < 2: return None
    x = (win_df["Time"] - win_df["Time"].iloc[0]).dt.total_seconds().to_numpy()
    y = win_df["HR"].to_numpy()
    if len(x) != len(y) or len(x) == 0: return None
    return np.polyfit(x, y, 1)[0]

def _clip01(v): 
    return float(np.clip(v, 0.0, 1.0)) if v is not None and np.isfinite(v) else None

def _get_dynamic_ideal_slope(starting_hr: float, resting_hr: float = 60.0, peak_hr: float = 130.0, max_ideal_slope: float = 1.0) -> float:
    """
    PHYSIOLOGICALLY-ADAPTIVE NORMALIZATION:
    Calculates expected recovery slope based on individual's HR range.
    Higher starting HR → steeper expected slope for equivalent recovery quality.
    """
    if starting_hr <= resting_hr: return 0.1
    scale = (starting_hr - resting_hr) / (peak_hr - resting_hr)
    ideal_slope = max_ideal_slope * scale
    return max(0.1, ideal_slope)

def analyze_event(hr_df: pd.DataFrame, end_apnea_time: pd.Timestamp, config: Dict) -> Optional[Dict]:
    """
    CORE RECOVERY ANALYSIS ENGINE:
    
    Processing Pipeline:
    1. Baseline calculation (pre-apnea window)
    2. Peak detection (post-apnea search window)  
    3. Recovery ratio calculation at 60s and 90s
    4. Slope analysis with peak-timing adaptation
    5. HRV proxy feature extraction
    6. ERS composite score generation
    
    Handles edge cases: abnormal peaks, missing data, irregular sampling
    """
    if pd.isna(end_apnea_time): return None
    df = hr_df.copy().dropna(subset=["Time", "HR"]).sort_values("Time")
    if df.empty: return None
    event_t0 = end_apnea_time

    base_df = df[(df["Time"] >= event_t0 - timedelta(seconds=config['base_window'][0])) &
                 (df["Time"] < event_t0 - timedelta(seconds=config['base_window'][1]))]
    baseline = base_df["HR"].mean() if not base_df.empty else None

    peak_search_df = df[(df["Time"] >= event_t0) &
                        (df["Time"] <= event_t0 + timedelta(seconds=config['peak_max_seconds']))]
    if peak_search_df.empty or peak_search_df['HR'].isnull().all(): return None
    peak = peak_search_df["HR"].max()
    if pd.isna(peak): return None

    time_peak_hr_timestamp = peak_search_df.loc[peak_search_df['HR'].idxmax(), 'Time']
    time_to_peak_hr = (time_peak_hr_timestamp - event_t0).total_seconds()
    
    slope_window_config = config['slope_w1'] if time_to_peak_hr <= 30 else config['slope_w2']
    ts_slope = event_t0 + timedelta(seconds=slope_window_config[0])
    te_slope = event_t0 + timedelta(seconds=slope_window_config[1])
    is_peak_abnormal = (time_to_peak_hr > 45) or (ts_slope <= time_peak_hr_timestamp <= te_slope)
    # PEAK ANOMALY DETECTION: Skip slope calculation if peak occurs too late (>45s)
    # or falls within slope measurement window (would bias the slope calculation)
    slope = np.nan if is_peak_abnormal else _lin_slope_with_tolerance(df, ts_slope, te_slope, tol=3)

    hr60 = _get_hr_with_tolerance(df, event_t0 + timedelta(seconds=60))
    hr90 = _get_hr_with_tolerance(df, event_t0 + timedelta(seconds=90))
    denom = peak - baseline if baseline is not None and peak > baseline else 0
    def _rr(hr_x):
        return _clip01((peak - hr_x) / denom) if denom > 0 and hr_x is not None else None
    rr60, rr90 = _rr(hr60), _rr(hr90)
    hrr60 = (peak - hr60) if peak is not None and hr60 is not None else None

    normalized_slope = None
    if slope is not None and np.isfinite(slope):
        hr_slope_start = _get_hr_with_tolerance(df, ts_slope)
        if hr_slope_start and baseline and peak:
            dynamic_ideal_slope = _get_dynamic_ideal_slope(hr_slope_start, baseline, peak)
            normalized_slope = _clip01(abs(slope) / dynamic_ideal_slope)

    valid_metrics = [v for v in (rr60, rr90, normalized_slope) if v is not None]
    ers = np.mean(valid_metrics) if valid_metrics else None

    hrv_df = df[(df["Time"] >= event_t0 + timedelta(seconds=config['hrv_window'][0])) &
                (df["Time"] <= event_t0 + timedelta(seconds=config['hrv_window'][1]))]
    rmssd, sdnn, pnn50, mean_rr = None, None, None, None
    if len(hrv_df) >= 10:
        rr_intervals = (60000.0 / hrv_df["HR"]).to_numpy()
        # HRV PROXY CALCULATION: Convert HR to RR intervals (milliseconds)
        # Formula: RR = 60000 / HR, where 60000 = 60 seconds * 1000 ms/second
        if len(rr_intervals) > 1:
            diffs = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(diffs**2))
            sdnn = np.std(rr_intervals)
            pnn50 = (np.sum(np.abs(diffs) > 50) / len(diffs)) * 100
            mean_rr = np.mean(rr_intervals)

    return {"end_apnea_time": event_t0, "HRbaseline": baseline, "HRpeak": peak, "hrr60": hrr60,
            "recovery_ratio_60s": rr60, "recovery_ratio_90s": rr90, "normalized_slope": normalized_slope, 
            "ERS": ers, "ers_feature_count": len(valid_metrics), "rmssd_post": rmssd, "sdnn_post": sdnn,
            "pnn50_post": pnn50, "mean_rr_post": mean_rr}

def discover_day_files(converted_dir: Path, suffix: str = '') -> List[DayFiles]:
    day_files = []
    for hr_path in sorted(converted_dir.glob(f"hr{suffix}_*.csv")):
        if match := DATE_RE.search(hr_path.name):
            date_str = match.group(1)
            apnea_path = converted_dir / f'apnea_events{suffix}_{date_str}.csv'
            if apnea_path.exists():
                day_files.append(DayFiles(date_str, hr_path, apnea_path))
    logging.info(f"Discovered {len(day_files)} paired data files with suffix '{suffix}'.")
    return day_files

def load_day(day: DayFiles) -> LoadedDay:
    hr_df = pd.read_csv(day.hr_path, parse_dates=["Time"])
    apnea_df = pd.read_csv(day.apnea_path, parse_dates=["end_apnea"])
    if "row_id" not in apnea_df.columns and "end_apnea" in apnea_df.columns: 
        apnea_df["row_id"] = apnea_df["end_apnea"].dt.strftime("%Y%m%d_%H%M%S")
    apnea_df["row_id"] = apnea_df["row_id"].astype(str)
    return LoadedDay(day.date_str, hr_df, apnea_df)

def process_day(day_data: LoadedDay, config: Dict) -> List[Dict]:
    new_features = []
    for _, event in day_data.apnea_df.iterrows():
        features = analyze_event(day_data.hr_df, event["end_apnea"], config)
        if features:
            features["row_id"] = event["row_id"]
            features["date"] = day_data.date_str
            new_features.append(features)
    return new_features

def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.sort_values("end_apnea_time").reset_index(drop=True)
    rolling_base = df.rolling(on='end_apnea_time', window='28D', closed='left')['HRbaseline'].median()
    df['personal_baseline_28d'] = rolling_base.to_numpy()
    df['baseline_diff'] = df['HRbaseline'] - df['personal_baseline_28d']
    df['time_since_last_apnea'] = df['end_apnea_time'].diff().dt.total_seconds()
    conditions = [df['baseline_diff'].abs() <= 3, df['baseline_diff'].abs() <= 5]
    # BASELINE QUALITY SCORING: 1.0 for excellent (±3 bpm), 0.5 for good (±5 bpm), 0.0 for poor
    df['baseline_quality_score'] = np.select(conditions, [1.0, 0.5], default=0.0)
    return df

def merge_and_finalize(new_features: List[Dict], features_dir: Path, suffix: str = '') -> pd.DataFrame:
    if not new_features:
        logging.warning("No new features were generated.")
        return pd.DataFrame()

    new_df = pd.DataFrame(new_features)
    features_csv_path = features_dir / f"features{suffix}.csv"
    
    old_df = pd.read_csv(features_csv_path) if features_csv_path.exists() else pd.DataFrame()
    combined_df = pd.concat([old_df, new_df], ignore_index=True)

    if combined_df.empty: return combined_df

    combined_df["end_apnea_time"] = pd.to_datetime(combined_df["end_apnea_time"])
    final_df = combined_df.sort_values("end_apnea_time").drop_duplicates(subset=["row_id"], keep="last")
    final_df = add_contextual_features(final_df)
    
    final_cols_ordered = [col for col in FINAL_COLUMNS if col in final_df.columns]
    final_df = final_df[final_cols_ordered]
    
    features_dir.mkdir(exist_ok=True, parents=True)
    final_df.to_csv(features_csv_path, index=False)
    
    try:
        parquet_path = features_dir / f"features_ml{suffix}.parquet"
        final_df.to_parquet(parquet_path, index=False)
        logging.info(f"Successfully saved Parquet file: {parquet_path}")
    except Exception as e:
        logging.warning(f"Could not save as Parquet format: {e}")
        
    return final_df

def main():
    """Main execution function with CI/CD awareness."""
    logging.info("===== Starting execution of 02_features.py =====")

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    is_ci = os.getenv('CI') or os.getenv('GITHUB_ACTIONS')
    file_suffix = '_ci' if is_ci else ''
    if is_ci:
        logging.info("CI environment detected. Reading files with suffix '_ci'.")

    day_files = discover_day_files(DIR_CONVERTED, file_suffix)
    if not day_files:
        logging.warning("No paired data files found. Script will terminate.")
        DIR_FEATURES.mkdir(exist_ok=True, parents=True)
        (DIR_FEATURES / f"features{file_suffix}.csv").touch()
        (DIR_FEATURES / f"features_ml{file_suffix}.parquet").touch()
        return

    all_new_features = [feature for day in day_files 
                        for feature in process_day(load_day(day), params['feature_engineering'])]

    final_df = merge_and_finalize(all_new_features, DIR_FEATURES, file_suffix)
    logging.info("===== Script execution finished =====")
    if not final_df.empty:
        logging.info(f"Final features file saved to: {DIR_FEATURES / f'features{file_suffix}.csv'} ({len(final_df)} records)")

if __name__ == "__main__":
    main()