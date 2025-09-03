"""
02_features.py – Heart Rate Recovery Feature Engineering
Calculates recovery features and contextual metrics.
"""
# --- Library Imports ---
import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# --- Local Modules ---
from paths import DIR_CONVERTED, DIR_FEATURES, get_daily_path

# --- Constants and Settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

@dataclass(frozen=True)
class FeatureConfig:
    """Configuration parameters for feature calculation."""
    BASE_WINDOW: Tuple[int, int] = (330, 150)
    SLOPE_W1: Tuple[int, int] = (30, 90)
    SLOPE_W2: Tuple[int, int] = (45, 105)
    PEAK_MAX_SECONDS: int = 120
    HRV_WINDOW: Tuple[int, int] = (180, 360)
    FINAL_COLUMNS: List[str] = field(default_factory=lambda: [
        # Event Identifiers
        "row_id", "date", "end_apnea_time",
        # Core HR Metrics
        "HRbaseline", "HRpeak",
        # Raw & Composite Recovery Metrics
        "hrr60",  # <-- Added
        "recovery_ratio_60s", "recovery_ratio_90s", "normalized_slope", 
        "ERS", "ers_feature_count", 
        # HRV Proxy Features
        "rmssd_post", "sdnn_post", "pnn50_post", "mean_rr_post",
        # Contextual Features
        "personal_baseline_28d", "baseline_diff", 
        "time_since_last_apnea", "baseline_quality_score"
    ])

CONFIG = FeatureConfig()
DATE_RE = re.compile(r"(\d{8})")

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
    Get HR at target_time with ±tol seconds tolerance (to handle irregular Apple Watch HR sampling).
    """
    df_win = df[(df["Time"] >= target_time - timedelta(seconds=tol)) &
                (df["Time"] <= target_time + timedelta(seconds=tol))]
    if df_win.empty:
        return None
    idx = (df_win["Time"] - target_time).abs().idxmin()
    if pd.isna(idx):
        return None
    return df_win.loc[idx, "HR"]


def _lin_slope_with_tolerance(df: pd.DataFrame, ts: pd.Timestamp, te: pd.Timestamp, tol: int = 3) -> Optional[float]:
    """
    Compute slope with ±tol seconds tolerance (to handle irregular Apple Watch HR sampling).
    """
    win_df = df[(df["Time"] >= ts - timedelta(seconds=tol)) & (df["Time"] <= te + timedelta(seconds=tol))]
    if len(win_df) < 2:
        return None
    
    x = (win_df["Time"] - win_df["Time"].iloc[0]).dt.total_seconds().to_numpy()
    y = win_df["HR"].to_numpy()
    if len(x) != len(y) or len(x) == 0:
        return None
    
    coeffs = np.polyfit(x, y, 1)  # linear fit
    return coeffs[0]  # slope

def _clip01(v): 
    """Clips a numeric value to the [0, 1] interval."""
    return float(np.clip(v, 0.0, 1.0)) if v is not None and np.isfinite(v) else None

def _get_dynamic_ideal_slope(starting_hr: float, resting_hr: float = 60.0, peak_hr: float = 130.0, max_ideal_slope: float = 1.0) -> float:
    """
    Calculates a dynamic ideal slope based on the starting HR of the recovery window.
    The ideal slope is scaled linearly between the resting HR and a typical peak HR.
    
    Args:
        starting_hr: The heart rate at the start of the slope calculation window
        resting_hr: The baseline resting heart rate (default: 60 bpm)
        peak_hr: The typical peak heart rate during exercise (default: 130 bpm)  
        max_ideal_slope: The maximum ideal slope value (default: 1.0)
    
    Returns:
        The dynamically calculated ideal slope value
    """
    if starting_hr <= resting_hr:
        return 0.1 # Return a small non-zero value to avoid division by zero
    
    # Linearly interpolate the ideal slope
    scale = (starting_hr - resting_hr) / (peak_hr - resting_hr)
    ideal_slope = max_ideal_slope * scale
    
    return max(0.1, ideal_slope) # Ensure it's at least a small positive number

def analyze_event(hr_df: pd.DataFrame, end_apnea_time: pd.Timestamp) -> Optional[Dict]:
    """
    Performs a full feature calculation for a single apnea event.
    This version uses a dynamic benchmark for slope normalization,
    making the feature context-aware of the starting heart rate level.
    """
    # --- Step 1: Baseline and Peak Detection (unchanged) ---
    if pd.isna(end_apnea_time): return None
    df = hr_df.copy().dropna(subset=["Time", "HR"]).sort_values("Time")
    if df.empty: return None
    event_t0 = end_apnea_time
    base_df = df[(df["Time"] >= event_t0 - timedelta(seconds=CONFIG.BASE_WINDOW[0])) &
                 (df["Time"] < event_t0 - timedelta(seconds=CONFIG.BASE_WINDOW[1]))]
    baseline = base_df["HR"].mean() if not base_df.empty else None
    peak_search_df = df[(df["Time"] >= event_t0) &
                        (df["Time"] <= event_t0 + timedelta(seconds=CONFIG.PEAK_MAX_SECONDS))]
    if peak_search_df.empty or peak_search_df['HR'].isnull().all(): return None
    peak = peak_search_df["HR"].max()
    if pd.isna(peak): return None

    # --- Step 2: Calculate Raw Slope using Tolerance (unchanged) ---
    time_peak_hr_timestamp = peak_search_df.loc[peak_search_df['HR'].idxmax(), 'Time']
    time_to_peak_hr = (time_peak_hr_timestamp - event_t0).total_seconds()
    slope_window_config = CONFIG.SLOPE_W1 if time_to_peak_hr <= 30 else CONFIG.SLOPE_W2
    ts_slope = event_t0 + timedelta(seconds=slope_window_config[0])
    te_slope = event_t0 + timedelta(seconds=slope_window_config[1])
    is_peak_abnormal = (time_to_peak_hr > 45) or (ts_slope <= time_peak_hr_timestamp <= te_slope)
    slope = np.nan if is_peak_abnormal else _lin_slope_with_tolerance(df, ts_slope, te_slope, tol=3)

    # --- Step 3: Calculate Relative Recovery Magnitude (rr60, rr90) (unchanged) ---
    hr60 = _get_hr_with_tolerance(df, event_t0 + timedelta(seconds=60))
    hr90 = _get_hr_with_tolerance(df, event_t0 + timedelta(seconds=90))
    denom = peak - baseline if baseline is not None and peak > baseline else 0
    def _rr(hr_x):
        if denom == 0 or hr_x is None: return None
        return _clip01((peak - hr_x) / denom)
    rr60, rr90 = _rr(hr60), _rr(hr90)
    hrr60 = (peak - hr60) if peak is not None and hr60 is not None else None

    # --- Step 4: [FIXED] Calculate Context-Aware Normalized Slope with proper parameters ---
    normalized_slope = None
    if slope is not None and np.isfinite(slope):
        # Get the heart rate at the start of the slope window to understand the context
        hr_slope_start = _get_hr_with_tolerance(df, ts_slope)
        if hr_slope_start is not None and baseline is not None and peak is not None:
            # Calculate the ideal slope for this specific context using actual baseline and peak
            dynamic_ideal_slope = _get_dynamic_ideal_slope(
                starting_hr=hr_slope_start,
                resting_hr=baseline,  # Use actual baseline instead of fixed 60
                peak_hr=peak,         # Use actual peak instead of fixed 130
                max_ideal_slope=1.0
            )
            # Normalize the actual slope against the dynamic ideal slope
            normalized_slope = _clip01(abs(slope) / dynamic_ideal_slope)

    # --- Step 5: Final ERS Calculation (unchanged) ---
    valid_metrics = [v for v in (rr60, rr90, normalized_slope) if v is not None]
    ers = np.mean(valid_metrics) if valid_metrics else None
    ers_feature_count = len(valid_metrics)

    # --- Step 6: [UPGRADED] Comprehensive HRV Proxy Feature Calculation ---
    hrv_df = df[(df["Time"] >= event_t0 + timedelta(seconds=CONFIG.HRV_WINDOW[0])) &
                (df["Time"] <= event_t0 + timedelta(seconds=CONFIG.HRV_WINDOW[1]))]
    
    # Initialize all HRV metrics to None to ensure they exist, preventing errors later.
    rmssd, sdnn, pnn50, mean_rr = None, None, None, None

    # We need at least 10 HR samples to get a somewhat reliable HRV reading.
    if len(hrv_df) >= 10:
        # Convert HR (beats/min) to RR intervals (milliseconds).
        rr_intervals = (60000.0 / hrv_df["HR"]).to_numpy()
        
        # Check if we have at least 2 intervals to calculate differences.
        if len(rr_intervals) > 1:
            diffs = np.diff(rr_intervals)
            
            # RMSSD: Root mean square of successive differences. Sensitive to parasympathetic activity.
            rmssd = np.sqrt(np.mean(diffs**2))
            
            # SDNN: Standard deviation of all NN (RR) intervals. Reflects overall variability.
            sdnn = np.std(rr_intervals)

            # pNN50: Percentage of successive differences > 50ms. Also reflects parasympathetic control.
            pnn50 = (np.sum(np.abs(diffs) > 50) / len(diffs)) * 100
            
            # Mean RR: Average RR interval duration. Inversely related to average heart rate.
            mean_rr = np.mean(rr_intervals)

    # --- Step 7: Final Return Dictionary ---
    # Consolidate all calculated features into a single dictionary for this event.
    return {
        "end_apnea_time": event_t0, 
        "HRbaseline": baseline, 
        "HRpeak": peak,
        "HR_peak_time_post_apnea": time_to_peak_hr,
        "hrr60": hrr60,
        # ERS related metrics
        "recovery_ratio_60s": rr60, 
        "recovery_ratio_90s": rr90,
        "normalized_slope": normalized_slope, 
        "ERS": ers,
        "ers_feature_count": ers_feature_count,
        # HRV Proxy Features
        "rmssd_post": rmssd,
        "sdnn_post": sdnn,
        "pnn50_post": pnn50,
        "mean_rr_post": mean_rr
    }

def discover_day_files(converted_dir: Path) -> List[DayFiles]:
    """Scans the `converted` directory to find paired hr and apnea_events files."""
    day_files = []
    for hr_path in sorted(converted_dir.glob("hr_*.csv")):
        if match := DATE_RE.search(hr_path.name):
            date_str = match.group(1)
            apnea_path = get_daily_path(directory=converted_dir, data_type='apnea_events', date_obj=pd.to_datetime(date_str).date(), extension='.csv')
            if apnea_path.exists(): 
                day_files.append(DayFiles(date_str, hr_path, apnea_path))
    logging.info(f"Successfully discovered {len(day_files)} paired daily data files.")
    return day_files

def load_day(day: DayFiles) -> LoadedDay:
    """Loads HR and Apnea data for a single day from DayFiles paths into DataFrames."""
    hr_df = pd.read_csv(day.hr_path, parse_dates=["Time"])
    apnea_df = pd.read_csv(day.apnea_path, parse_dates=["end_apnea"])
    if "row_id" not in apnea_df.columns and "end_apnea" in apnea_df.columns: 
        apnea_df["row_id"] = apnea_df["end_apnea"].dt.strftime("%Y%m%d_%H%M%S")
    apnea_df["row_id"] = apnea_df["row_id"].astype(str)
    return LoadedDay(day.date_str, hr_df, apnea_df)

def process_day(day_data: LoadedDay) -> List[Dict]:
    """Iterates through all apnea events of a single day and calls analyze_event for each."""
    new_features = []
    for _, event in day_data.apnea_df.iterrows():
        features = analyze_event(day_data.hr_df, event["end_apnea"])
        if features:
            features["row_id"] = event["row_id"]
            features["date"] = day_data.date_str
            new_features.append(features)
    return new_features

def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates features that require a global context, such as rolling metrics and time differences."""
    if df.empty: return df
    df['end_apnea_time'] = pd.to_datetime(df['end_apnea_time'])
    df = df.sort_values("end_apnea_time").reset_index(drop=True)

    rolling_base = df.rolling(on='end_apnea_time', window='28D', closed='left')['HRbaseline'].median()
    df['personal_baseline_28d'] = rolling_base.to_numpy()
    df['baseline_diff'] = df['HRbaseline'] - df['personal_baseline_28d']
    df['time_since_last_apnea'] = df['end_apnea_time'].diff().dt.total_seconds()

    conditions = [df['baseline_diff'].abs() <= 3, df['baseline_diff'].abs() <= 5]
    scores = [1.0, 0.5]
    df['baseline_quality_score'] = np.select(conditions, scores, default=0.0)
    return df

def merge_and_finalize(new_features: List[Dict], features_dir: Path) -> pd.DataFrame:
    """Merges newly calculated features, robustly handles duplicates, post-processes, and saves."""
    if not new_features:
        logging.warning("No new features were generated. Reading from existing file if available.")
        features_csv_path = features_dir / "features.csv"
        parquet_path = features_dir / "features_ml.parquet"

        if features_csv_path.exists():
            df = pd.read_csv(features_csv_path)
        else:
            df = pd.DataFrame()

        features_dir.mkdir(exist_ok=True, parents=True)
        try:
            df.to_parquet(parquet_path, index=False)
            logging.info(f"Saved (possibly empty) Parquet file: {parquet_path}")
        except Exception as e:
            logging.warning(f"Could not save as Parquet format: {e}")

        return df

    new_df = pd.DataFrame(new_features)
    features_csv_path = features_dir / "features.csv"
    
    if features_csv_path.exists():
        old_df = pd.read_csv(features_csv_path)
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        
    if combined_df.empty: return combined_df

    # --- Post-processing ---
    combined_df["end_apnea_time"] = pd.to_datetime(combined_df["end_apnea_time"])
    final_df = combined_df.sort_values("end_apnea_time").drop_duplicates(subset=["row_id"], keep="last")
    final_df = add_contextual_features(final_df)
    
    # --- [CRITICAL FIX] Remove any ghost columns created by faulty merges ---
    final_df = final_df.loc[:, ~final_df.columns.str.match(r'.*\.\d$')]
    
    if "date" in final_df.columns:
        final_df["date"] = final_df["date"].astype(str)

    # Ensure final columns are in the correct order and filter out any extras
    final_cols_ordered = [col for col in CONFIG.FINAL_COLUMNS if col in final_df.columns]
    final_df = final_df[final_cols_ordered]
    
    # --- Save artifacts ---
    features_dir.mkdir(exist_ok=True, parents=True)
    final_df.to_csv(features_csv_path, index=False)
    
    try:
        parquet_path = features_dir / "features_ml.parquet"
        final_df.to_parquet(parquet_path, index=False)
        logging.info(f"Successfully saved Parquet file: {parquet_path}")
    except Exception as e:
        logging.warning(f"Could not save as Parquet format: {e}")
        
    return final_df

def main():
    """Main execution function."""
    logging.info("===== Starting execution of 02_features.py (Final Version) =====")
    day_files = discover_day_files(DIR_CONVERTED)
    if not day_files:
        logging.warning("No paired data files found in the `converted` directory. Script will now terminate.")
        return

    all_new_features = []
    for day in day_files:
        try:
            day_data = load_day(day)
            features_from_day = process_day(day_data)
            all_new_features.extend(features_from_day)
        except Exception as e:
            logging.error(f"An error occurred while processing date {day.date_str}: {e}", exc_info=True)

    final_df = merge_and_finalize(all_new_features, DIR_FEATURES)
    logging.info(f"===== Script execution finished =====")
    logging.info(f"Final features file saved to: {DIR_FEATURES / 'features.csv'} ({len(final_df)} records total)")

if __name__ == "__main__":
    main()