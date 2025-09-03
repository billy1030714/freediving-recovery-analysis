# ers_tolerance_validation.py
# Validate the effect of different tolerance intervals on ERS feature outputs
# Based on the logic of original features_02.py, testing ±1, ±2, ±3 second tolerances

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class ERSToleranceValidator:
    """ERS Tolerance Validator"""
    
    def __init__(self):
        # Configuration based on features_02.py
        self.BASE_WINDOW = (330, 150)  # baseline window: -330s to -150s
        self.SLOPE_W1 = (30, 90)       # slope window 1: 30s to 90s
        self.SLOPE_W2 = (45, 105)      # slope window 2: 45s to 105s  
        self.PEAK_MAX_SECONDS = 120    # peak search window: 0s to 120s
        
    def _get_hr_with_tolerance(self, df: pd.DataFrame, target_time: pd.Timestamp, tol: int = 3) -> Optional[float]:
        """Get heart rate at a specific time within tolerance"""
        df_win = df[(df["Time"] >= target_time - timedelta(seconds=tol)) &
                    (df["Time"] <= target_time + timedelta(seconds=tol))]
        if df_win.empty:
            return None
        idx = (df_win["Time"] - target_time).abs().idxmin()
        if pd.isna(idx):
            return None
        return df_win.loc[idx, "HR"]

    def _lin_slope_with_tolerance(self, df: pd.DataFrame, ts: pd.Timestamp, te: pd.Timestamp, tol: int = 3) -> Optional[float]:
        """Calculate linear slope within tolerance"""
        win_df = df[(df["Time"] >= ts - timedelta(seconds=tol)) & 
                    (df["Time"] <= te + timedelta(seconds=tol))]
        if len(win_df) < 2:
            return None
        
        x = (win_df["Time"] - win_df["Time"].iloc[0]).dt.total_seconds().to_numpy()
        y = win_df["HR"].to_numpy()
        if len(x) != len(y) or len(x) == 0:
            return None
        
        try:
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0]
        except:
            return None

    def _clip01(self, v):
        """Clip values to [0, 1] range"""
        return float(np.clip(v, 0.0, 1.0)) if v is not None and np.isfinite(v) else None

    def _get_dynamic_ideal_slope(self, starting_hr: float, resting_hr: float = 60.0, 
                                peak_hr: float = 130.0, max_ideal_slope: float = 1.0) -> float:
        """Dynamic ideal slope calculation"""
        if starting_hr <= resting_hr:
            return 0.1
        
        scale = (starting_hr - resting_hr) / (peak_hr - resting_hr)
        ideal_slope = max_ideal_slope * scale
        return max(0.1, ideal_slope)

    def analyze_event_with_tolerance(self, hr_df: pd.DataFrame, end_apnea_time: pd.Timestamp, tolerance: int = 3) -> Optional[Dict]:
        """Analyze a single apnea event using specified tolerance"""
        if pd.isna(end_apnea_time):
            return None
            
        df = hr_df.copy().dropna(subset=["Time", "HR"]).sort_values("Time")
        if df.empty:
            return None
            
        event_t0 = end_apnea_time
        
        # 1. Calculate baseline
        base_df = df[(df["Time"] >= event_t0 - timedelta(seconds=self.BASE_WINDOW[0])) &
                     (df["Time"] < event_t0 - timedelta(seconds=self.BASE_WINDOW[1]))]
        baseline = base_df["HR"].mean() if not base_df.empty else None
        
        # 2. Calculate peak
        peak_search_df = df[(df["Time"] >= event_t0) &
                           (df["Time"] <= event_t0 + timedelta(seconds=self.PEAK_MAX_SECONDS))]
        if peak_search_df.empty or peak_search_df['HR'].isnull().all():
            return None
        peak = peak_search_df["HR"].max()
        if pd.isna(peak):
            return None

        # 3. Calculate slope (with specified tolerance)
        time_peak_hr_timestamp = peak_search_df.loc[peak_search_df['HR'].idxmax(), 'Time']
        time_to_peak_hr = (time_peak_hr_timestamp - event_t0).total_seconds()
        slope_window_config = self.SLOPE_W1 if time_to_peak_hr <= 30 else self.SLOPE_W2
        ts_slope = event_t0 + timedelta(seconds=slope_window_config[0])
        te_slope = event_t0 + timedelta(seconds=slope_window_config[1])
        is_peak_abnormal = (time_to_peak_hr > 45) or (ts_slope <= time_peak_hr_timestamp <= te_slope)
        slope = np.nan if is_peak_abnormal else self._lin_slope_with_tolerance(df, ts_slope, te_slope, tol=tolerance)

        # 4. Calculate recovery ratios (with specified tolerance)
        hr60 = self._get_hr_with_tolerance(df, event_t0 + timedelta(seconds=60), tol=tolerance)
        hr90 = self._get_hr_with_tolerance(df, event_t0 + timedelta(seconds=90), tol=tolerance)
        
        denom = peak - baseline if baseline is not None and peak > baseline else 0
        def _rr(hr_x):
            if denom == 0 or hr_x is None:
                return None
            return self._clip01((peak - hr_x) / denom)
        
        rr60, rr90 = _rr(hr60), _rr(hr90)
        hrr60 = (peak - hr60) if peak is not None and hr60 is not None else None

        # 5. Calculate normalized slope
        normalized_slope = None
        if slope is not None and np.isfinite(slope):
            hr_slope_start = self._get_hr_with_tolerance(df, ts_slope, tol=tolerance)
            if hr_slope_start is not None and baseline is not None and peak is not None:
                dynamic_ideal_slope = self._get_dynamic_ideal_slope(
                    starting_hr=hr_slope_start,
                    resting_hr=baseline,
                    peak_hr=peak,
                    max_ideal_slope=1.0
                )
                normalized_slope = self._clip01(abs(slope) / dynamic_ideal_slope)

        # 6. Calculate ERS
        valid_metrics = [v for v in (rr60, rr90, normalized_slope) if v is not None]
        ers = np.mean(valid_metrics) if valid_metrics else None
        ers_feature_count = len(valid_metrics)

        return {
            "tolerance": tolerance,
            "baseline": baseline,
            "peak": peak,
            "hrr60": hrr60,
            "recovery_ratio_60s": rr60,
            "recovery_ratio_90s": rr90,
            "normalized_slope": normalized_slope,
            "ERS": ers,
            "ers_feature_count": ers_feature_count,
            "slope_raw": slope if np.isfinite(slope) else None
        }

    def validate_tolerance_effects(self, hr_df: pd.DataFrame, apnea_events: List[pd.Timestamp], 
                                 tolerances: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """Validate the impact of different tolerances on ERS features"""
        results = []
        
        for i, end_apnea_time in enumerate(apnea_events):
            event_id = f"event_{i:04d}"
            base_result = {"event_id": event_id, "end_apnea_time": end_apnea_time}
            
            for tol in tolerances:
                analysis = self.analyze_event_with_tolerance(hr_df, end_apnea_time, tolerance=tol)
                
                if analysis is None:
                    # If analysis fails, record NaN
                    for key in ["baseline", "peak", "hrr60", "recovery_ratio_60s", 
                               "recovery_ratio_90s", "normalized_slope", "ERS", "ers_feature_count"]:
                        base_result[f"{key}_tol{tol}"] = np.nan
                else:
                    # Record analysis results
                    for key, value in analysis.items():
                        if key != "tolerance":
                            base_result[f"{key}_tol{tol}"] = value
            
            results.append(base_result)
        
        return pd.DataFrame(results)

    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate summary report"""
        tolerances = [1, 2, 3]
        total_events = len(results_df)
        
        summary = {
            "total_events": total_events,
            "tolerance_comparison": {}
        }
        
        for tol in tolerances:
            ers_col = f"ERS_tol{tol}"
            rr60_col = f"recovery_ratio_60s_tol{tol}"
            rr90_col = f"recovery_ratio_90s_tol{tol}"
            nslope_col = f"normalized_slope_tol{tol}"
            
            if ers_col in results_df.columns:
                nan_count_ers = results_df[ers_col].isna().sum()
                nan_count_rr60 = results_df[rr60_col].isna().sum()
                nan_count_rr90 = results_df[rr90_col].isna().sum()
                nan_count_nslope = results_df[nslope_col].isna().sum()
                
                summary["tolerance_comparison"][f"±{tol}s"] = {
                    "ERS_success_rate": f"{((total_events - nan_count_ers) / total_events * 100):.1f}%",
                    "ERS_nan_count": nan_count_ers,
                    "RR60_nan_count": nan_count_rr60,
                    "RR90_nan_count": nan_count_rr90,
                    "NSlope_nan_count": nan_count_nslope,
                    "avg_feature_count": results_df[f"ers_feature_count_tol{tol}"].mean()
                }
        
        return summary


def load_sample_data(hr_csv_path: str, apnea_csv_path: str) -> tuple:
    """Load sample data - based on features_02.py format"""
    
    # Load HR data
    hr_df = pd.read_csv(hr_csv_path)
    if 'Time' not in hr_df.columns or 'HR' not in hr_df.columns:
        raise ValueError("HR file must contain 'Time' and 'HR' columns")
    hr_df['Time'] = pd.to_datetime(hr_df['Time'])
    
    # Load apnea events data
    apnea_df = pd.read_csv(apnea_csv_path)
    if 'end_apnea' not in apnea_df.columns:
        raise ValueError("Apnea file must contain 'end_apnea' column")
    apnea_df['end_apnea'] = pd.to_datetime(apnea_df['end_apnea'])
    
    # Extract event timestamps
    apnea_events = apnea_df['end_apnea'].dropna().tolist()
    
    return hr_df, apnea_events


def main():
    """Main execution function"""
    logging.info("===== ERS Tolerance Validation Started =====")
    
    # 1. Set file paths - please update with your actual file paths
    hr_csv_path = "converted/hr_20250818.csv"  # HR data file, format: Time, HR
    apnea_csv_path = "converted/apnea_events_20250818.csv"  # Apnea events file, format: end_apnea, row_id
    
    # Alternatively, you can use directory scanning (if you have multiple days of data)
    # converted_dir = Path("converted")  # Directory containing hr_*.csv and apnea_events_*.csv
    
    try:
        hr_df, apnea_events = load_sample_data(hr_csv_path, apnea_csv_path)
        logging.info(f"Data loaded successfully: {len(hr_df)} HR records, {len(apnea_events)} apnea events")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        logging.info("Please ensure the following files exist:")
        logging.info(f"  HR file: {hr_csv_path} (columns: Time, HR)")
        logging.info(f"  Events file: {apnea_csv_path} (columns: end_apnea, row_id)")
        return
    except Exception as e:
        logging.error(f"Error occurred while loading data: {e}")
        return
    
    # 2. Initialize validator
    validator = ERSToleranceValidator()
    
    # 3. Run validation
    logging.info("Analyzing impact of different tolerances...")
    results_df = validator.validate_tolerance_effects(hr_df, apnea_events, tolerances=[1, 2, 3])
    
    # 4. Generate report
    summary = validator.generate_summary_report(results_df)
    
    # 5. Display results
    print("\n" + "="*60)
    print("ERS Tolerance Validation Summary")
    print("="*60)
    print(f"Total events: {summary['total_events']}")
    print("\nSuccess rates for each tolerance setting:")
    
    for tolerance, stats in summary['tolerance_comparison'].items():
        print(f"\n{tolerance} tolerance:")
        print(f"  ERS success rate: {stats['ERS_success_rate']}")
        print(f"  ERS NaN count: {stats['ERS_nan_count']}")
        print(f"  RR60 NaN count: {stats['RR60_nan_count']}")
        print(f"  RR90 NaN count: {stats['RR90_nan_count']}")
        print(f"  Normalized slope NaN count: {stats['NSlope_nan_count']}")
        print(f"  Average feature count: {stats['avg_feature_count']:.2f}")
    
    # 6. Save detailed results
    output_path = "ers_tolerance_validation_results.csv"
    results_df.to_csv(output_path, index=False)
    logging.info(f"Detailed results saved to: {output_path}")
    
    logging.info("===== Validation Completed =====")


if __name__ == "__main__":
    main()