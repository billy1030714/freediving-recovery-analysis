"""
tolerance_validation.py - Tolerance Window Validation Script

Validates the robustness of feature engineering under different temporal tolerance settings.
Tests whether the ±3 second tolerance introduces systematic bias or affects model performance.

Usage:
    python tolerance_validation.py
"""

import logging
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from paths import DIR_CONVERTED, DIR_FEATURES

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DATE_RE = re.compile(r"(\d{8})")

# --- Configuration ---
TOLERANCE_LEVELS = [1, 2, 3]  # Test ±1s, ±2s, ±3s
OUTPUT_DIR = DIR_FEATURES.parent / "validation_output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --- Data Structures ---
@dataclass(frozen=True)
class DayFiles:
    date_str: str
    hr_path: Path
    apnea_path: Path

@dataclass
class ToleranceMetrics:
    tolerance: int
    success_rate: float
    avg_feature_count: float
    total_nan: int
    completeness: float
    feature_stability: Dict[str, float]  # Feature-wise variance

# --- Core Functions ---
def discover_day_files(converted_dir: Path) -> List[DayFiles]:
    """Discover paired HR and apnea event files."""
    day_files = []
    for hr_path in sorted(converted_dir.glob("hr_*.csv")):
        if match := DATE_RE.search(hr_path.name):
            date_str = match.group(1)
            apnea_path = converted_dir / f'apnea_events_{date_str}.csv'
            if apnea_path.exists():
                day_files.append(DayFiles(date_str, hr_path, apnea_path))
    logging.info(f"Discovered {len(day_files)} paired data files.")
    return day_files

def _get_hr_with_tolerance(df: pd.DataFrame, target_time: pd.Timestamp, tol: int) -> Optional[float]:
    """Get HR value with specified tolerance."""
    df_win = df[
        (df["Time"] >= target_time - timedelta(seconds=tol)) & 
        (df["Time"] <= target_time + timedelta(seconds=tol))
    ]
    if df_win.empty:
        return None
    idx = (df_win["Time"] - target_time).abs().idxmin()
    return None if pd.isna(idx) else df_win.loc[idx, "HR"]

def _lin_slope_with_tolerance(df: pd.DataFrame, ts: pd.Timestamp, te: pd.Timestamp, tol: int) -> Optional[float]:
    """Calculate linear slope with specified tolerance."""
    win_df = df[
        (df["Time"] >= ts - timedelta(seconds=tol)) & 
        (df["Time"] <= te + timedelta(seconds=tol))
    ]
    if len(win_df) < 2:
        return None
    x = (win_df["Time"] - win_df["Time"].iloc[0]).dt.total_seconds().to_numpy()
    y = win_df["HR"].to_numpy()
    if len(x) != len(y) or len(x) == 0:
        return None
    return np.polyfit(x, y, 1)[0]

def _clip01(v):
    return float(np.clip(v, 0.0, 1.0)) if v is not None and np.isfinite(v) else None

def _get_dynamic_ideal_slope(starting_hr: float, resting_hr: float = 60.0, 
                             peak_hr: float = 130.0, max_ideal_slope: float = 1.0) -> float:
    """Calculate dynamic ideal slope based on HR range."""
    if starting_hr <= resting_hr:
        return 0.1
    scale = (starting_hr - resting_hr) / (peak_hr - resting_hr)
    ideal_slope = max_ideal_slope * scale
    return max(0.1, ideal_slope)

def analyze_event_with_tolerance(hr_df: pd.DataFrame, end_apnea_time: pd.Timestamp, 
                                 tolerance: int, config: Dict) -> Optional[Dict]:
    """
    Analyze single event with specified tolerance.
    
    This is a modified version of analyze_event() from 02_features.py
    that allows dynamic tolerance adjustment.
    """
    if pd.isna(end_apnea_time):
        return None
    
    df = hr_df.copy().dropna(subset=["Time", "HR"]).sort_values("Time")
    if df.empty:
        return None
    
    event_t0 = end_apnea_time
    
    # Baseline calculation (use config from params.yaml)
    base_df = df[
        (df["Time"] >= event_t0 - timedelta(seconds=config['base_window'][0])) &
        (df["Time"] < event_t0 - timedelta(seconds=config['base_window'][1]))
    ]
    baseline = base_df["HR"].mean() if not base_df.empty else None
    
    # Peak detection (use config from params.yaml)
    peak_search_df = df[
        (df["Time"] >= event_t0) &
        (df["Time"] <= event_t0 + timedelta(seconds=config['peak_max_seconds']))
    ]
    if peak_search_df.empty or peak_search_df['HR'].isnull().all():
        return None
    peak = peak_search_df["HR"].max()
    if pd.isna(peak):
        return None
    
    time_peak_hr_timestamp = peak_search_df.loc[peak_search_df['HR'].idxmax(), 'Time']
    time_to_peak_hr = (time_peak_hr_timestamp - event_t0).total_seconds()
    
    # Slope calculation (use config from params.yaml)
    slope_window = config['slope_w1'] if time_to_peak_hr <= 30 else config['slope_w2']
    ts_slope = event_t0 + timedelta(seconds=slope_window[0])
    te_slope = event_t0 + timedelta(seconds=slope_window[1])
    is_peak_abnormal = (time_to_peak_hr > 45) or (ts_slope <= time_peak_hr_timestamp <= te_slope)
    slope = np.nan if is_peak_abnormal else _lin_slope_with_tolerance(df, ts_slope, te_slope, tol=tolerance)
    
    # Recovery ratios with tolerance
    hr60 = _get_hr_with_tolerance(df, event_t0 + timedelta(seconds=60), tolerance)
    hr90 = _get_hr_with_tolerance(df, event_t0 + timedelta(seconds=90), tolerance)
    
    denom = peak - baseline if baseline is not None and peak > baseline else 0
    def _rr(hr_x):
        return _clip01((peak - hr_x) / denom) if denom > 0 and hr_x is not None else None
    
    rr60 = _rr(hr60)
    rr90 = _rr(hr90)
    hrr60 = (peak - hr60) if peak is not None and hr60 is not None else None
    
    # Normalized slope
    normalized_slope = None
    if slope is not None and np.isfinite(slope):
        hr_slope_start = _get_hr_with_tolerance(df, ts_slope, tolerance)
        if hr_slope_start and baseline and peak:
            dynamic_ideal_slope = _get_dynamic_ideal_slope(hr_slope_start, baseline, peak)
            normalized_slope = _clip01(abs(slope) / dynamic_ideal_slope)
    
    # ERS calculation
    valid_metrics = [v for v in (rr60, rr90, normalized_slope) if v is not None]
    ers = np.mean(valid_metrics) if valid_metrics else None
    
    return {
        "end_apnea_time": event_t0,
        "HRbaseline": baseline,
        "HRpeak": peak,
        "hrr60": hrr60,
        "recovery_ratio_60s": rr60,
        "recovery_ratio_90s": rr90,
        "normalized_slope": normalized_slope,
        "ERS": ers,
        "ers_feature_count": len(valid_metrics),
        "tolerance": tolerance
    }

def process_with_tolerance(day_files: List[DayFiles], tolerance: int, config: Dict) -> pd.DataFrame:
    """Process all events with specified tolerance level."""
    all_features = []
    
    for day in day_files:
        hr_df = pd.read_csv(day.hr_path, parse_dates=["Time"])
        apnea_df = pd.read_csv(day.apnea_path, parse_dates=["end_apnea"])
        
        # Generate row_id if not present (same logic as 02_features.py)
        if "row_id" not in apnea_df.columns and "end_apnea" in apnea_df.columns:
            apnea_df["row_id"] = apnea_df["end_apnea"].dt.strftime("%Y%m%d_%H%M%S")
        apnea_df["row_id"] = apnea_df["row_id"].astype(str)
        
        for _, event in apnea_df.iterrows():
            features = analyze_event_with_tolerance(hr_df, event["end_apnea"], tolerance, config)
            if features:
                features["date"] = day.date_str
                features["row_id"] = event["row_id"]  # Add row_id for deduplication
                all_features.append(features)
    
    df = pd.DataFrame(all_features)
    
    # CRITICAL: Deduplicate by row_id (same as 02_features.py)
    if not df.empty and "row_id" in df.columns:
        original_count = len(df)
        df = df.sort_values("end_apnea_time").drop_duplicates(subset=["row_id"], keep="last")
        dedup_count = len(df)
        if original_count != dedup_count:
            logging.info(f"   Deduplicated: {original_count} → {dedup_count} events (removed {original_count - dedup_count} duplicates)")
    
    return df

def calculate_metrics(df: pd.DataFrame, tolerance: int) -> ToleranceMetrics:
    """Calculate comprehensive metrics for a tolerance level."""
    total_events = len(df)
    success_rate = (df['ERS'].notna().sum() / total_events * 100) if total_events > 0 else 0
    avg_feature_count = df['ers_feature_count'].mean()
    
    # Count total NaN across key features
    key_features = ['recovery_ratio_60s', 'recovery_ratio_90s', 'normalized_slope']
    total_nan = df[key_features].isna().sum().sum()
    completeness = (1 - total_nan / (len(df) * len(key_features))) * 100
    
    # Feature stability (coefficient of variation)
    feature_stability = {}
    for feat in key_features:
        valid_values = df[feat].dropna()
        if len(valid_values) > 1:
            cv = (valid_values.std() / valid_values.mean()) * 100 if valid_values.mean() != 0 else np.nan
            feature_stability[feat] = cv
        else:
            feature_stability[feat] = np.nan
    
    return ToleranceMetrics(
        tolerance=tolerance,
        success_rate=success_rate,
        avg_feature_count=avg_feature_count,
        total_nan=int(total_nan),
        completeness=completeness,
        feature_stability=feature_stability
    )

def test_systematic_bias(results_dict: Dict[int, pd.DataFrame]) -> Dict[str, any]:
    """
    Test for systematic bias across tolerance levels.
    
    Tests:
    1. Mean value shifts (paired t-test)
    2. Distribution changes (KS test)
    3. Correlation stability
    """
    bias_tests = {}
    
    # Compare ±1s vs ±3s
    df_1s = results_dict[1]
    df_3s = results_dict[3]
    
    # Merge on common events
    merged = df_1s.merge(df_3s, on='end_apnea_time', suffixes=('_1s', '_3s'))
    
    key_features = ['recovery_ratio_60s', 'recovery_ratio_90s', 'normalized_slope', 'ERS']
    
    for feat in key_features:
        feat_1s = f"{feat}_1s"
        feat_3s = f"{feat}_3s"
        
        valid_pairs = merged[[feat_1s, feat_3s]].dropna()
        
        if len(valid_pairs) > 5:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(valid_pairs[feat_1s], valid_pairs[feat_3s])
            
            # KS test
            ks_stat, ks_p = stats.ks_2samp(valid_pairs[feat_1s], valid_pairs[feat_3s])
            
            # Mean difference
            mean_diff = valid_pairs[feat_1s].mean() - valid_pairs[feat_3s].mean()
            
            # Correlation
            corr = valid_pairs[feat_1s].corr(valid_pairs[feat_3s])
            
            bias_tests[feat] = {
                'mean_diff': mean_diff,
                'mean_diff_pct': (mean_diff / valid_pairs[feat_3s].mean() * 100) if valid_pairs[feat_3s].mean() != 0 else np.nan,
                't_test_p': p_value,
                'ks_test_p': ks_p,
                'correlation': corr,
                'n_pairs': len(valid_pairs)
            }
        else:
            bias_tests[feat] = {'insufficient_data': True}
    
    return bias_tests

def visualize_results(results_dict: Dict[int, pd.DataFrame], metrics_list: List[ToleranceMetrics]):
    """Create comprehensive visualization of tolerance validation."""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Success Rate & Completeness
    ax1 = plt.subplot(3, 3, 1)
    tolerances = [m.tolerance for m in metrics_list]
    success_rates = [m.success_rate for m in metrics_list]
    completeness = [m.completeness for m in metrics_list]
    
    ax1.plot(tolerances, success_rates, 'o-', label='ERS Success Rate', linewidth=2, markersize=8)
    ax1.plot(tolerances, completeness, 's-', label='Feature Completeness', linewidth=2, markersize=8)
    ax1.set_xlabel('Tolerance (seconds)', fontsize=11)
    ax1.set_ylabel('Percentage (%)', fontsize=11)
    ax1.set_title('A. Data Completeness vs Tolerance', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(tolerances)
    
    # 2. Average Feature Count
    ax2 = plt.subplot(3, 3, 2)
    avg_counts = [m.avg_feature_count for m in metrics_list]
    ax2.bar(tolerances, avg_counts, color=['#d62728', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax2.set_xlabel('Tolerance (seconds)', fontsize=11)
    ax2.set_ylabel('Average Count', fontsize=11)
    ax2.set_title('B. Average ERS Component Count', fontweight='bold')
    ax2.set_xticks(tolerances)
    ax2.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Maximum (3)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Total NaN Count
    ax3 = plt.subplot(3, 3, 3)
    nan_counts = [m.total_nan for m in metrics_list]
    ax3.bar(tolerances, nan_counts, color=['#d62728', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax3.set_xlabel('Tolerance (seconds)', fontsize=11)
    ax3.set_ylabel('Total NaN Count', fontsize=11)
    ax3.set_title('C. Missing Data by Tolerance', fontweight='bold')
    ax3.set_xticks(tolerances)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4-6. Distribution comparisons for key features
    features_to_plot = ['recovery_ratio_60s', 'recovery_ratio_90s', 'ERS']
    feature_labels = ['Recovery Ratio (60s)', 'Recovery Ratio (90s)', 'ERS']
    
    for idx, (feat, label) in enumerate(zip(features_to_plot, feature_labels), start=4):
        ax = plt.subplot(3, 3, idx)
        for tol in TOLERANCE_LEVELS:
            df = results_dict[tol]
            valid_data = df[feat].dropna()
            if len(valid_data) > 0:
                ax.hist(valid_data, bins=20, alpha=0.5, label=f'±{tol}s', density=True)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{chr(67+idx-3)}. {label} Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 7-9. Bland-Altman style comparison (±1s vs ±3s)
    df_1s = results_dict[1]
    df_3s = results_dict[3]
    merged = df_1s.merge(df_3s, on='end_apnea_time', suffixes=('_1s', '_3s'))
    
    for idx, (feat, label) in enumerate(zip(features_to_plot, feature_labels), start=7):
        ax = plt.subplot(3, 3, idx)
        feat_1s = f"{feat}_1s"
        feat_3s = f"{feat}_3s"
        valid = merged[[feat_1s, feat_3s]].dropna()
        
        if len(valid) > 0:
            mean_vals = (valid[feat_1s] + valid[feat_3s]) / 2
            diff_vals = valid[feat_1s] - valid[feat_3s]
            
            ax.scatter(mean_vals, diff_vals, alpha=0.6, s=30)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.axhline(y=diff_vals.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean diff: {diff_vals.mean():.3f}')
            ax.axhline(y=diff_vals.mean() + 1.96*diff_vals.std(), color='red', linestyle=':', linewidth=1.5)
            ax.axhline(y=diff_vals.mean() - 1.96*diff_vals.std(), color='red', linestyle=':', linewidth=1.5)
            ax.set_xlabel(f'Mean {label}', fontsize=11)
            ax.set_ylabel('Difference (±1s - ±3s)', fontsize=11)
            ax.set_title(f'{chr(67+idx-3)}. Bland-Altman: {label}', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tolerance_validation.png', dpi=300, bbox_inches='tight')
    logging.info(f"Visualization saved to: {OUTPUT_DIR / 'tolerance_validation.png'}")
    plt.close()

def generate_report(metrics_list: List[ToleranceMetrics], bias_tests: Dict[str, any]):
    """Generate comprehensive validation report."""
    report_path = OUTPUT_DIR / 'tolerance_validation_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TOLERANCE WINDOW VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary Table
        f.write("1. SUMMARY METRICS BY TOLERANCE LEVEL\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Tolerance':<12} {'Success Rate':<15} {'Avg Features':<15} {'Total NaN':<12} {'Completeness':<12}\n")
        f.write("-" * 80 + "\n")
        
        for m in metrics_list:
            f.write(f"±{m.tolerance} sec{'':<6} {m.success_rate:>6.1f}%{'':<8} "
                   f"{m.avg_feature_count:>6.2f}{'':<9} {m.total_nan:>6}{'':<6} "
                   f"{m.completeness:>6.1f}%\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Feature Stability
        f.write("2. FEATURE STABILITY (Coefficient of Variation, %)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Tolerance':<12} {'RR60':<20} {'RR90':<20} {'Norm Slope':<20}\n")
        f.write("-" * 80 + "\n")
        
        for m in metrics_list:
            rr60_cv = m.feature_stability.get('recovery_ratio_60s', np.nan)
            rr90_cv = m.feature_stability.get('recovery_ratio_90s', np.nan)
            slope_cv = m.feature_stability.get('normalized_slope', np.nan)
            
            f.write(f"±{m.tolerance} sec{'':<6} {rr60_cv:>6.2f}%{'':<13} "
                   f"{rr90_cv:>6.2f}%{'':<13} {slope_cv:>6.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Systematic Bias Tests
        f.write("3. SYSTEMATIC BIAS TESTS (±1s vs ±3s)\n")
        f.write("-" * 80 + "\n")
        
        for feat, results in bias_tests.items():
            if 'insufficient_data' in results:
                f.write(f"\n{feat}: Insufficient paired data for testing\n")
                continue
            
            f.write(f"\n{feat}:\n")
            f.write(f"  Mean Difference: {results['mean_diff']:.4f} ({results['mean_diff_pct']:.2f}%)\n")
            f.write(f"  Paired t-test p-value: {results['t_test_p']:.4f} {'✓ No bias' if results['t_test_p'] > 0.05 else '⚠ Potential bias'}\n")
            f.write(f"  KS test p-value: {results['ks_test_p']:.4f} {'✓ Same distribution' if results['ks_test_p'] > 0.05 else '⚠ Different distribution'}\n")
            f.write(f"  Correlation: {results['correlation']:.4f}\n")
            f.write(f"  Sample size: {results['n_pairs']} paired events\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Recommendations
        f.write("4. RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n\n")
        
        # Check if ±3s introduces bias
        ers_bias = bias_tests.get('ERS', {})
        has_significant_bias = ers_bias.get('t_test_p', 1.0) < 0.05
        
        if has_significant_bias:
            f.write("⚠ WARNING: ±3s tolerance shows statistically significant bias compared to ±1s\n")
            f.write(f"  Mean difference: {ers_bias['mean_diff']:.4f} ({ers_bias['mean_diff_pct']:.2f}%)\n")
            f.write("  → Recommendation: Use ±2s or conduct sensitivity analysis\n\n")
        else:
            f.write("✓ VALIDATED: ±3s tolerance does not introduce significant systematic bias\n")
            f.write(f"  Mean difference: {ers_bias.get('mean_diff', 0):.4f} ({ers_bias.get('mean_diff_pct', 0):.2f}%)\n")
            f.write(f"  Correlation: {ers_bias.get('correlation', 0):.4f}\n")
            f.write("  → Recommendation: ±3s is acceptable for maximizing sample size\n\n")
        
        # Data completeness analysis
        completeness_3s = metrics_list[2].completeness
        completeness_1s = metrics_list[0].completeness
        
        f.write(f"Data Completeness Improvement:\n")
        f.write(f"  ±1s: {completeness_1s:.1f}%\n")
        f.write(f"  ±3s: {completeness_3s:.1f}%\n")
        f.write(f"  Gain: {completeness_3s - completeness_1s:.1f} percentage points\n\n")
        
        if completeness_3s >= 90 and not has_significant_bias:
            f.write("✓ CONCLUSION: ±3s tolerance is optimal for this dataset\n")
            f.write("  - Achieves >90% completeness\n")
            f.write("  - No significant systematic bias\n")
            f.write("  - Maximizes usable sample size\n")
        else:
            f.write("⚠ CONCLUSION: Consider alternative tolerance settings\n")
            f.write("  - Review bias test results carefully\n")
            f.write("  - Consider using ±2s as compromise\n")
            f.write("  - Report tolerance setting as limitation\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logging.info(f"Report saved to: {report_path}")

def main():
    """Main execution function."""
    logging.info("=" * 80)
    logging.info("Starting Tolerance Window Validation")
    logging.info("=" * 80)
    
    # Load params.yaml (same as 02_features.py)
    import yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    config = params['feature_engineering']
    logging.info(f"\nUsing configuration from params.yaml:")
    logging.info(f"  Baseline window: {config['base_window']}")
    logging.info(f"  Peak search: {config['peak_max_seconds']}s")
    logging.info(f"  Slope windows: {config.get('slope_w1', 'default')} / {config.get('slope_w2', 'default')}")
    
    # Discover data files
    day_files = discover_day_files(DIR_CONVERTED)
    if not day_files:
        logging.error("No data files found. Exiting.")
        return
    
    # Process with different tolerance levels
    results_dict = {}
    metrics_list = []
    
    for tol in TOLERANCE_LEVELS:
        logging.info(f"\nProcessing with tolerance: ±{tol} seconds")
        df = process_with_tolerance(day_files, tol, config)  # Pass config
        results_dict[tol] = df
        
        metrics = calculate_metrics(df, tol)
        metrics_list.append(metrics)
        
        logging.info(f"  Success Rate: {metrics.success_rate:.1f}%")
        logging.info(f"  Avg Features: {metrics.avg_feature_count:.2f}")
        logging.info(f"  Completeness: {metrics.completeness:.1f}%")
    
    # Test for systematic bias
    logging.info("\nTesting for systematic bias...")
    bias_tests = test_systematic_bias(results_dict)
    
    # Generate visualizations
    logging.info("\nGenerating visualizations...")
    visualize_results(results_dict, metrics_list)
    
    # Generate report
    logging.info("\nGenerating validation report...")
    generate_report(metrics_list, bias_tests)
    
    # Save detailed results
    for tol, df in results_dict.items():
        output_file = OUTPUT_DIR / f"features_tolerance_{tol}s.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"Saved detailed results: {output_file}")
    
    logging.info("\n" + "=" * 80)
    logging.info("Validation Complete!")
    logging.info(f"Results saved to: {OUTPUT_DIR}")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()