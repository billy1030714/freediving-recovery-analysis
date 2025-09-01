# ers_tolerance_validation.py
# 驗證不同容忍區間對 ERS 特徵產出的影響
# 基於原始 features_02.py 的邏輯，測試 ±1, ±2, ±3 秒容忍度

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class ERSToleranceValidator:
    """ERS 容忍度驗證器"""
    
    def __init__(self):
        # 基於 features_02.py 的配置
        self.BASE_WINDOW = (330, 150)  # baseline window: -330s to -150s
        self.SLOPE_W1 = (30, 90)       # slope window 1: 30s to 90s
        self.SLOPE_W2 = (45, 105)      # slope window 2: 45s to 105s  
        self.PEAK_MAX_SECONDS = 120    # peak search window: 0s to 120s
        
    def _get_hr_with_tolerance(self, df: pd.DataFrame, target_time: pd.Timestamp, tol: int = 3) -> Optional[float]:
        """根據容忍度獲取指定時間點的心率"""
        df_win = df[(df["Time"] >= target_time - timedelta(seconds=tol)) &
                    (df["Time"] <= target_time + timedelta(seconds=tol))]
        if df_win.empty:
            return None
        idx = (df_win["Time"] - target_time).abs().idxmin()
        if pd.isna(idx):
            return None
        return df_win.loc[idx, "HR"]

    def _lin_slope_with_tolerance(self, df: pd.DataFrame, ts: pd.Timestamp, te: pd.Timestamp, tol: int = 3) -> Optional[float]:
        """根據容忍度計算線性斜率"""
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
        """將數值限制在 [0, 1] 區間"""
        return float(np.clip(v, 0.0, 1.0)) if v is not None and np.isfinite(v) else None

    def _get_dynamic_ideal_slope(self, starting_hr: float, resting_hr: float = 60.0, 
                                peak_hr: float = 130.0, max_ideal_slope: float = 1.0) -> float:
        """動態理想斜率計算"""
        if starting_hr <= resting_hr:
            return 0.1
        
        scale = (starting_hr - resting_hr) / (peak_hr - resting_hr)
        ideal_slope = max_ideal_slope * scale
        return max(0.1, ideal_slope)

    def analyze_event_with_tolerance(self, hr_df: pd.DataFrame, end_apnea_time: pd.Timestamp, tolerance: int = 3) -> Optional[Dict]:
        """使用指定容忍度分析單個呼吸中止事件"""
        if pd.isna(end_apnea_time):
            return None
            
        df = hr_df.copy().dropna(subset=["Time", "HR"]).sort_values("Time")
        if df.empty:
            return None
            
        event_t0 = end_apnea_time
        
        # 1. 計算 baseline
        base_df = df[(df["Time"] >= event_t0 - timedelta(seconds=self.BASE_WINDOW[0])) &
                     (df["Time"] < event_t0 - timedelta(seconds=self.BASE_WINDOW[1]))]
        baseline = base_df["HR"].mean() if not base_df.empty else None
        
        # 2. 計算 peak
        peak_search_df = df[(df["Time"] >= event_t0) &
                           (df["Time"] <= event_t0 + timedelta(seconds=self.PEAK_MAX_SECONDS))]
        if peak_search_df.empty or peak_search_df['HR'].isnull().all():
            return None
        peak = peak_search_df["HR"].max()
        if pd.isna(peak):
            return None

        # 3. 計算斜率（使用指定容忍度）
        time_peak_hr_timestamp = peak_search_df.loc[peak_search_df['HR'].idxmax(), 'Time']
        time_to_peak_hr = (time_peak_hr_timestamp - event_t0).total_seconds()
        slope_window_config = self.SLOPE_W1 if time_to_peak_hr <= 30 else self.SLOPE_W2
        ts_slope = event_t0 + timedelta(seconds=slope_window_config[0])
        te_slope = event_t0 + timedelta(seconds=slope_window_config[1])
        is_peak_abnormal = (time_to_peak_hr > 45) or (ts_slope <= time_peak_hr_timestamp <= te_slope)
        slope = np.nan if is_peak_abnormal else self._lin_slope_with_tolerance(df, ts_slope, te_slope, tol=tolerance)

        # 4. 計算恢復比率（使用指定容忍度）
        hr60 = self._get_hr_with_tolerance(df, event_t0 + timedelta(seconds=60), tol=tolerance)
        hr90 = self._get_hr_with_tolerance(df, event_t0 + timedelta(seconds=90), tol=tolerance)
        
        denom = peak - baseline if baseline is not None and peak > baseline else 0
        def _rr(hr_x):
            if denom == 0 or hr_x is None:
                return None
            return self._clip01((peak - hr_x) / denom)
        
        rr60, rr90 = _rr(hr60), _rr(hr90)
        hrr60 = (peak - hr60) if peak is not None and hr60 is not None else None

        # 5. 計算標準化斜率
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

        # 6. 計算 ERS
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
        """驗證不同容忍度對 ERS 特徵的影響"""
        results = []
        
        for i, end_apnea_time in enumerate(apnea_events):
            event_id = f"event_{i:04d}"
            base_result = {"event_id": event_id, "end_apnea_time": end_apnea_time}
            
            for tol in tolerances:
                analysis = self.analyze_event_with_tolerance(hr_df, end_apnea_time, tolerance=tol)
                
                if analysis is None:
                    # 如果分析失敗，記錄 NaN
                    for key in ["baseline", "peak", "hrr60", "recovery_ratio_60s", 
                               "recovery_ratio_90s", "normalized_slope", "ERS", "ers_feature_count"]:
                        base_result[f"{key}_tol{tol}"] = np.nan
                else:
                    # 記錄分析結果
                    for key, value in analysis.items():
                        if key != "tolerance":
                            base_result[f"{key}_tol{tol}"] = value
            
            results.append(base_result)
        
        return pd.DataFrame(results)

    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict:
        """生成摘要報告"""
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
    """載入樣本資料 - 基於 features_02.py 的格式"""
    
    # 載入 HR 資料
    hr_df = pd.read_csv(hr_csv_path)
    if 'Time' not in hr_df.columns or 'HR' not in hr_df.columns:
        raise ValueError("HR 檔案必須包含 'Time' 和 'HR' 欄位")
    hr_df['Time'] = pd.to_datetime(hr_df['Time'])
    
    # 載入呼吸中止事件資料
    apnea_df = pd.read_csv(apnea_csv_path)
    if 'end_apnea' not in apnea_df.columns:
        raise ValueError("Apnea 檔案必須包含 'end_apnea' 欄位")
    apnea_df['end_apnea'] = pd.to_datetime(apnea_df['end_apnea'])
    
    # 提取事件時間點
    apnea_events = apnea_df['end_apnea'].dropna().tolist()
    
    return hr_df, apnea_events


def main():
    """主執行函數"""
    logging.info("===== ERS 容忍度驗證開始 =====")
    
    # 1. 設定檔案路徑 - 請修改為你的實際檔案路徑
    hr_csv_path = "converted/hr_20250818.csv"  # HR 資料檔案，格式: Time, HR
    apnea_csv_path = "converted/apnea_events_20250818.csv"  # 呼吸中止事件檔案，格式: end_apnea, row_id
    
    # 或者你可以使用目錄掃描方式（如果你有多天的資料）
    # converted_dir = Path("converted")  # 包含 hr_*.csv 和 apnea_events_*.csv 的目錄
    
    try:
        hr_df, apnea_events = load_sample_data(hr_csv_path, apnea_csv_path)
        logging.info(f"成功載入資料：{len(hr_df)} 筆心率記錄，{len(apnea_events)} 個呼吸中止事件")
    except FileNotFoundError as e:
        logging.error(f"找不到檔案：{e}")
        logging.info("請確認以下檔案存在：")
        logging.info(f"  HR 檔案: {hr_csv_path} (欄位: Time, HR)")
        logging.info(f"  事件檔案: {apnea_csv_path} (欄位: end_apnea, row_id)")
        return
    except Exception as e:
        logging.error(f"載入資料時發生錯誤：{e}")
        return
    
    # 2. 初始化驗證器
    validator = ERSToleranceValidator()
    
    # 3. 執行驗證
    logging.info("開始分析不同容忍度的影響...")
    results_df = validator.validate_tolerance_effects(hr_df, apnea_events, tolerances=[1, 2, 3])
    
    # 4. 生成報告
    summary = validator.generate_summary_report(results_df)
    
    # 5. 顯示結果
    print("\n" + "="*60)
    print("ERS 容忍度驗證結果摘要")
    print("="*60)
    print(f"總事件數: {summary['total_events']}")
    print("\n各容忍度設定的成功率:")
    
    for tolerance, stats in summary['tolerance_comparison'].items():
        print(f"\n{tolerance} 容忍度:")
        print(f"  ERS 成功率: {stats['ERS_success_rate']}")
        print(f"  ERS NaN 數量: {stats['ERS_nan_count']}")
        print(f"  RR60 NaN 數量: {stats['RR60_nan_count']}")
        print(f"  RR90 NaN 數量: {stats['RR90_nan_count']}")
        print(f"  標準化斜率 NaN 數量: {stats['NSlope_nan_count']}")
        print(f"  平均特徵數量: {stats['avg_feature_count']:.2f}")
    
    # 6. 儲存詳細結果
    output_path = "ers_tolerance_validation_results.csv"
    results_df.to_csv(output_path, index=False)
    logging.info(f"詳細結果已儲存至: {output_path}")
    
    logging.info("===== 驗證完成 =====")


if __name__ == "__main__":
    main()