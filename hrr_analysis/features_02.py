"""
features_02.py – HRR 特徵工程腳本 (最終完整版)

本腳本是數據處理流程的第二步，負責計算與恢復相關的核心特徵，
並擴充了時間序列相關的上下文特徵與品質分數。
"""
# --- 導入函式庫 ---
import logging, re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np, pandas as pd

# --- 本地模組 ---
from paths import DIR_CONVERTED, DIR_FEATURES, get_daily_path

# --- 常數與設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

@dataclass(frozen=True)
class FeatureConfig:
    """特徵計算的參數設定"""
    BASE_WINDOW: Tuple[int, int] = (330, 150)
    SLOPE_W1: Tuple[int, int] = (30, 90)
    SLOPE_W2: Tuple[int, int] = (45, 105)
    PEAK_MAX_SECONDS: int = 120
    HRV_WINDOW: Tuple[int, int] = (180, 360)
    FINAL_COLUMNS: List[str] = field(default_factory=lambda: [
        "row_id", "date", "end_apnea_time", "HRbaseline", "HRpeak",
        "recovery_ratio_60s", "recovery_ratio_90s", "normalized_slope", "ERS",
        "personal_baseline_28d", "baseline_diff", "rmssd_post",
        "time_since_last_apnea", "baseline_quality_score"
    ])

CONFIG = FeatureConfig()
DATE_RE = re.compile(r"(\d{8})")

# --- 資料結構定義 ---
@dataclass(frozen=True)
class DayFiles:
    date_str: str; hr_path: Path; apnea_path: Path

@dataclass(frozen=True)
class LoadedDay:
    date_str: str; hr_df: pd.DataFrame; apnea_df: pd.DataFrame

# --- 核心函數 ---
def _get_hr_at(df: pd.DataFrame, t, mode: str = 'linear') -> Optional[float]:
    """使用線性插值獲取特定時間點的 HR。"""
    s = df[['Time', 'HR']].dropna().sort_values('Time')
    if s.empty: return None
    t = pd.to_datetime(t)
    exact = s[s['Time'] == t]
    if not exact.empty: return float(exact['HR'].iloc[0])
    before = s[s['Time'] <= t].tail(1); after = s[s['Time'] >= t].head(1)
    if mode == "linear" and not before.empty and not after.empty and before.index[0] != after.index[0]:
        t0, h0 = before['Time'].iloc[0], float(before['HR'].iloc[0])
        t1, h1 = after['Time'].iloc[0], float(after['HR'].iloc[0])
        dt = (t1 - t0).total_seconds()
        if dt > 0: return float(h0 + (t - t0).total_seconds() / dt * (h1 - h0))
    idx = (s['Time'] - t).abs().idxmin()
    return float(s.loc[idx, 'HR']) if idx in s.index else None

def _lin_slope(df: pd.DataFrame, t0, ts, te):
    """計算指定時間窗內的線性回歸斜率。"""
    seg = df[(df["Time"] >= ts) & (df["Time"] <= te)][["Time", "HR"]].dropna()
    if len(seg) < 2: return None
    x = (seg["Time"] - t0).dt.total_seconds().to_numpy(float)
    y = pd.to_numeric(seg["HR"], errors="coerce").to_numpy(float)
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else None

def _clip01(v): 
    """將數值裁剪至 [0, 1] 區間。"""
    return float(np.clip(v, 0.0, 1.0)) if v is not None and np.isfinite(v) else None

def analyze_event(hr_df: pd.DataFrame, end_apnea_time: pd.Timestamp) -> Optional[Dict]:
    """
    對單一事件進行完整的特徵計算 (v3.0 最終版邏輯)。
    - 新增: 基於 HR Peak 時間的滑動窗口斜率計算。
    - 新增: 對異常 HR Peak 的數據進行標記與清理。
    """
    if pd.isna(end_apnea_time):
        return None
    df = hr_df.copy().dropna(subset=["Time", "HR"]).sort_values("Time")
    if df.empty:
        return None

    # --- 1. 基礎指標計算 ---
    t0 = end_apnea_time
    base_df = df[(df["Time"] >= t0 - timedelta(seconds=CONFIG.BASE_WINDOW[0])) & (df["Time"] < t0 - timedelta(seconds=CONFIG.BASE_WINDOW[1]))]
    baseline = base_df["HR"].mean() if not base_df.empty else None

    peak_search_df = df[(df["Time"] >= t0) & (df["Time"] <= t0 + timedelta(seconds=CONFIG.PEAK_MAX_SECONDS))]
    peak = peak_search_df["HR"].max() if not peak_search_df.empty else None

    if baseline is None or peak is None:
        return None

    # --- 2. 【V3.0 新邏輯】基於 HR Peak 時間的滑動窗口 ---
    time_peak_hr_timestamp = peak_search_df.loc[peak_search_df['HR'].idxmax(), 'Time'] if not peak_search_df.empty else t0
    time_to_peak_hr = (time_peak_hr_timestamp - t0).total_seconds()
    
    # 根據 peak 出現時間選擇擬合窗口
    if time_to_peak_hr <= 30:
        # 正常恢復模式
        slope_window_config = CONFIG.SLOPE_W1
    else: # 包含了 30 < peak <= 45 的情況 (後續會檢查 >45)
        # 延遲達峰模式
        slope_window_config = CONFIG.SLOPE_W2

    ts_slope = t0 + timedelta(seconds=slope_window_config[0])
    te_slope = t0 + timedelta(seconds=slope_window_config[1])
    
    # 最終有效性檢查
    is_peak_abnormal = (time_to_peak_hr > 45) or \
                       (ts_slope <= time_peak_hr_timestamp <= te_slope)

    if is_peak_abnormal:
        slope = np.nan # 標記為無效斜率
    else:
        slope = _lin_slope(df, t0, ts_slope, te_slope)

    # --- 3. 計算恢復率與 ERS 分數 ---
    hr60 = _get_hr_at(df, t0 + timedelta(seconds=60))
    hr90 = _get_hr_at(df, t0 + timedelta(seconds=90))
    denom = peak - baseline
    
    def _rr(hr_x):
        if denom == 0 or hr_x is None: return None
        return _clip01((peak - hr_x) / denom)
    
    rr60, rr90 = _rr(hr60), _rr(hr90)
    normalized_slope = _clip01(-slope / denom) if slope is not None and np.isfinite(slope) and denom != 0 else None
    
    # 這裡我們將 ECS 更名為 ERS (Early Recovery Score)
    valid_metrics = [v for v in (rr60, rr90, normalized_slope) if v is not None]
    ers = np.mean(valid_metrics) if valid_metrics else None

    # --- 4. 計算 HRV 指標 ---
    hrv_df = df[(df["Time"] >= t0 + timedelta(seconds=CONFIG.HRV_WINDOW[0])) & (df["Time"] <= t0 + timedelta(seconds=CONFIG.HRV_WINDOW[1]))]
    rmssd = None
    if len(hrv_df) >= 10:
        rr_intervals = 60000.0 / hrv_df["HR"]
        diffs = np.diff(rr_intervals)
        if len(diffs) > 0:
            rmssd = np.sqrt(np.mean(diffs**2))

    return {
        "end_apnea_time": t0,
        "HRbaseline": baseline,
        "HRpeak": peak,
        "HR_peak_time_post_apnea": time_to_peak_hr, # 保留此特徵
        "rmssd_post": rmssd,
        "recovery_ratio_60s": rr60,
        "recovery_ratio_90s": rr90,
        "normalized_slope": normalized_slope,
        "ERS": ers # 將 ECS 正式更名為 ERS
    }

def discover_day_files(converted_dir: Path) -> List[DayFiles]:
    """掃描 `converted` 資料夾，尋找成對的 hr 與 apnea_events 檔案。"""
    day_files = []
    for hr_path in sorted(converted_dir.glob("hr_*.csv")):
        if match := DATE_RE.search(hr_path.name):
            date_str = match.group(1)
            apnea_path = get_daily_path(directory=converted_dir, data_type='apnea_events', date_obj=pd.to_datetime(date_str).date(), extension='.csv')
            if apnea_path.exists(): day_files.append(DayFiles(date_str, hr_path, apnea_path))
    logging.info(f"成功發現 {len(day_files)} 個成對的每日數據檔案。")
    return day_files

def load_day(day: DayFiles) -> LoadedDay:
    """從 DayFiles 路徑載入單日的 HR 與 Apnea 數據到 DataFrame。"""
    hr_df = pd.read_csv(day.hr_path, parse_dates=["Time"])
    apnea_df = pd.read_csv(day.apnea_path, parse_dates=["end_apnea"])
    if "row_id" not in apnea_df.columns and "end_apnea" in apnea_df.columns: apnea_df["row_id"] = apnea_df["end_apnea"].dt.strftime("%Y%m%d_%H%M%S")
    apnea_df["row_id"] = apnea_df["row_id"].astype(str)
    return LoadedDay(day.date_str, hr_df, apnea_df)

def process_day(day_data: LoadedDay) -> List[Dict]:
    """遍歷單日的所有 apnea 事件，並為每個事件調用 analyze_event。"""
    new_features = []
    for _, event in day_data.apnea_df.iterrows():
        features = analyze_event(day_data.hr_df, event["end_apnea"])
        if features:
            features["row_id"] = event["row_id"]; features["date"] = day_data.date_str
            new_features.append(features)
    return new_features

def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """計算需要全局上下文的特徵，例如滾動指標和時間差。"""
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
    """將新計算的特徵與歷史特徵合併、去重、後處理並儲存。"""
    new_df = pd.DataFrame(new_features)
    features_csv_path = features_dir / "features.csv"
    if features_csv_path.exists():
        old_df = pd.read_csv(features_csv_path)
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    if combined_df.empty: return combined_df

    combined_df["end_apnea_time"] = pd.to_datetime(combined_df["end_apnea_time"])
    final_df = combined_df.sort_values("end_apnea_time").drop_duplicates(subset=["row_id"], keep="last")
    final_df = add_contextual_features(final_df)
    
    final_df['end_apnea_time'] = final_df['end_apnea_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    final_cols = [col for col in CONFIG.FINAL_COLUMNS if col in final_df.columns]
    final_df = final_df[final_cols]
    
    features_dir.mkdir(exist_ok=True, parents=True)
    final_df.to_csv(features_csv_path, index=False)
    
    try:
        parquet_path = features_dir / "features_ml.parquet"
        final_df.to_parquet(parquet_path, index=False)
        logging.info(f"成功儲存 Parquet 檔案: {parquet_path}")
    except Exception as e:
        logging.warning(f"無法儲存為 Parquet 格式: {e}")
        
    return final_df

def main():
    """主執行函數。"""
    logging.info("===== 開始執行 02_features 特徵工程腳本 (最終完整版) =====")
    day_files = discover_day_files(DIR_CONVERTED)
    if not day_files:
        logging.warning("在 `converted` 資料夾中找不到任何成對的數據檔案，腳本終止。")
        return

    all_new_features = []
    for day in day_files:
        try:
            day_data = load_day(day)
            features_from_day = process_day(day_data)
            all_new_features.extend(features_from_day)
        except Exception as e:
            logging.error(f"處理日期 {day.date_str} 時發生錯誤: {e}", exc_info=True)

    final_df = merge_and_finalize(all_new_features, DIR_FEATURES)
    logging.info(f"===== 腳本執行完畢 =====")
    logging.info(f"最終特徵檔儲存於: {DIR_FEATURES / 'features.csv'} (共 {len(final_df)} 筆紀錄)")

if __name__ == "__main__":
    main()