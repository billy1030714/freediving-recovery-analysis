"""
cleaning_01.py – Apple Health HR & Apnea Event Parser

功能:
1.  解析來自 Apple Health 的 export.xml，提取心率資料。
2.  對心率資料進行清洗、過濾、插值，並儲存為當日的 hr_{YYYYMMDD}.csv。
3.  掃描 `converted` 資料夾中所有歷史的 apnea_events_*.csv。
4.  透過互動式介面，讓使用者輸入當日新增的呼吸中止事件。
5.  合併歷史與新增事件，去重後，儲存為包含「累積全量」的 apnea_events_{YYYYMMDD}.csv。
"""

# --- 導入函式庫 ---
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.impute import KNNImputer

# --- 本地模組 (我們的路徑設定檔) ---
from paths import (
    DIR_CONVERTED,
    FILE_APPLE_HEALTH_XML,
    get_daily_path
)

# --- 常數設定 ---
HR_MIN, HR_MAX = 30, 200 # 有效心率的最小與最大值

# --- 函數定義 ---

def parse_hr_from_xml(xml_path: Path, today_date: datetime.date) -> None:
    """從 export.xml 解析心率數據，清理後儲存為 CSV。"""
    if not xml_path.exists():
        raise FileNotFoundError(f"Apple Health 的 export.xml 不存在，請檢查路徑: {xml_path}")

    rows = []
    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag == "Record" and elem.get("type") == "HKQuantityTypeIdentifierHeartRate":
            t, v = elem.get("startDate"), elem.get("value")
            if t and v:
                try: rows.append({"Time": t, "HR": float(v)})
                except (ValueError, TypeError): pass
        elem.clear()

    if not rows: raise ValueError("在 export.xml 中找不到任何心率 (HeartRate) 記錄。")
    df = pd.DataFrame(rows)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    df = df[(df["HR"] >= HR_MIN) & (df["HR"] <= HR_MAX)]
    if df["HR"].isna().any():
        df["HR"] = KNNImputer(n_neighbors=3).fit_transform(df[["HR"]])

    # --- 【修正】使用正確的變數名稱 DIR_CONVERTED ---
    out_hr_path = get_daily_path(DIR_CONVERTED, 'hr', today_date, '.csv')
    df.to_csv(out_hr_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"✅ 今日心率資料已儲存: {out_hr_path} ({len(df)} rows)")

def load_historical_apnea_events(directory: Path) -> pd.DataFrame:
    """掃描 `converted` 資料夾下所有 apnea_events_*.csv，合併並去重。"""
    event_files = sorted(directory.glob("apnea_events_*.csv"))
    if not event_files: return pd.DataFrame(columns=["row_id", "end_apnea"])
    dfs = []
    for f in event_files:
        try:
            df = pd.read_csv(f, parse_dates=["end_apnea"])
            if "end_apnea" in df.columns:
                df = df.dropna(subset=["end_apnea"])
                if "row_id" not in df.columns: df["row_id"] = df["end_apnea"].dt.strftime("%Y%m%d_%H%M%S")
                dfs.append(df[["row_id", "end_apnea"]])
        except Exception as e:
            print(f"⚠️ 警告: 無法解析 {f}. 錯誤: {e}")
    if not dfs: return pd.DataFrame(columns=["row_id", "end_apnea"])
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["row_id"]).sort_values("end_apnea").reset_index(drop=True)
    return all_df

def input_new_apnea_events() -> pd.DataFrame:
    """透過互動介面讓使用者輸入新的呼吸中止事件。"""
    print("\n--- 請輸入新的呼吸中止結束時間 (格式: YYYY-MM-DD HH:MM:SS) ---")
    print("--- 直接按 Enter 結束輸入 ---")
    new_events = []
    while True:
        try: s = input("End apnea time: ").strip()
        except EOFError: break
        if not s: break
        try: new_events.append(pd.to_datetime(s))
        except Exception: print("❌ 格式錯誤，請使用 'YYYY-MM-DD HH:MM:SS'")
    if not new_events: return pd.DataFrame(columns=["row_id", "end_apnea"])
    df = pd.DataFrame({"end_apnea": new_events})
    df["row_id"] = df["end_apnea"].dt.strftime("%Y%m%d_%H%M%S")
    return df[["row_id", "end_apnea"]]

def main():
    """主執行函數"""
    DIR_CONVERTED.mkdir(parents=True, exist_ok=True)
    
    today_date = datetime.now().date()
    print(f"\n===== 開始執行數據清理 {today_date.strftime('%Y-%m-%d')} =====")

    parse_hr_from_xml(FILE_APPLE_HEALTH_XML, today_date)

    historical_events = load_historical_apnea_events(DIR_CONVERTED)
    new_events = input_new_apnea_events()
    
    if new_events.empty:
        merged_events = historical_events.copy()
    else:
        merged_events = pd.concat([historical_events, new_events], ignore_index=True)
        merged_events = merged_events.drop_duplicates(subset=["row_id"]).sort_values("end_apnea").reset_index(drop=True)

    out_apnea_path = get_daily_path(DIR_CONVERTED, 'apnea_events', today_date, '.csv')
    merged_events.to_csv(out_apnea_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"\n✅ 累積呼吸中止事件已儲存: {out_apnea_path} ({len(merged_events)} events)")
    print("===== 處理完畢 =====")

# --- 腳本執行入口 ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 執行失敗: {e}")