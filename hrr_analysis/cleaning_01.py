"""
cleaning_01.py – Apple Health Data Processor
Parses heart rate data and manages apnea events.
"""

# --- Library Imports ---
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.impute import KNNImputer

# --- Local Modules (our path configuration file) ---
from paths import (
    DIR_CONVERTED,
    FILE_APPLE_HEALTH_XML,
    get_daily_path
)

# --- Constants ---
HR_MIN, HR_MAX = 30, 200 # Valid heart rate range (min and max)

# --- Function Definitions ---

def parse_hr_from_xml(xml_path: Path, today_date: datetime.date) -> None:
    """Parses heart rate data from export.xml, cleans it, and saves it as a CSV."""
    if not xml_path.exists():
        raise FileNotFoundError(f"Apple Health's export.xml not found. Please check the path: {xml_path}")

    rows = []
    # Use iterparse for memory-efficient XML parsing
    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag == "Record" and elem.get("type") == "HKQuantityTypeIdentifierHeartRate":
            t, v = elem.get("startDate"), elem.get("value")
            if t and v:
                try: 
                    rows.append({"Time": t, "HR": float(v)})
                except (ValueError, TypeError): 
                    pass # Ignore records with invalid values
        elem.clear() # Free up memory

    if not rows: 
        raise ValueError("No heart rate (HeartRate) records found in export.xml.")
        
    df = pd.DataFrame(rows)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    
    # Filter out heart rates outside the valid range
    df = df[(df["HR"] >= HR_MIN) & (df["HR"] <= HR_MAX)]
    
    # Impute missing values if any exist
    if df["HR"].isna().any():
        imputer = KNNImputer(n_neighbors=3)
        df["HR"] = imputer.fit_transform(df[["HR"]])

    out_hr_path = get_daily_path(DIR_CONVERTED, 'hr', today_date, '.csv')
    df.to_csv(out_hr_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"✅ Today's heart rate data has been saved: {out_hr_path} ({len(df)} rows)")

def load_historical_apnea_events(directory: Path) -> pd.DataFrame:
    """Scans the `converted` directory for all apnea_events_*.csv files, then merges and deduplicates them."""
    event_files = sorted(directory.glob("apnea_events_*.csv"))
    if not event_files: 
        return pd.DataFrame(columns=["row_id", "end_apnea"])
        
    dfs = []
    for f in event_files:
        try:
            df = pd.read_csv(f, parse_dates=["end_apnea"])
            if "end_apnea" in df.columns:
                df = df.dropna(subset=["end_apnea"])
                # Generate a row_id if it doesn't exist for backward compatibility
                if "row_id" not in df.columns: 
                    df["row_id"] = df["end_apnea"].dt.strftime("%Y%m%d_%H%M%S")
                dfs.append(df[["row_id", "end_apnea"]])
        except Exception as e:
            print(f"⚠️ Warning: Could not parse {f}. Error: {e}")
            
    if not dfs: 
        return pd.DataFrame(columns=["row_id", "end_apnea"])
        
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["row_id"]).sort_values("end_apnea").reset_index(drop=True)
    return all_df

def input_new_apnea_events() -> pd.DataFrame:
    """Allows the user to input new apnea events via an interactive interface."""
    print("\n--- Please enter the new apnea end times (Format: YYYY-MM-DD HH:MM:SS) ---")
    print("--- Press Enter on an empty line to finish input ---")
    new_events = []
    while True:
        try: 
            s = input("End apnea time: ").strip()
        except EOFError: 
            break # Handle end of input stream
        if not s: 
            break # User finished entering data
        try: 
            new_events.append(pd.to_datetime(s))
        except Exception: 
            print("❌ Invalid format. Please use 'YYYY-MM-DD HH:MM:SS'")
            
    if not new_events: 
        return pd.DataFrame(columns=["row_id", "end_apnea"])
        
    df = pd.DataFrame({"end_apnea": new_events})
    df["row_id"] = df["end_apnea"].dt.strftime("%Y%m%d_%H%M%S")
    return df[["row_id", "end_apnea"]]

def main():
    """Main execution function."""
    DIR_CONVERTED.mkdir(parents=True, exist_ok=True)
    
    today_date = datetime.now().date()
    print(f"\n===== Starting Data Cleaning for {today_date.strftime('%Y-%m-%d')} =====")

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
    print(f"\n✅ Cumulative apnea events have been saved: {out_apnea_path} ({len(merged_events)} events)")
    print("===== Processing Complete =====")

# --- Script execution entry point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")