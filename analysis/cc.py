"""
check_sample_size.py - Quick Sample Size Verification

Checks all relevant files to determine actual sample sizes used in the study.
Run this before your interview to get accurate numbers.

Usage:
    python check_sample_size.py
"""

import pandas as pd
from pathlib import Path
from paths import DIR_FEATURES, DIR_CONVERTED
import os

def check_features_file():
    """Check main features file."""
    print("=" * 80)
    print("1. MAIN FEATURES FILE")
    print("=" * 80)
    
    # Check for both regular and CI versions (same logic as 02_features.py)
    is_ci = os.getenv('CI') or os.getenv('GITHUB_ACTIONS')
    file_suffix = '_ci' if is_ci else ''
    
    features_path = DIR_FEATURES / f"features{file_suffix}.csv"
    
    print(f"\n🔍 Looking for: {features_path}")
    
    if not features_path.exists():
        print(f"❌ File not found: {features_path}")
        # Try alternative
        alt_path = DIR_FEATURES / "features.csv"
        if alt_path.exists() and file_suffix:
            print(f"✓ Found alternative: {alt_path}")
            features_path = alt_path
        else:
            return None
    
    df = pd.read_csv(features_path)
    
    print(f"\n📁 File: {features_path}")
    print(f"📊 Total rows: {len(df)}")
    print(f"\n🔍 Column check:")
    print(f"   Columns: {list(df.columns)}")
    
    # Check ERS availability
    if 'ERS' in df.columns:
        ers_available = df['ERS'].notna().sum()
        ers_missing = df['ERS'].isna().sum()
        print(f"\n📈 ERS Statistics:")
        print(f"   Total events: {len(df)}")
        print(f"   ERS available: {ers_available} ({ers_available/len(df)*100:.1f}%)")
        print(f"   ERS missing: {ers_missing} ({ers_missing/len(df)*100:.1f}%)")
    
    # Check component availability
    components = ['recovery_ratio_60s', 'recovery_ratio_90s', 'normalized_slope']
    print(f"\n🧩 Component Availability:")
    for comp in components:
        if comp in df.columns:
            available = df[comp].notna().sum()
            print(f"   {comp}: {available}/{len(df)} ({available/len(df)*100:.1f}%)")
    
    # Check feature count
    if 'ers_feature_count' in df.columns:
        print(f"\n📊 Feature Count Distribution:")
        print(df['ers_feature_count'].value_counts().sort_index())
        print(f"   Average: {df['ers_feature_count'].mean():.2f}")
    
    return df

def check_ml_file():
    """Check ML parquet file."""
    print("\n" + "=" * 80)
    print("2. ML PARQUET FILE")
    print("=" * 80)
    
    # Check for both regular and CI versions
    is_ci = os.getenv('CI') or os.getenv('GITHUB_ACTIONS')
    file_suffix = '_ci' if is_ci else ''
    
    ml_path = DIR_FEATURES / f"features_ml{file_suffix}.parquet"
    
    print(f"\n🔍 Looking for: {ml_path}")
    
    if not ml_path.exists():
        print(f"❌ File not found: {ml_path}")
        # Try alternative
        alt_path = DIR_FEATURES / "features_ml.parquet"
        if alt_path.exists() and file_suffix:
            print(f"✓ Found alternative: {alt_path}")
            ml_path = alt_path
        else:
            return None
    
    df = pd.read_parquet(ml_path)
    
    print(f"\n📁 File: {ml_path}")
    print(f"📊 Total rows: {len(df)}")
    
    if 'ERS' in df.columns:
        ers_available = df['ERS'].notna().sum()
        print(f"📈 ERS available: {ers_available} ({ers_available/len(df)*100:.1f}%)")
    
    return df

def check_train_test_split():
    """Check actual train/test split used in modeling."""
    print("\n" + "=" * 80)
    print("3. TRAIN/TEST SPLIT (from last model run)")
    print("=" * 80)
    
    from pathlib import Path
    models_dir = Path("models")
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    # Try to find train/test files
    train_files = list(models_dir.glob("*train*.csv"))
    test_files = list(models_dir.glob("*test*.csv"))
    
    if not train_files and not test_files:
        print("⚠️  No train/test CSV files found in models directory")
        print("    Checking for other indicators...")
        
        # Check for model metadata
        import glob
        import json
        
        json_files = glob.glob(str(models_dir / "*.json"))
        if json_files:
            print(f"\n📄 Found {len(json_files)} model metadata files")
            for jf in json_files[:3]:  # Show first 3
                try:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        if 'n_samples' in data or 'train_size' in data:
                            print(f"   {Path(jf).name}: {data}")
                except:
                    pass
        return
    
    print(f"\n📂 Found files:")
    print(f"   Train files: {len(train_files)}")
    print(f"   Test files: {len(test_files)}")
    
    if train_files:
        latest_train = max(train_files, key=lambda x: x.stat().st_mtime)
        print(f"\n🔍 Latest train file: {latest_train.name}")
        try:
            train_df = pd.read_csv(latest_train)
            print(f"   Train samples: {len(train_df)}")
        except Exception as e:
            print(f"   ⚠️ Could not read: {e}")
    
    if test_files:
        latest_test = max(test_files, key=lambda x: x.stat().st_mtime)
        print(f"\n🔍 Latest test file: {latest_test.name}")
        try:
            test_df = pd.read_csv(latest_test)
            print(f"   Test samples: {len(test_df)}")
        except Exception as e:
            print(f"   ⚠️ Could not read: {e}")
    
    if train_files and test_files:
        try:
            train_size = len(pd.read_csv(latest_train))
            test_size = len(pd.read_csv(latest_test))
            total = train_size + test_size
            print(f"\n📊 Split Summary:")
            print(f"   Train: {train_size} ({train_size/total*100:.1f}%)")
            print(f"   Test: {test_size} ({test_size/total*100:.1f}%)")
            print(f"   Total: {total}")
        except:
            pass

def check_converted_files():
    """Check raw data files."""
    print("\n" + "=" * 80)
    print("4. RAW DATA FILES")
    print("=" * 80)
    
    if not DIR_CONVERTED.exists():
        print(f"❌ Converted directory not found: {DIR_CONVERTED}")
        return
    
    # Check for both regular and CI versions (same as 02_features.py)
    is_ci = os.getenv('CI') or os.getenv('GITHUB_ACTIONS')
    file_suffix = '_ci' if is_ci else ''
    
    # Get files with correct suffix pattern (same as 02_features.py)
    apnea_pattern = f"apnea_events{file_suffix}_*.csv"
    hr_pattern = f"hr{file_suffix}_*.csv"
    
    all_apnea_files = list(DIR_CONVERTED.glob("apnea_events*.csv"))
    all_hr_files = list(DIR_CONVERTED.glob("hr*.csv"))
    
    # Filter: if no suffix, exclude files with '_ci' in the middle
    if not file_suffix:
        apnea_files = [f for f in all_apnea_files if not ('_ci_' in f.name or f.name.startswith('apnea_events_ci'))]
        hr_files = [f for f in all_hr_files if not ('_ci_' in f.name or f.name.startswith('hr_ci'))]
    else:
        apnea_files = [f for f in all_apnea_files if file_suffix in f.name]
        hr_files = [f for f in all_hr_files if file_suffix in f.name]
    
    print(f"\n🔍 File pattern: apnea_events{file_suffix}_*.csv (excluding CI files)")
    print(f"📂 Files in {DIR_CONVERTED}:")
    print(f"   Apnea event files found: {len(apnea_files)}")
    print(f"   HR data files found: {len(hr_files)}")
    
    if apnea_files:
        total_events = 0
        print(f"\n📋 All apnea event files found:")
        file_info = []
        for af in sorted(apnea_files):
            try:
                df = pd.read_csv(af)
                events = len(df)
                file_info.append((af, events))
                print(f"   {af.name}: {events} events")
            except Exception as e:
                print(f"   {af.name}: ⚠️ Could not read")
        
        if file_info:
            # Find the latest file (same logic as 02_features.py - discovers all but processes)
            latest_file, latest_events = max(file_info, key=lambda x: x[0].name)
            
            print(f"\n   ⭐ Latest file (used for analysis): {latest_file.name}")
            print(f"   📊 Events in latest file: {latest_events}")
            print(f"\n   💡 Note: 02_features.py discovers all files but merges/deduplicates")
            print(f"            The final features.csv contains unique events only.")
            
            return latest_events
        return None
    else:
        print(f"\n⚠️  No files found matching pattern")
        return None

def check_params_yaml():
    """Check params.yaml for configuration."""
    print("\n" + "=" * 80)
    print("5. CONFIGURATION CHECK")
    print("=" * 80)
    
    import yaml
    params_path = Path("params.yaml")
    
    if not params_path.exists():
        print(f"❌ File not found: {params_path}")
        return
    
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    
    print(f"\n📄 params.yaml settings:")
    
    if 'data_split' in params:
        print(f"\n🔀 Data Split:")
        print(f"   Train ratio: {params['data_split'].get('train_ratio', 'N/A')}")
        print(f"   Test ratio: {params['data_split'].get('test_ratio', 'N/A')}")
    
    if 'feature_engineering' in params:
        fe = params['feature_engineering']
        print(f"\n⚙️  Feature Engineering:")
        if 'base_window' in fe:
            print(f"   Baseline window: {fe['base_window']}")
        if 'peak_max_seconds' in fe:
            print(f"   Peak search window: {fe['peak_max_seconds']}s")

def generate_summary(features_df):
    """Generate final summary for interview."""
    print("\n" + "=" * 80)
    print("📋 INTERVIEW CHEAT SHEET")
    print("=" * 80)
    
    if features_df is None:
        print("⚠️  Could not generate summary - features file not found")
        return
    
    total_events = len(features_df)
    ers_available = features_df['ERS'].notna().sum() if 'ERS' in features_df.columns else 0
    
    print(f"""
📊 KEY NUMBERS TO REMEMBER:

1. Sample Size:
   • Total apnea events: {total_events}
   • Events with valid ERS: {ers_available} ({ers_available/total_events*100:.1f}%)
   
2. Data Split (approximately 70/30):
   • Training set: ~{int(total_events * 0.7)} events
   • Validation set: ~{int(total_events * 0.3)} events
   
3. Feature Completeness (with ±3s tolerance):
   • Average components available: {features_df['ers_feature_count'].mean():.2f}/3
   • Success rate: {ers_available/total_events*100:.1f}%

4. What to say in interview:
   "I collected {total_events} freediving apnea events from my personal training.
   After feature engineering, {ers_available} events had valid ERS scores.
   Using a 70/30 time-series split, approximately {int(total_events * 0.7)} events 
   were used for training and {int(total_events * 0.3)} for validation."
""")

    # Component breakdown
    if 'recovery_ratio_60s' in features_df.columns:
        print("5. Component Availability:")
        components = ['recovery_ratio_60s', 'recovery_ratio_90s', 'normalized_slope']
        for comp in components:
            if comp in features_df.columns:
                avail = features_df[comp].notna().sum()
                print(f"   • {comp}: {avail}/{total_events} ({avail/total_events*100:.1f}%)")

def main():
    """Main execution."""
    print("\n" + "🔍 " * 20)
    print("SAMPLE SIZE VERIFICATION REPORT")
    print("🔍 " * 20)
    
    # Check environment
    is_ci = os.getenv('CI') or os.getenv('GITHUB_ACTIONS')
    if is_ci:
        print("\n⚙️  CI environment detected - looking for '_ci' suffix files")
    else:
        print("\n⚙️  Regular environment - looking for standard files")
    
    # Check all sources
    features_df = check_features_file()
    check_ml_file()
    check_train_test_split()
    total_raw_events = check_converted_files()
    check_params_yaml()
    
    # Generate summary
    if features_df is not None:
        generate_summary(features_df)
    elif total_raw_events:
        print("\n" + "=" * 80)
        print("📋 PARTIAL SUMMARY (from raw data only)")
        print("=" * 80)
        print(f"""
⚠️  Features file not found, but raw data shows:

📊 Raw Data (Latest File):
   • Events in latest raw file: {total_raw_events}
   • Estimated after 70/30 split:
     - Training: ~{int(total_raw_events * 0.7)} events
     - Validation: ~{int(total_raw_events * 0.3)} events
   
💡 Action needed:
   Run: python hrr_analysis/02_features.py
   
   This will process the raw data and generate features.csv
   The actual sample size may differ after feature engineering
   (some events may be excluded due to data quality).
        """)
    
    print("\n" + "=" * 80)
    print("✅ Verification Complete!")
    print("=" * 80)
    print("\n💡 Tip: If features files are missing, run 02_features.py first!")

if __name__ == "__main__":
    main()