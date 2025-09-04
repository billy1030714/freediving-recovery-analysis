"""
data_loader.py - Shared Data Loading Utility Module

Centralizes the logic for searching and loading data files,
so that all scripts requiring feature data (e.g., 04_models, 05_explainability)
can use a unified and robust interface, in compliance with the DRY (Don't Repeat Yourself) principle.
"""
import os
import logging
from pathlib import Path
import pandas as pd
from paths import DIR_FEATURES

def find_data_file() -> Path:
    """
    INTELLIGENT DATA FILE DISCOVERY:
    
    Search priority:
    1. Parquet format (better performance for large datasets)
    2. CSV format (fallback compatibility)
    3. CI environment auto-detection (_ci suffix)
    4. File size validation (non-empty files only)
    
    Ensures consistent data loading across all pipeline stages.
    """
    is_ci = os.getenv('CI') or os.getenv('GITHUB_ACTIONS')
    file_suffix = '_ci' if is_ci else ''
    
    # Performance-optimized format checking: Parquet first for large datasets
    parquet_path = DIR_FEATURES / f"features_ml{file_suffix}.parquet"
    csv_path = DIR_FEATURES / f"features{file_suffix}.csv"

    if parquet_path.exists() and parquet_path.stat().st_size > 0:
        return parquet_path
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path
        
    raise FileNotFoundError(f"No feature file found (searched for {parquet_path.name} and {csv_path.name})")

def load_dataframe(path: Path) -> pd.DataFrame:
    """Load DataFrame from the specified path based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)