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
    Search for available data files in the features directory, following a priority order.
    Parquet format is preferred, and support is provided for specifying the file via an environment variable.
    """
    # 1. Read path from environment variable to provide maximum flexibility
    env_path = os.environ.get("DATA_FILE", "").strip()
    if env_path and Path(env_path).exists():
        logging.info(f"Locate data file from environment variable DATA_FILE: {env_path}")
        return Path(env_path)
        
    # 2. Search according to the default priority order
    candidates = [
        DIR_FEATURES / "features_ml_aug.parquet",
        DIR_FEATURES / "features_aug.csv",
        DIR_FEATURES / "features_ml.parquet",
        DIR_FEATURES / "features.csv",
    ]
    for path in candidates:
        if path.exists():
            logging.info(f"# Locate data file from the default path: {path}")
            return path
            
    raise FileNotFoundError(f"Locate data file in the features directory {DIR_FEATURES} No available data file found in the features directory.")

def load_dataframe(path: Path) -> pd.DataFrame:
    """Load DataFrame from the specified path based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)