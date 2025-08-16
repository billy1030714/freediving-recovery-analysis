"""
data_loader.py - 共用的數據讀取工具模組

將數據檔案的搜尋與讀取邏輯集中於此，
讓所有需要讀取特徵數據的腳本 (如 04_models, 05_explainability)
都能使用統一、穩健的接口，遵守 DRY (Don't Repeat Yourself) 原則。
"""
import os
import logging
from pathlib import Path
import pandas as pd
from paths import DIR_FEATURES

def find_data_file() -> Path:
    """
    在特徵資料夾中，按優先順序尋找可用的數據檔案。
     parquet 格式優先，並支援從環境變數指定檔案。
    """
    # 1. 從環境變數讀取路徑，提供最高的靈活性
    env_path = os.environ.get("DATA_FILE", "").strip()
    if env_path and Path(env_path).exists():
        logging.info(f"從環境變數 DATA_FILE 中找到數據: {env_path}")
        return Path(env_path)
        
    # 2. 按預設的優先順序搜尋
    candidates = [
        DIR_FEATURES / "features_ml_aug.parquet", # 03 的主要輸出
        DIR_FEATURES / "features_aug.csv",
        DIR_FEATURES / "features_ml.parquet",
        DIR_FEATURES / "features.csv",
    ]
    for path in candidates:
        if path.exists():
            logging.info(f"在預設路徑中找到數據: {path}")
            return path
            
    raise FileNotFoundError(f"在特徵資料夾 {DIR_FEATURES} 中找不到任何可用的數據檔案。")

def load_dataframe(path: Path) -> pd.DataFrame:
    """根據副檔名，從指定路徑載入 DataFrame。"""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)