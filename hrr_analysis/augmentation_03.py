"""
augmentation_03.py – 特徵數據增強腳本

本腳本為數據處理流程的第三步，目的在於平衡數據分佈。
"""

# --- 導入函式庫 ---
import argparse
import logging
import math
import random
import sys  # <--- 【修正】補上遺漏的 import
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# --- 本地模組 ---
from paths import DIR_FEATURES, DIR_UTILS

# --- 常數與設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

@dataclass(frozen=True)
class AugmentationConfig:
    """數據增強的設定參數"""
    BINS: List[str] = field(default_factory=lambda: ["sluggish", "average", "good", "outstanding"])
    SLOPE_RANGES: Dict[str, tuple] = field(default_factory=lambda: {
        "sluggish": (-0.19, -0.05), "average": (-0.39, -0.21),
        "good": (-0.59, -0.41), "outstanding": (-0.85, -0.61)
    })
    R60_RANGES: Dict[str, tuple] = field(default_factory=lambda: {
        "sluggish": (0.20, 0.50), "average": (0.40, 0.70),
        "good": (0.60, 0.85), "outstanding": (0.80, 0.95)
    })
    DUMMY_RATIO_CAP: float = 0.24
    TARGET_COL: str = "ERS"
    SLOPE_COL: str = "normalized_slope"

# --- 動態載入可選工具 ---
def load_optional_utils() -> tuple:
    """嘗試從 utils 資料夾載入 ers_score 和 mi_selection 函數。"""
    ers_scorer, mi_selector = None, None
    # 暫時將專案根目錄加入 python 路徑，以利找到 utils 模組
    project_root = DIR_UTILS.parent
    sys.path.insert(0, str(project_root))
    try:
        from utils import score_utils
        for cand in ("compute_ers", "ers_from_components", "composite_score", "score_ers"):
            fn = getattr(score_utils, cand, None)
            if callable(fn):
                ers_scorer = fn
                logging.info(f"成功載入 ERS 評分函數: utils.score_utils.{cand}")
                break
    except ImportError:
        logging.warning("未找到或無法載入 ERS 評分函數 (utils/score_utils.py)，將使用內建啟發式規則。")
    except Exception as e:
        logging.error(f"載入 ERS 評分函數時發生錯誤: {e}")

    try:
        from utils.mi import feature_selection_mi
        mi_selector = feature_selection_mi
        logging.info("成功載入 MI 特徵選擇函數: utils.mi.feature_selection_mi")
    except ImportError:
        logging.warning("未找到或無法載入 MI 特徵選擇函數 (utils/mi.py)。")
    except Exception as e:
        logging.error(f"載入 MI 特徵選擇函數時發生錯誤: {e}")
        
    # 恢復原始路徑
    sys.path.pop(0)
    return ers_scorer, mi_selector

# --- 數據增強核心類 ---
class DataAugmenter:
    """封裝所有數據增強相關邏輯的類別。"""

    def __init__(self, config: AugmentationConfig, seed: int):
        self.config = config
        self.seed = seed
        self.df: pd.DataFrame = pd.DataFrame()
        self.ers_scorer, self.mi_selector = load_optional_utils()
        np.random.seed(self.seed)
        random.seed(self.seed)

    def load_and_validate(self, input_path: Path):
        """載入數據並執行所有健全性檢查。"""
        logging.info(f"從 {input_path} 載入特徵數據...")
        if not input_path.exists():
            raise FileNotFoundError(f"找不到輸入檔案：{input_path}")
        if "_aug" in input_path.name.lower():
            raise ValueError(f"輸入檔名包含 '_aug'。請務必使用原始的 features.csv。")
        
        self.df = pd.read_csv(input_path)
        if self.df.empty:
            raise ValueError("輸入的 features.csv 為空。")
        if "dummy_flag" in self.df.columns and self.df["dummy_flag"].sum() > 0:
            raise ValueError("輸入數據已包含 dummy 樣本，請使用原始檔案。")
        if self.config.SLOPE_COL not in self.df.columns:
            raise ValueError(f"數據中缺少必要的斜率欄位: {self.config.SLOPE_COL}")
        
        logging.info(f"數據載入並驗證成功，共 {len(self.df)} 筆原始紀錄。")

    def _get_bin(self, slope: float) -> str:
        """根據 slope 值判斷其所屬的箱體。"""
        if pd.isna(slope): return "average"
        if slope > -0.2: return "sluggish"
        if slope > -0.4: return "average"
        if slope > -0.6: return "good"
        return "outstanding"

    def run_augmentation(self) -> pd.DataFrame:
        """執行完整數據增強流程。"""
        if self.df.empty:
            raise RuntimeError("在執行增強前，請先調用 load_and_validate。")
            
        self.df["__bin__"] = self.df[self.config.SLOPE_COL].apply(self._get_bin)
        counts = self.df["__bin__"].value_counts().reindex(self.config.BINS, fill_value=0)
        
        max_count = int(counts.max())
        needs = {b: max(0, max_count - int(counts[b])) for b in self.config.BINS}
        
        total_needed = sum(needs.values())
        cap = math.floor(self.config.DUMMY_RATIO_CAP * len(self.df))
        
        if total_needed > cap:
            logging.warning(f"所需增補數量 {total_needed} 超過上限 {cap}，將按比例縮減。")
            scale = cap / total_needed if total_needed > 0 else 0
            needs = {b: int(v * scale) for b, v in needs.items()}
        
        logging.info(f"計畫增補的樣本數量: {needs}")
        
        dummies = self._generate_dummies(needs)
        logging.info(f"成功生成 {len(dummies)} 筆 dummy 樣本。")

        final_df = self._finalize_dataframe(dummies)
        
        return final_df
    
    def _generate_dummies(self, needs: Dict[str, int]) -> List[Dict]:
        """根據各箱體所需數量，生成對應的 dummy 樣本。"""
        dummies = []
        for bin_name, k in needs.items():
            if k <= 0: continue
            
            source_df = self.df[self.df["__bin__"] == bin_name]
            if source_df.empty: source_df = self.df
            
            base_rows = source_df.sample(n=k, replace=True, random_state=self.seed)
            
            slopes = np.random.uniform(*self.config.SLOPE_RANGES[bin_name], k)
            r60s = np.random.uniform(*self.config.R60_RANGES[bin_name], k)
            
            for i in range(k):
                new_row = base_rows.iloc[i].copy()
                new_row[self.config.SLOPE_COL] = slopes[i]
                new_row["recovery_ratio_60s"] = r60s[i]
                
                new_row["dummy_flag"] = 1
                base_id = new_row.get("row_id", f"D{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                new_row["row_id"] = f"{base_id}#d{i+1:03d}"
                dummies.append(new_row.to_dict())
        return dummies
        
    def _finalize_dataframe(self, dummies: List[Dict]) -> pd.DataFrame:
        """將 dummy 數據與原始數據合併，並進行最後的清理與檢查。"""
        original_df = self.df.drop(columns=["__bin__"])
        original_df["dummy_flag"] = 0
        
        if not dummies:
            return original_df
            
        dummies_df = pd.DataFrame(dummies)
        
        for col in original_df.columns:
            if col not in dummies_df.columns:
                dummies_df[col] = np.nan
        dummies_df = dummies_df[original_df.columns]

        final_df = pd.concat([original_df, dummies_df], ignore_index=True)
        
        ratio = final_df["dummy_flag"].mean()
        if ratio >= 0.2:
            raise RuntimeError(f"生成後 Dummy 佔比 {ratio:.3f} ≥ 0.2，已中止。請調整策略。")

        for col in ["row_id", "date", "end_apnea_time"]:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str)
        final_df["dummy_flag"] = final_df["dummy_flag"].astype(int)
        
        return final_df

# --- 主執行流程 ---
def main(args: argparse.Namespace):
    """主執行函數，協調 DataAugmenter 的工作流程。"""
    try:
        config = AugmentationConfig()
        augmenter = DataAugmenter(config, args.seed)
        
        input_path = Path(args.input)
        augmenter.load_and_validate(input_path)
        augmented_df = augmenter.run_augmentation()
        
        output_csv = DIR_FEATURES / "features_aug.csv"
        output_parquet = DIR_FEATURES / "features_ml_aug.parquet"
        augmented_df.to_csv(output_csv, index=False)
        augmented_df.to_parquet(output_parquet, index=False)
        logging.info(f"成功儲存增強後的數據: {output_csv} | {output_parquet}")
        logging.info(f"數據摘要: 共 {len(augmented_df)} 筆, Dummy 佔比 {augmented_df['dummy_flag'].mean():.2%}")
        
        if args.run_mi:
            if augmenter.mi_selector:
                logging.info("開始執行 MI 特徵選擇...")
                mi_df = augmenter.mi_selector(augmented_df, target_col=config.TARGET_COL)
                mi_output_path = DIR_FEATURES / "mi_scores_aug.csv"
                mi_df.to_csv(mi_output_path, index=False)
                logging.info(f"成功儲存 MI 分數: {mi_output_path}")
            else:
                logging.warning("無法執行 MI 分析，因為找不到對應的函數。")

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        logging.error(f"腳本執行失敗: {e}")
    except Exception as e:
        logging.error(f"發生未預期的錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    default_input = DIR_FEATURES / "features.csv"
    
    parser = argparse.ArgumentParser(description="03_augmentation - 數據增強模組")
    parser.add_argument("--input", type=str, default=str(default_input), help="原始 features.csv 的路徑")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--run_mi", action="store_true", help="執行 MI 特徵選擇 (需有 utils/mi.py)")
    
    args = parser.parse_args()
    main(args)