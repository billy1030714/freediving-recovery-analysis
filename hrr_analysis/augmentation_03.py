"""
augmentation_03.py – Data Augmentation for Recovery Metrics
Generates synthetic samples to balance data distribution.
"""

# --- Library Imports ---
import argparse
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# --- Local Modules ---
from paths import DIR_FEATURES, DIR_UTILS

# --- Constants and Settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration parameters for data augmentation."""
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

# --- Dynamic Loading of Optional Utilities ---
def load_optional_utils() -> tuple:
    """Attempts to load ers_scorer and mi_selector functions from the installed utils package."""
    ers_scorer, mi_selector = None, None
    try:
        # Import directly, as 'utils' is an identifiable package
        from utils import score_utils
        ers_scorer = score_utils.compute_ers
        logging.info(f"Successfully loaded ERS scoring function: utils.score_utils.compute_ers")

    except ImportError:
        logging.warning("ERS scoring function (utils/score_utils.py) not found or could not be loaded. Will use built-in heuristics if needed.")
    except Exception as e:
        logging.error(f"An error occurred while loading the ERS scoring function: {e}")

    try:
        # Import directly as well
        from utils.mi import feature_selection_mi
        mi_selector = feature_selection_mi
        logging.info("Successfully loaded MI feature selection function: utils.mi.feature_selection_mi")
    except ImportError:
        logging.warning("MI feature selection function (utils/mi.py) not found or could not be loaded.")
    except Exception as e:
        logging.error(f"An error occurred while loading the MI feature selection function: {e}")
        
    return ers_scorer, mi_selector

# --- Core Data Augmentation Class ---
class DataAugmenter:
    """A class that encapsulates all logic related to data augmentation."""

    def __init__(self, config: AugmentationConfig, seed: int):
        self.config = config
        self.seed = seed
        self.df: pd.DataFrame = pd.DataFrame()
        self.ers_scorer, self.mi_selector = load_optional_utils()
        np.random.seed(self.seed)
        random.seed(self.seed)

    def load_and_validate(self, input_path: Path):
        """Loads data and performs all sanity checks."""
        logging.info(f"Loading feature data from {input_path}...")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if "_aug" in input_path.name.lower():
            raise ValueError(f"Input filename contains '_aug'. Please use the original features.csv.")
        
        self.df = pd.read_csv(input_path)
        if self.df.empty:
            raise ValueError("The input features.csv is empty.")
        if "dummy_flag" in self.df.columns and self.df["dummy_flag"].sum() > 0:
            raise ValueError("Input data already contains dummy samples. Please use the original file.")
        if self.config.SLOPE_COL not in self.df.columns:
            raise ValueError(f"Data is missing the required slope column: {self.config.SLOPE_COL}")
        
        logging.info(f"Data loaded and validated successfully, with {len(self.df)} original records.")

    def _get_bin(self, slope: float) -> str:
        """Determines which bin a given slope value belongs to."""
        if pd.isna(slope): return "average"
        if slope > -0.2: return "sluggish" # Note: Assumes slope is negative
        if slope > -0.4: return "average"
        if slope > -0.6: return "good"
        return "outstanding"

    def run_augmentation(self) -> pd.DataFrame:
        """Executes the full data augmentation process."""
        if self.df.empty:
            raise RuntimeError("Please call load_and_validate before running augmentation.")
            
        self.df["__bin__"] = self.df[self.config.SLOPE_COL].apply(self._get_bin)
        counts = self.df["__bin__"].value_counts().reindex(self.config.BINS, fill_value=0)
        
        max_count = int(counts.max())
        needs = {b: max(0, max_count - int(counts[b])) for b in self.config.BINS}
        
        total_needed = sum(needs.values())
        cap = math.floor(self.config.DUMMY_RATIO_CAP * len(self.df))
        
        if total_needed > cap:
            logging.warning(f"Number of samples to generate ({total_needed}) exceeds the cap ({cap}). The number will be scaled down proportionally.")
            scale = cap / total_needed if total_needed > 0 else 0
            needs = {b: int(v * scale) for b, v in needs.items()}
        
        logging.info(f"Planned number of samples to generate: {needs}")
        
        dummies = self._generate_dummies(needs)
        logging.info(f"Successfully generated {len(dummies)} dummy samples.")

        final_df = self._finalize_dataframe(dummies)
        
        return final_df
    
    def _generate_dummies(self, needs: Dict[str, int]) -> List[Dict]:
        """Generates synthetic dummy samples based on the required number for each bin."""
        dummies = []
        for bin_name, k in needs.items():
            if k <= 0: continue
            
            source_df = self.df[self.df["__bin__"] == bin_name]
            if source_df.empty: source_df = self.df # Fallback to the full dataset if a bin is empty
            
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
        """Merges dummy data with original data and performs final cleaning and checks."""
        original_df = self.df.drop(columns=["__bin__"])
        original_df["dummy_flag"] = 0
        
        if not dummies:
            return original_df
            
        dummies_df = pd.DataFrame(dummies)
        
        # Ensure dummy DataFrame has the same columns as the original
        for col in original_df.columns:
            if col not in dummies_df.columns:
                dummies_df[col] = np.nan
        dummies_df = dummies_df[original_df.columns]

        final_df = pd.concat([original_df, dummies_df], ignore_index=True)
        
        ratio = final_df["dummy_flag"].mean()
        if ratio >= self.config.DUMMY_RATIO_CAP:
            raise RuntimeError(f"Dummy ratio after generation ({ratio:.3f}) is >= the cap ({self.config.DUMMY_RATIO_CAP}). Process aborted. Please adjust the strategy.")

        for col in ["row_id", "date", "end_apnea_time"]:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str)
        final_df["dummy_flag"] = final_df["dummy_flag"].astype(int)
        
        return final_df

# --- Main Execution Flow ---
def main(args: argparse.Namespace):
    """Main execution function that coordinates the DataAugmenter workflow."""
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
        logging.info(f"Successfully saved augmented data to: {output_csv} | {output_parquet}")
        logging.info(f"Data summary: {len(augmented_df)} total records, dummy ratio: {augmented_df['dummy_flag'].mean():.2%}")
        
        if args.run_mi:
            if augmenter.mi_selector:
                logging.info("Starting MI feature selection...")
                mi_df = augmenter.mi_selector(augmented_df, target_col=config.TARGET_COL)
                mi_output_path = DIR_FEATURES / "mi_scores_aug.csv"
                mi_df.to_csv(mi_output_path, index=False)
                logging.info(f"Successfully saved MI scores to: {mi_output_path}")
            else:
                logging.warning("Cannot run MI analysis because the corresponding function was not found.")

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        logging.error(f"Script execution failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    default_input = DIR_FEATURES / "features.csv"
    
    parser = argparse.ArgumentParser(description="03_augmentation - Data Augmentation Module")
    parser.add_argument("--input", type=str, default=str(default_input), help="Path to the original features.csv file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--run_mi", action="store_true", help="Run MI feature selection (requires utils/mi.py)")
    
    args = parser.parse_args()
    main(args)