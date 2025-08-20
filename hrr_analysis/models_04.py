"""
models_04.py – Model Training, Evaluation, and Versioning Script

[Update]:
- When saving artifacts, an additional `target_distribution.json` file is now saved.
- This file contains the target values from the real training data (`y_real_train`) and will be used by the downstream
  `06_predict.py` script to calculate percentile rankings.
"""

# --- Library Imports ---
import argparse, json, logging, os, warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple
import joblib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# --- Local Modules ---
from paths import DIR_MODELS
from utils.data_loader import find_data_file, load_dataframe

# --- Global Settings and Version Freezing ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
RANDOM_SEED = 42
MODEL_CONFIG = {
    "xgb": {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "reg_alpha": 0.0, "random_state": RANDOM_SEED, "n_jobs": -1, "tree_method": "hist"},
    "rf": {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 2, "random_state": RANDOM_SEED, "n_jobs": -1},
    "ridge": {"alpha": 1.0, "random_state": RANDOM_SEED}
}

class ModelTrainer:
    """A class to encapsulate the model training and evaluation workflow."""
    def __init__(self, target_name: str, seed: int):
        self.target = target_name
        self.seed = seed
        self.output_dir = DIR_MODELS / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_df = pd.DataFrame()
        self.y_real_train = pd.Series()
        self.feature_cols: List[str] = []
        logging.info(f"===== [Initializing] ModelTrainer for target: {self.target} =====")

    def run(self, df: pd.DataFrame, data_path: Path):
        """Executes the full training pipeline for the given target."""
        try:
            df_valid = self._prepare_data(df)
            X_train, X_test, y_train, y_test, meta = self._split_data(df_valid)
            board, best_model_info = self._train_and_evaluate(X_train, X_test, y_train, y_test)
            self._save_artifacts(best_model_info, board, self.feature_cols, meta, data_path)
            logging.info(f"===== [Success] Target '{self.target}' training complete. =====")
        except Exception as e:
            logging.error(f"An unexpected error occurred while training for target '{self.target}': {e}", exc_info=True)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares and validates data using a dual-track strategy based on TASK_TYPE.
        - 'short_term': For ERS algorithm design, allows ERS components as features.
        - 'long_term': For HRV predictability research, uses strict leakage prevention.
        """
        # --- Get task type from environment variable to control the logic ---
        task_type = os.environ.get("TASK_TYPE", "long_term").strip().lower()
        
        # Log the current configuration for transparency
        logging.info(f"TASK_TYPE environment variable: '{task_type}'")
        logging.info(f"Target variable: '{self.target}'")

        # --- Define column groups for robust feature exclusion ---
        ALL_POSSIBLE_TARGETS = {"ERS", "rmssd_post", "sdnn_post", "pnn50_post", "mean_rr_post", "hrr60"}
        METADATA_COLS = {"row_id", "date", "end_apnea_time", "dummy_flag"}
        ERS_COMPONENTS = {"recovery_ratio_60s", "recovery_ratio_90s", "normalized_slope"}

        # --- Data validation ---
        if self.target not in df.columns:
            raise KeyError(f"Target column '{self.target}' does not exist in dataframe columns: {list(df.columns)}")
        
        df_valid = df[df[self.target].notna()].copy()
        
        # Apply target-specific filtering
        if self.target in ALL_POSSIBLE_TARGETS and self.target != "ERS":
            original_len = len(df_valid)
            df_valid = df_valid[df_valid[self.target] > 0]
            filtered_len = len(df_valid)
            logging.info(f"Filtered out {original_len - filtered_len} samples with non-positive {self.target} values")
        
        if df_valid.empty:
            raise ValueError(f"No valid data available for target '{self.target}' after filtering.")
        
        if 'end_apnea_time' in df_valid.columns:
            df_valid['end_apnea_time'] = pd.to_datetime(df_valid['end_apnea_time'])
        
        # --- Dual-Track Feature Exclusion Logic ---
        logging.info(f"Running in '{task_type}' mode for target '{self.target}'.")
        cols_to_exclude = set()

        if task_type == 'short_term':
            # --- Product Track (A-Track): ERS Algorithm Design ---
            logging.info("PRODUCT DESIGN TRACK: Allowing ERS components for algorithm validation")
            
            # The goal is to analyze feature importance for ERS components.
            # We allow ERS components as features, excluding only the target itself and metadata.
            cols_to_exclude.update(METADATA_COLS)
            cols_to_exclude.add(self.target)
            
            # Exclude other potential high-level targets to keep the focus clean,
            # but keep ERS components if target is ERS
            if self.target == "ERS":
                other_targets = {"rmssd_post", "sdnn_post", "pnn50_post", "mean_rr_post", "hrr60"}
            else:
                other_targets = ALL_POSSIBLE_TARGETS - {self.target}
            cols_to_exclude.update(other_targets)
            
            logging.info(f"Short-term track: ERS components will be {'INCLUDED' if self.target == 'ERS' else 'EXCLUDED'}")

        else: # Default to 'long_term' for safety
            # --- Research Track (B-Track): Predictability Validation ---
            logging.info("SCIENTIFIC RESEARCH TRACK: Strict leakage prevention enabled")
            
            # Use the strict exclusion logic to prevent any leakage.
            cols_to_exclude.update(METADATA_COLS)
            
            # Exclude all possible targets and their derivatives (e.g., 'rmssd_post', 'rmssd_post.1')
            derived_target_cols = {c for c in df.columns if any(c.startswith(t) for t in ALL_POSSIBLE_TARGETS)}
            cols_to_exclude.update(derived_target_cols)
            
            # ERS components must also be removed in this strict research track
            cols_to_exclude.update(ERS_COMPONENTS)
            
            logging.info("Long-term track: All post-dive features and targets excluded")

        # Create and store the final, clean feature list
        available_numeric_cols = set(df.select_dtypes(include=np.number).columns)
        self.feature_cols = [c for c in available_numeric_cols if c not in cols_to_exclude]
        
        # Validate that we have features to work with
        if not self.feature_cols:
            raise ValueError(f"No valid features remaining after exclusion. Available: {available_numeric_cols}, Excluded: {cols_to_exclude}")
        
        logging.info(f"Data preparation complete: {len(df_valid)} valid samples, {len(self.feature_cols)} features.")
        logging.info(f"Excluded columns: {sorted(cols_to_exclude)}")
        logging.info(f"Final feature list: {sorted(self.feature_cols)}")
        
        return df_valid

    def _split_data(self, df_valid: pd.DataFrame) -> tuple:
        """
        Splits the data into training and testing sets using a time-series strategy.
        This version now uses the 'self.feature_cols' attribute created by _prepare_data.
        """
        logging.info("Performing time-series split...")
        meta = {"split_strategy": "time_series_70_30"}

        # --- Isolate real data for splitting ---
        if "dummy_flag" in df_valid.columns:
            real_df = df_valid[df_valid["dummy_flag"] == 0].copy()
        else:
            real_df = df_valid.copy()
        
        if 'end_apnea_time' not in real_df.columns:
            raise ValueError("Missing 'end_apnea_time' column for time-series split.")
        
        # --- Perform time-based split (70/30) ---
        real_df_sorted = real_df.sort_values("end_apnea_time")
        split_idx = int(len(real_df_sorted) * 0.7)
        if split_idx < 1 or split_idx >= len(real_df_sorted):
            raise ValueError(f"Real dataset size ({len(real_df_sorted)}) is too small for a meaningful split.")
        
        real_train_df = real_df_sorted.iloc[:split_idx]
        real_test_df = real_df_sorted.iloc[split_idx:]
        
        # --- CRITICAL CHANGE: Use `self.feature_cols` to select columns for X ---
        # This ensures the clean, leak-free feature list is used every time.
        X_real_train = real_train_df[self.feature_cols]
        y_real_train = real_train_df[self.target]
        self.y_real_train = y_real_train  # Store for later saving
        
        X_test = real_test_df[self.feature_cols]
        y_test = real_test_df[self.target]
        
        # In this final version, training set is the real training set.
        X_train = X_real_train
        y_train = y_real_train

        # Update metadata
        n_train_dummy = len(self.dummy_df) if hasattr(self, 'dummy_df') and not self.dummy_df.empty else 0
        meta.update({
            "n_train_real": len(X_train), 
            "n_train_dummy": n_train_dummy, 
            "n_test_real": len(X_test)
        })
        logging.info(f"Data split complete: Training set has {len(X_train)} samples, Validation set has {len(X_test)} samples.")
        
        return X_train, X_test, y_train, y_test, meta

    def _train_and_evaluate(self, X_train, X_test, y_train, y_test) -> tuple:
        """Trains multiple models and evaluates them to find the best one."""
        # --- DEBUG PROBE (ARCHIVED FOR FUTURE USE) ---
        # if self.target == 'ERS':
        #     logging.info("===== Debug Probe: Checking training data for ERS model =====")
        #     logging.info(f"y_train (ERS) stats:\n{y_train.describe().to_string()}")
        #     logging.info(f"X_train columns:\n{X_train.columns.tolist()}")
        #     logging.info("==================== End of Debug Probe ====================")
        
        models_to_train = {
            "xgb": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", xgb.XGBRegressor(**MODEL_CONFIG["xgb"]))]),
            "rf": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(**MODEL_CONFIG["rf"]))]),
            "ridge": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", Ridge(**MODEL_CONFIG["ridge"]))]),
        }
        leaderboard, best_model_info = [], {"r2": -1e9}
        
        for name, pipe in models_to_train.items():
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = {"name": name, "r2": r2_score(y_test, y_pred), "mae": mean_absolute_error(y_test, y_pred), "rmse": np.sqrt(mean_squared_error(y_test, y_pred))}
            leaderboard.append(metrics)
            logging.info(f"[Evaluation] {name}: R²={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}")
            if metrics["r2"] > best_model_info["r2"]:
                best_model_info = metrics.copy()
                best_model_info["pipeline"] = pipe
                
        logging.info(f"[Best Model] The winner of this run is: {best_model_info['name']} (R² = {best_model_info['r2']:.4f})")
        return leaderboard, best_model_info

    def _save_artifacts(self, best_model_info, leaderboard, feature_columns, dataset_metadata, data_path):
        """Saves all model artifacts, including the model file, metadata, and the new target distribution file."""
        # Save model, leaderboard, and schemas
        best_model_path = self.output_dir / "best_model.joblib"
        joblib.dump(best_model_info["pipeline"], best_model_path)
        
        # Special handling for ERS model (backwards compatibility)
        if self.target == "ERS": 
            joblib.dump(best_model_info["pipeline"], self.output_dir / "model.joblib")
        
        # Save leaderboard
        with open(self.output_dir / "leaderboard.json", "w", encoding="utf-8") as f: 
            json.dump(leaderboard, f, indent=2, ensure_ascii=False)
        
        # Save feature schema
        feature_schema = {
            "schema_version": "1.0", 
            "feature_count": len(feature_columns), 
            "features": feature_columns,
            "task_type": os.environ.get("TASK_TYPE", "long_term").strip(),  # Record the task type
            "target": self.target
        }
        with open(self.output_dir / "feature_schema.json", "w", encoding="utf-8") as f: 
            json.dump(feature_schema, f, indent=2, ensure_ascii=False)
        
        # Quality gate check
        quality_threshold = 0.05
        passed_quality_gate = best_model_info.get('r2', -1) > quality_threshold

        # Save comprehensive dataset card
        dataset_card = {
            "card_version": "1.0", 
            "source_file": str(data_path), 
            "target_variable": self.target,
            "task_type": os.environ.get("TASK_TYPE", "long_term").strip(),  # Record the task type
            "best_model_name": best_model_info["name"], 
            "evaluation_metrics": {k: v for k, v in best_model_info.items() if k != 'pipeline'},
            "passed_quality_gate": passed_quality_gate,
            "quality_threshold": quality_threshold,
            "dataset_split": dataset_metadata, 
            "random_seed": self.seed,
            "feature_exclusion_strategy": {
                "excluded_columns_count": len(set(feature_columns)) if hasattr(self, 'excluded_cols') else 0,
                "final_feature_count": len(feature_columns)
            }
        }
        with open(self.output_dir / "dataset_card.json", "w", encoding="utf-8") as f: 
            json.dump(dataset_card, f, indent=2, ensure_ascii=False)

        # Save the target distribution of the real training data
        dist_path = self.output_dir / "target_distribution.json"
        dist_data = {
            "target": self.target,
            "task_type": os.environ.get("TASK_TYPE", "long_term").strip(),
            "training_data_distribution": self.y_real_train.tolist(),
            "distribution_stats": {
                "mean": float(self.y_real_train.mean()),
                "std": float(self.y_real_train.std()),
                "min": float(self.y_real_train.min()),
                "max": float(self.y_real_train.max()),
                "count": len(self.y_real_train)
            }
        }
        with open(dist_path, "w", encoding="utf-8") as f:
            json.dump(dist_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Target distribution of real training data saved to: {dist_path}")
        logging.info(f"All artifacts have been successfully saved to: {self.output_dir}")

def main():
    """Main function to run the training process for all specified targets."""
    # Get configuration from environment variables
    targets_str = os.environ.get("TARGETS", "ERS,rmssd_post").strip()
    task_type = os.environ.get("TASK_TYPE", "long_term").strip()
    
    targets = [t.strip() for t in targets_str.split(",") if t.strip()]
    
    logging.info("="*60)
    logging.info(f"STARTING MODEL TRAINING SESSION")
    logging.info(f"Task Type: {task_type}")
    logging.info(f"Targets: {targets}")
    logging.info("="*60)
    
    try:
        data_path = find_data_file()
        df = load_dataframe(data_path)
        logging.info(f"Loaded dataset: {data_path} ({len(df)} samples)")
        
        for target in targets:
            logging.info(f"\n{'='*50}")
            logging.info(f"PROCESSING TARGET: {target}")
            logging.info(f"{'='*50}")
            
            trainer = ModelTrainer(target_name=target, seed=RANDOM_SEED)
            trainer.run(df, data_path)
            
            logging.info(f"Completed training for target: {target}")
            
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main process: {e}", exc_info=True)
        raise
    
if __name__ == "__main__":
    main()