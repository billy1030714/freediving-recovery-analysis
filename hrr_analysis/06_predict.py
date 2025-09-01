"""
06_predict.py – Model Prediction and Explanation Script
Generates predictions and explanations using trained models.
"""
# --- Stabilization Settings and Library Imports ---
import os
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
import json, logging, warnings, argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib, numpy as np, pandas as pd, shap
from lime.lime_tabular import LimeTabularExplainer
# --- [FIX] Import the new RMSE function ---
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import percentileofscore

# --- Local Modules ---
from paths import DIR_MODELS, DIR_PREDICTIONS
from utils.data_loader import find_data_file, load_dataframe

# --- Global Settings ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Core Classes (save_results method in Predictor is modified) ---
class Predictor:
    """A class dedicated to executing prediction tasks."""
    def __init__(self, target: str):
        self.target = target
        self.model_dir = DIR_MODELS / self.target
        self.output_dir = DIR_PREDICTIONS / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipe, self.features, self.df = None, None, None
        self.target_distribution = []
        
    def load_artifacts(self):
        """Loads all necessary artifacts for prediction."""
        logging.info(f"===== [Prediction Phase] target = {self.target} =====")
        model_path = self.model_dir / "best_model.joblib"
        schema_path = self.model_dir / "feature_schema.json"
        dist_path = self.model_dir / "target_distribution.json"
        
        self.pipe = joblib.load(model_path)
        with open(schema_path, 'r', encoding='utf-8') as f: self.features = json.load(f)['features']
        
        if dist_path.exists():
            with open(dist_path, 'r', encoding='utf-8') as f: self.target_distribution = json.load(f)['training_data_distribution']
            logging.info(f"Successfully loaded target distribution file with {len(self.target_distribution)} data points.")
        else:
            logging.warning(f"Target distribution file not found: {dist_path}. Cannot calculate percentile rank.")

        data_path = find_data_file()
        self.df = load_dataframe(data_path)
        logging.info(f"Successfully loaded model, features, and data from {data_path.name}.")

    def run_predictions(self) -> pd.DataFrame:
        """Runs predictions on the loaded data."""
        X = self.df[[col for col in self.features if col in self.df.columns]]
        self.df['y_pred'] = self.pipe.predict(X)
        if self.target in self.df.columns:
            self.df['y_true'] = self.df[self.target]
            self.df['abs_err'] = abs(self.df['y_pred'] - self.df['y_true'])
        if self.target_distribution and self.target == "ERS":
            self.df['percentile_rank'] = self.df['y_pred'].apply(lambda x: percentileofscore(self.target_distribution, x))
            logging.info("Successfully calculated personalized percentile ranks.")
        return self.df

    def save_results(self, df_with_preds: pd.DataFrame):
        """Saves prediction results and evaluation metrics."""
        cols_to_save = ["row_id", "y_pred", "y_true", "abs_err", "percentile_rank"]
        final_cols = [c for c in cols_to_save if c in df_with_preds.columns]
        out_df = df_with_preds[final_cols]
        out_df.to_csv(self.output_dir / "preds.csv", index=False)
        out_df.to_parquet(self.output_dir / "preds.parquet", index=False)
        logging.info(f"Prediction results saved to: {self.output_dir / 'preds.csv'}")

        # --- [CRITICAL FIX] ---
        # We need to recreate the time-series split to isolate the validation set for metric calculation.
        # This ensures the metrics here are consistent with the ones from the training script.
        
        # Load the original full dataset again to get the time column
        source_data_path = find_data_file()
        df_all = load_dataframe(source_data_path)
        df_all = df_all[df_all[self.target].notna()] # Filter for valid target rows
        
        if 'end_apnea_time' not in df_all.columns:
            raise ValueError("Missing 'end_apnea_time' for validation split.")
        
        df_all['end_apnea_time'] = pd.to_datetime(df_all['end_apnea_time'])
        df_sorted = df_all.sort_values("end_apnea_time")
        
        # Identify the row_ids that belong to the validation set (the last 30%)
        split_idx = int(len(df_sorted) * 0.7)
        validation_ids = df_sorted.iloc[split_idx:]['row_id'].tolist()
        
        # Filter the prediction output to only include rows from the validation set
        validation_preds_df = out_df[out_df['row_id'].isin(validation_ids)]

        metrics = {}
        if not validation_preds_df.empty and 'y_true' in validation_preds_df.columns:
            # Drop rows where true value is missing, just in case
            evaluation_df = validation_preds_df.dropna(subset=['y_true'])
            if not evaluation_df.empty:
                metrics = {
                    "r2": r2_score(evaluation_df['y_true'], evaluation_df['y_pred']),
                    "mae": mean_absolute_error(evaluation_df['y_true'], evaluation_df['y_pred']),
                    "rmse": root_mean_squared_error(evaluation_df['y_true'], evaluation_df['y_pred']),
                }
        
        data_profile = {"total_rows": len(df_with_preds), "validation_rows_evaluated": len(validation_preds_df)}
        output_data = {"metrics_on_validation_set": metrics, "data_profile": data_profile}
        with open(self.output_dir / "metrics.json", "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Evaluation metrics (on validation set only) saved to: {self.output_dir / 'metrics.json'}")

    def execute(self) -> dict:
        """Executes the full prediction workflow."""
        self.load_artifacts()
        df_with_preds = self.run_predictions()
        self.save_results(df_with_preds)
        return {"target": self.target, "pipe": self.pipe, "features": self.features, "df_with_preds": df_with_preds}

class Explainer:
    """A class dedicated to executing model explanation tasks."""
    def __init__(self, predictor_results: dict):
        self.target = predictor_results['target']
        self.pipe = predictor_results['pipe']
        self.features = predictor_results['features']
        self.df = predictor_results['df_with_preds']
        self.output_dir = DIR_PREDICTIONS / self.target / "explain"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"===== [Explanation Phase] target = {self.target} =====")
        
    def _save_fig(self, filename: str):
        """Saves the current matplotlib figure."""
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logging.info(f"Explanation plot saved: {path}")

    def run_all_explanations(self):
        """
        Runs all explanation methods (SHAP, LIME) with correct feature names.
        """
        X = self.df[self.features]
        
        # --- [FIX] Impute and re-wrap data to preserve feature names for XAI libraries ---
        # 1. Perform imputation to get a NumPy array
        X_imputed_np = self.pipe.named_steps["imputer"].transform(X)
        # 2. Re-wrap the NumPy array into a DataFrame with correct columns and index
        X_imputed = pd.DataFrame(X_imputed_np, columns=self.features, index=X.index)
        
        # --- SHAP Analysis Section ---
        # Now X_imputed is a DataFrame with feature names, which can be safely passed to SHAP
        try:
            # Pass the X_imputed DataFrame to the Explainer
            explainer = shap.Explainer(self.pipe.named_steps["model"], X_imputed)
            shap_values = explainer(X_imputed)
            
            plt.figure()
            # Pass the X_imputed DataFrame as the feature reference for correct labeling
            shap.summary_plot(shap_values, X_imputed, show=False)
            self._save_fig(f"shap_summary_{self.target}.png")
        except Exception as e: 
            logging.warning(f"SHAP analysis failed: {e}")
            
        # --- LIME Analysis Section ---
        try:
            # First, correctly identify the real predictions to analyze
            if "dummy_flag" in self.df.columns:
                real_preds = self.df.dropna(subset=['y_true'])[(self.df["dummy_flag"] == 0)]
            else:
                real_preds = self.df.dropna(subset=['y_true'])

            # Only proceed if there are valid real predictions with an error column
            if not real_preds.empty and 'abs_err' in real_preds.columns:
                # --- [CORE FIX] ---
                # All LIME logic is now moved inside this 'if' block.
                # It will only run if a valid worst-case instance is found.
                
                worst_case_idx = real_preds['abs_err'].idxmax()
                
                # Get the specific instance to explain from the imputed DataFrame
                worst_case_instance = X_imputed.loc[worst_case_idx].values
                
                # Create the LIME explainer
                explainer_lime = LimeTabularExplainer(
                    X_imputed.values, 
                    feature_names=self.features, 
                    mode="regression", 
                    random_state=42
                )
                
                # Explain the instance and generate the plot
                exp = explainer_lime.explain_instance(
                    worst_case_instance, 
                    self.pipe.predict, 
                    num_features=min(10, len(self.features))
                )
                
                fig = exp.as_pyplot_figure()
                self._save_fig(f"lime_worst_case_{self.target}.png")
            else:
                # Log a message if no suitable instance is found for LIME analysis
                logging.info(f"Skipping LIME analysis for target '{self.target}': No valid real predictions found to identify a worst case.")

        except Exception as e: 
            logging.warning(f"LIME analysis failed: {e}")

def main(args):
    """Main function to run prediction and explanation for each target."""
    # --- [CRITICAL FIX] Read targets from environment variable to process all models ---
    targets_str = os.environ.get("TARGETS", "ERS,rmssd_post").strip()
    targets = [t.strip() for t in targets_str.split(",") if t.strip()]
    logging.info(f"Predictor will run for the following targets: {targets}")
    
    for target in targets:
        try:
            predictor = Predictor(target)
            predictor_results = predictor.execute()
            # Only run time-consuming explanations if --fast is not specified
            if not args.fast:
                explainer = Explainer(predictor_results)
                explainer.run_all_explanations()
        except FileNotFoundError as e:
            logging.error(f"Could not process target '{target}' due to missing model files: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing target '{target}': {e}", exc_info=True)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="06_predict - Prediction and Explanation Script")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode: runs prediction only, skips time-consuming SHAP/LIME explanations")
    args = parser.parse_args()
    main(args)