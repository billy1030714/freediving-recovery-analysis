"""
05_explainability.py – Model Explainability Analysis
Generates SHAP, LIME, permutation importance, and PDP visualizations.
"""
# --- Stabilization Settings and Library Imports ---
import os
# This setting is placed before importing scientific computing libraries to ensure stability.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0") 
import json
import logging
import warnings
from pathlib import Path

# Set Matplotlib backend to 'Agg' to allow saving figures in a non-GUI environment.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split

# --- Local Modules ---
from paths import DIR_MODELS, DIR_EXPLAINABILITY
from utils.data_loader import find_data_file, load_dataframe # <-- Import shared module

# --- Global Settings ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# English plots do not require special CJK font settings.
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'PingFang TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.dpi"] = 140

# --- Core Model Explainer Class ---
class ModelExplainer:
    """Encapsulates the complete explainability analysis workflow for a single model."""

    def __init__(self, target: str):
        self.target = target
        self.model_dir = DIR_MODELS / self.target
        self.output_dir = DIR_EXPLAINABILITY / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-load all necessary artifacts
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads the model, data, and metadata, recreating the validation set using a rigorous time-series split."""
        logging.info(f"===== [Loading] target = {self.target} =====")
        
        # 1. Load metadata
        card_path = self.model_dir / "dataset_card.json"
        schema_path = self.model_dir / "feature_schema.json"
        with open(card_path, 'r', encoding='utf-8') as f: self.card = json.load(f)
        with open(schema_path, 'r', encoding='utf-8') as f: self.schema = json.load(f)
        
        self.feature_cols = self.schema['features']
        self.seed = self.card['random_seed']
        self.data_path = Path(self.card['source_file'])
        
        # 2. Load model
        model_path = self.model_dir / "best_model.joblib"
        self.model = joblib.load(model_path)
        
        # 3. [CORE FIX] Load data and recreate the "time-series" train/test split
        df_all = load_dataframe(self.data_path)
        df_valid = df_all[df_all[self.target].notna()].copy()
        if self.target == 'rmssd_post': df_valid = df_valid[df_valid[self.target] > 0]

        # If the dummy_flag column exists, filter for real data.
        # Otherwise, assume all data is real.
        if "dummy_flag" in df_valid.columns:
            real_df = df_valid[df_valid["dummy_flag"] == 0].copy()
        else:
            real_df = df_valid.copy()
        if 'end_apnea_time' not in real_df.columns:
            raise ValueError("Missing 'end_apnea_time' column, cannot perform time-series split.")
        
        real_df['end_apnea_time'] = pd.to_datetime(real_df['end_apnea_time'])
        real_df_sorted = real_df.sort_values("end_apnea_time")
        
        split_idx = int(len(real_df_sorted) * 0.7)
        train_df = real_df_sorted.iloc[:split_idx]
        test_df = real_df_sorted.iloc[split_idx:]
        
        self.X_train = train_df[self.feature_cols]
        self.y_train = train_df[self.target]
        self.X_test = test_df[self.feature_cols]
        self.y_test = test_df[self.target]

        # For SHAP/LIME, we still need a complete, imputed version of X
        X_full = df_valid[self.feature_cols]
        self.X_imputed = self.model.named_steps["imputer"].transform(X_full)
        # Note: SHAP is best explained on the full dataset, while Permutation Importance uses the test set.

    def run_all_explanations(self):
        """Sequentially executes all explainability analysis methods."""
        logging.info("Generating SHAP analysis plots...")
        self.generate_shap_plots()

        logging.info("Generating LIME analysis plot...")
        self.generate_lime_plot()

        logging.info("Generating Permutation Importance analysis plot...")
        self.generate_permutation_importance()

        logging.info("Generating Partial Dependence Plots (PDP)...")
        self.generate_pdp()
        logging.info(f"===== [Complete] All analysis plots for {self.target} have been saved to {self.output_dir} =====")

    def _save_fig(self, filename: str):
        """Unified function for saving plots."""
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logging.info(f"Plot saved: {path}")

    def generate_shap_plots(self):
        try:
            explainer = shap.Explainer(self.model.named_steps["model"], self.X_imputed)
            shap_values = explainer(self.X_imputed)
            
            plt.figure()
            shap.summary_plot(shap_values, self.X_imputed, feature_names=self.feature_cols, show=False)
            self._save_fig(f"shap_summary_{self.target}.png")
            
            plt.figure()
            shap.summary_plot(shap_values, feature_names=self.feature_cols, plot_type="bar", show=False)
            self._save_fig(f"shap_bar_{self.target}.png")
        except Exception as e:
            logging.warning(f"SHAP analysis failed: {e}")

    def generate_lime_plot(self):
        try:
            X_train_imputed = self.model.named_steps["imputer"].transform(self.X_train)
            explainer = LimeTabularExplainer(
                training_data=X_train_imputed, feature_names=self.feature_cols,
                mode="regression", random_state=self.seed
            )
            # Explain a sample from the middle of the test set
            mid_idx = len(self.X_test) // 2
            instance = self.model.named_steps["imputer"].transform(self.X_test.iloc[mid_idx:mid_idx+1])[0]
            
            exp = explainer.explain_instance(
                instance, self.model.named_steps["model"].predict, 
                num_features=min(10, len(self.feature_cols))
            )
            fig = exp.as_pyplot_figure()
            fig.suptitle(f"LIME Explanation (Single Instance) - {self.target}")
            self._save_fig(f"lime_instance_{self.target}.png")
        except Exception as e:
            logging.warning(f"LIME analysis failed: {e}")

    def generate_permutation_importance(self):
        try:
            X_test_imputed = self.model.named_steps["imputer"].transform(self.X_test)
            result = permutation_importance(
                self.model, X_test_imputed, self.y_test, 
                n_repeats=10, random_state=self.seed, n_jobs=1
            )
            importances = pd.Series(result.importances_mean, index=self.feature_cols).sort_values()
            
            plt.figure(figsize=(10, 8))
            importances.plot(kind="barh")
            plt.title(f"Permutation Importance - {self.target}")
            plt.xlabel("Performance Decrease")
            self._save_fig(f"permutation_importance_{self.target}.png")
        except Exception as e:
            logging.warning(f"Permutation Importance analysis failed: {e}")
            
    def generate_pdp(self):
        """
        Generates and saves Partial Dependence Plots (PDP).
        This function is now robust enough to handle models with either 
        .feature_importances_ (tree-based) or .coef_ (linear) attributes.
        """
        try:
            # Extract the final regressor model from the pipeline
            model_regressor = self.model.named_steps.get("model")
            
            # --- [CORE FIX] ---
            # Check for feature importances (e.g., XGBoost) or coefficients (e.g., Ridge) 
            # to determine feature importance.
            if hasattr(model_regressor, 'feature_importances_'):
                # For tree-based models, use feature_importances_ directly
                importances = pd.Series(model_regressor.feature_importances_, index=self.feature_cols)
            elif hasattr(model_regressor, 'coef_'):
                # For linear models, use the absolute value of coefficients as the importance metric
                importances = pd.Series(np.abs(model_regressor.coef_), index=self.feature_cols)
            else:
                # If neither attribute exists, log a warning and skip PDP analysis
                logging.warning(f"Model for {self.target} has neither 'feature_importances_' nor 'coef_'. Skipping PDP.")
                return

            # Sort by importance and select the top 4 features for plotting
            top_features = importances.sort_values(ascending=False).head(4).index.tolist()

            # Use scikit-learn's built-in functionality to plot PDP
            PartialDependenceDisplay.from_estimator(
                self.model, 
                self.X_imputed, 
                top_features, 
                feature_names=self.feature_cols, 
                n_cols=2
            )
            
            # Configure and save the figure
            fig = plt.gcf()
            fig.set_size_inches(12, 8)
            fig.suptitle(f"Partial Dependence Plots - {self.target}", fontsize=16)
            plt.subplots_adjust(top=0.9)
            self._save_fig(f"pdp_top4_{self.target}.png")
            
        except Exception as e:
            logging.warning(f"PDP analysis failed: {e}")

# --- Main Execution Flow ---
def main():
    """Main function to create an Explainer for each target and run the analysis."""
    targets_str = os.environ.get("TARGETS", "ERS,rmssd_post").strip()
    targets = [t.strip() for t in targets_str.split(",") if t.strip()]
    
    for target in targets:
        try:
            explainer = ModelExplainer(target)
            explainer.run_all_explanations()
        except FileNotFoundError as e:
            logging.error(f"Could not analyze target '{target}' due to missing files: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while analyzing target '{target}': {e}", exc_info=True)

if __name__ == "__main__":
    main()