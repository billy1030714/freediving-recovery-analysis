#
# 07_visualize.py – GitHub Optimized Version
#
import json
import logging
import os
import warnings
from pathlib import Path
from datetime import timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from paths import DIR_MODELS, DIR_PREDICTIONS, DIR_REPORT, DIR_CONVERTED, get_daily_path, PROJECT_ROOT
from utils.data_loader import find_data_file, load_dataframe

# --- Unified visualization style settings ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.bbox'] = 'tight'

# --- Global Settings ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
PALETTE_DEFAULT = 'colorblind'

def discover_available_targets() -> list:
    """Automatically discover which targets have been trained by checking for dataset_card.json files."""
    available_targets = []
    
    if DIR_MODELS.exists():
        for target_dir in DIR_MODELS.iterdir():
            if target_dir.is_dir():
                card_path = target_dir / "dataset_card.json"
                if card_path.exists():
                    available_targets.append(target_dir.name)
                    logging.info(f"Found trained model for target: {target_dir.name}")
    
    return sorted(available_targets)

def should_generate_detailed_plots(target: str) -> bool:
    """
    COMPUTATIONAL EFFICIENCY OPTIMIZATION:
    
    Only generates resource-intensive detailed visualizations for models
    with meaningful predictive performance (R² > 0.1) or for ERS regardless
    of track type.
    
    This prevents wasting computational resources on extensive plotting for
    models that demonstrate the negative results in our research track,
    while ensuring the successful ERS algorithm validation is always visualized.
    """
    try:
        card_path = DIR_MODELS / target / "dataset_card.json"
        if not card_path.exists():
            return False
            
        with open(card_path, 'r', encoding='utf-8') as f:
            card = json.load(f)
        
        r2_score = card.get("evaluation_metrics", {}).get("r2", -999)
        
        # Generate detailed plots only for models with R² > 0.1 or for ERS (regardless of track)
        return r2_score > 0.1 or target == "ERS"
        
    except Exception as e:
        logging.warning(f"Could not determine performance for {target}: {e}")
        return False

class Visualizer:
    """Generates visualizations for high-performing models only."""
    
    def __init__(self, target: str):
        self.target = target
        self.output_dir = DIR_REPORT / "figures" / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"===== [Visualizing] target = {self.target} | Output to: {self.output_dir} =====")
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads all necessary artifacts and recreates the validation set using a rigorous time-series split."""
        # Model-related artifacts
        model_dir = DIR_MODELS / self.target
        self.model = joblib.load(model_dir / "best_model.joblib")
        with open(model_dir / "dataset_card.json", 'r', encoding='utf-8') as f: 
            self.card = json.load(f)
        with open(model_dir / "feature_schema.json", 'r', encoding='utf-8') as f: 
            self.schema = json.load(f)
        
        self.feature_cols = self.schema['features']
        self.seed = self.card['random_seed']
        self.best_model_name = self.card['best_model_name']
        self.r2_score = self.card.get("evaluation_metrics", {}).get("r2", 0)
        
        # Prediction-related artifacts (if available)
        preds_path = DIR_PREDICTIONS / self.target / "preds.parquet"
        if preds_path.exists():
            self.preds_df = pd.read_parquet(preds_path)
            self._setup_prediction_data()
        else:
            logging.warning(f"No prediction data found for {self.target}. Skipping prediction-based plots.")
            self.preds_df = None

    def _setup_prediction_data(self):
        """Set up prediction data for plotting."""
        # Original data
        source_data_path = Path(self.card['source_file'])
        self.data_path = source_data_path 
        df_all = load_dataframe(self.data_path)
        
        # Recreate the strict Time-Series Split
        df_valid = df_all[df_all[self.target].notna()].copy()
        if self.target == 'rmssd_post': 
            df_valid = df_valid[df_valid[self.target] > 0]
        
        if "dummy_flag" in df_valid.columns:
            real_df = df_valid[df_valid["dummy_flag"] == 0].copy()
        else:
            real_df = df_valid.copy()
            
        if 'end_apnea_time' not in real_df.columns:
            raise ValueError("Missing 'end_apnea_time' column in data, cannot perform time-series split.")
        
        real_df['end_apnea_time'] = pd.to_datetime(real_df['end_apnea_time'])
        real_df_sorted = real_df.sort_values("end_apnea_time")
        
        split_idx = int(len(real_df_sorted) * 0.7)
        test_df = real_df_sorted.iloc[split_idx:]
        
        if "dummy_flag" in self.preds_df.columns:
            preds_on_real = self.preds_df[self.preds_df['dummy_flag'] == 0]
        else:
            preds_on_real = self.preds_df
            
        test_df_with_preds = test_df.merge(preds_on_real, on='row_id', how='left')
        
        self.y_pred = test_df_with_preds['y_pred'].values
        self.y_test = test_df_with_preds[self.target].values

    def _save_fig(self, filename: str, title: str = ""):
        """Unified function for saving plots."""
        if title:
            plt.title(title, fontsize=16)
        path = self.output_dir / filename
        plt.savefig(path, dpi=300, bbox_inches='tight') 
        plt.close()
        logging.info(f"Figure saved: {filename}")

    def run_essential_visualizations(self):
        """Run only essential visualizations for all models."""
        self.plot_model_leaderboard()
        
        # Only generate detailed plots for high-performing models
        if should_generate_detailed_plots(self.target):
            self.run_detailed_visualizations()
        else:
            logging.info(f"Skipping detailed plots for {self.target} (R² = {self.r2_score:.4f})")

    def run_detailed_visualizations(self):
        """Run detailed visualizations for high-performing models only."""
        if self.preds_df is not None:
            self.plot_predicted_vs_actual()
            self.plot_calibration_curve()
            self.plot_bland_altman()
        
        self.plot_feature_importance()
        self.plot_shap_summary()
        
        # ERS-specific plots
        if self.target == 'ERS':
            self.plot_typical_recovery_curves()

    def plot_predicted_vs_actual(self):
        from sklearn.metrics import r2_score
        plt.figure(figsize=(8, 8))
        
        valid_indices = ~np.isnan(self.y_test) & ~np.isnan(self.y_pred)
        r2 = r2_score(self.y_test[valid_indices], self.y_pred[valid_indices])
        
        sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.7, s=60, edgecolor='w', label='Data Points')
        
        all_values = np.concatenate([self.y_test[valid_indices], self.y_pred[valid_indices]])
        line_lims = [np.nanmin(all_values), np.nanmax(all_values)]
        
        plt.plot(line_lims, line_lims, 'r--', label='Perfect Prediction (y=x)')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        title = f'Predicted vs. Actual Values\n(Test Set R² = {r2:.4f})'
        self._save_fig("02_predicted_vs_actual.png", title)

    def plot_calibration_curve(self):
        plt.figure(figsize=(8, 8))
        cal_df = pd.DataFrame({'y_true': self.y_test, 'y_pred': self.y_pred}).dropna()
        if cal_df.empty: 
            return
            
        cal_df['pred_bin'] = pd.cut(cal_df['y_pred'], bins=min(10, len(cal_df)//2))
        bin_means = cal_df.groupby('pred_bin', as_index=False, observed=True).mean()
        
        plt.plot(bin_means['y_pred'], bin_means['y_true'], 'o-', label='Model Calibration')
        plt.plot([cal_df['y_true'].min(), cal_df['y_true'].max()], 
                [cal_df['y_true'].min(), cal_df['y_true'].max()], 'r--', label='Perfect Calibration') 
        
        plt.xlabel('Mean Predicted Value (in Bin)')
        plt.ylabel('Mean Actual Value (in Bin)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        self._save_fig("03_calibration_curve.png", f'Calibration Curve ({self.target})')

    def plot_bland_altman(self):
        plt.figure(figsize=(10, 6))
        diffs = self.y_pred - self.y_test
        avg = (self.y_pred + self.y_test) / 2
        
        valid_indices = ~np.isnan(diffs) & ~np.isnan(avg)
        diffs, avg = diffs[valid_indices], avg[valid_indices]
        if len(diffs) == 0: 
            return

        mean_diff, std_diff = np.mean(diffs), np.std(diffs, ddof=1)
        lower_limit, upper_limit = mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff
        
        sns.scatterplot(x=avg, y=diffs, alpha=0.6)
        plt.axhline(mean_diff, color='r', linestyle='--')
        plt.axhline(upper_limit, color='g', linestyle='--')
        plt.axhline(lower_limit, color='g', linestyle='--')
        
        ax = plt.gca()
        text_x_pos = ax.get_xlim()[1] 
        ax.text(text_x_pos, mean_diff, f'  Mean Bias: {mean_diff:.3f}', va='center', ha='left', color='r')
        ax.text(text_x_pos, upper_limit, f'  95% Limits of Agreement', va='center', ha='left', color='g')
        ax.text(text_x_pos, lower_limit, f'  95% Limits of Agreement', va='center', ha='left', color='g')

        plt.xlabel('Average of (Prediction + Actual)')
        plt.ylabel('Difference (Prediction - Actual)')
        plt.grid(True)
        self._save_fig("04_bland_altman_plot.png", f'Bland-Altman Plot ({self.target})')

    def plot_feature_importance(self):
        model_regressor = self.model.named_steps.get('model')
        if not hasattr(model_regressor, 'feature_importances_') and not hasattr(model_regressor, 'coef_'):
            logging.warning(f"Model for {self.target} has neither feature_importances_ nor coef_. Skipping plot.")
            return
        
        if hasattr(model_regressor, 'feature_importances_'):
            importances = pd.Series(model_regressor.feature_importances_, index=self.feature_cols).sort_values(ascending=False).head(10)
            palette, xlabel, title = PALETTE_DEFAULT, 'Importance Score', f'{self.best_model_name.upper()} Feature Importance'
        else: # ridge coef_
            coefs = pd.Series(model_regressor.coef_, index=self.feature_cols)
            importances = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)
            palette, xlabel, title = 'coolwarm', 'Coefficient Magnitude', f'{self.best_model_name.upper()} Coefficient Magnitude'

        plt.figure(figsize=(10, 8))
        sns.barplot(x=importances.values, y=importances.index, palette=palette, hue=importances.index, legend=False)
        plt.xlabel(xlabel)
        plt.ylabel('Features')
        self._save_fig(f"05_feature_importance_{self.best_model_name}.png", title)

    def plot_shap_summary(self):
        try:
            source_data_path = Path(self.card['source_file'])
            df_all = load_dataframe(source_data_path)
            X_full = df_all[self.feature_cols].dropna()
            
            if len(X_full) == 0:
                logging.warning(f"No valid data for SHAP analysis for {self.target}")
                return
                
            X_full_imputed = self.model.named_steps["imputer"].transform(X_full)
            explainer = shap.Explainer(self.model.named_steps["model"], X_full_imputed[:50])  # Limit for speed
            shap_values = explainer(X_full_imputed[:50])
            
            plt.figure()
            shap.summary_plot(shap_values, X_full.iloc[:50], feature_names=self.feature_cols, show=False)
            
            plt.gca().set_title('') 
            self._save_fig(f"06_shap_summary.png", f'SHAP Summary Plot ({self.target})')
            
        except Exception as e:
            logging.warning(f"SHAP analysis failed for target {self.target}: {e}")

    def plot_model_leaderboard(self):
        """Always generated: Plots a bar chart for the model competition leaderboard."""
        board_path = DIR_MODELS / self.target / "leaderboard.json"
        if not board_path.exists():
            logging.warning(f"Leaderboard file not found: {board_path}")
            return
            
        board_df = pd.read_json(board_path).sort_values("r2", ascending=False)
        board_df['name'] = board_df['name'].str.upper() 
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        sns.barplot(x="r2", y="name", data=board_df, ax=ax1, palette="viridis", hue="name", legend=False)
        ax1.set_title(f'R-squared (R²) - Higher is Better')
        ax1.set_xlabel('R² Score')
        ax1.set_ylabel('Model')
        ax1.set_xlim(left=max(-1, board_df['r2'].min() - 0.1))

        sns.barplot(x="mae", y="name", data=board_df, ax=ax2, palette="plasma", hue="name", legend=False)
        ax2.set_title(f'Mean Absolute Error (MAE) - Lower is Better')
        ax2.set_xlabel('MAE')
        ax2.set_ylabel('')
        
        fig.suptitle(f'Model Comparison for {self.target}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_fig(f"99_model_leaderboard.png")

    def plot_typical_recovery_curves(self):
        """ERS-specific: Plots typical recovery curves for high, medium, and low ERS scores."""
        try:
            source_data_path = Path(self.card['source_file'])
            df_full = load_dataframe(source_data_path)
            
            if "dummy_flag" in df_full.columns:
                df_real = df_full[df_full["dummy_flag"] == 0].dropna(subset=['ERS'])
            else:
                df_real = df_full.dropna(subset=['ERS'])
                
            if len(df_real) < 3: 
                return

            low_ers_event = df_real.loc[df_real['ERS'].idxmin()]
            high_ers_event = df_real.loc[df_real['ERS'].idxmax()]
            median_ers_event = df_real.iloc[(df_real['ERS'] - df_real['ERS'].median()).abs().argsort()[:1]]
            
            events_to_plot = {
                'High ERS': high_ers_event,
                'Median ERS': median_ers_event.iloc[0],
                'Low ERS': low_ers_event
            }
            
            plt.figure(figsize=(12, 7))
            
            for label, event in events_to_plot.items():
                try:
                    date_obj = pd.to_datetime(event['date'], format='%Y%m%d').date()
                    hr_path = get_daily_path(DIR_CONVERTED, 'hr', date_obj, '.csv')
                    
                    if not hr_path.exists():
                        continue
                        
                    hr_df = pd.read_csv(hr_path, parse_dates=['Time'])
                    t0 = pd.to_datetime(event['end_apnea_time'])
                    recovery_df = hr_df[(hr_df['Time'] >= t0) & (hr_df['Time'] <= t0 + timedelta(seconds=120))]
                    
                    if len(recovery_df) > 0:
                        time_since_apnea = (recovery_df['Time'] - t0).dt.total_seconds()
                        plt.plot(time_since_apnea, recovery_df['HR'], 
                               label=f"{label} (ERS: {event['ERS']:.2f})", alpha=0.8)

                except Exception as e:
                    logging.warning(f"Could not plot recovery curve for event {event['row_id']}: {e}")

            plt.xlabel('Time Since Apnea End (seconds)')
            plt.ylabel('Heart Rate (bpm)')
            plt.title('Typical Heart Rate Recovery Curves')
            plt.legend()
            plt.grid(True)
            self._save_fig("98_typical_recovery_curves.png")
            
        except Exception as e:
            logging.warning(f"Could not generate recovery curves: {e}")

def generate_final_summary_plot(targets: list):
    """Generates a single plot comparing the best R^2 scores of all targets."""
    logging.info("===== [Generating Final Summary] Comparing all models =====")
    results = []
    
    for target in targets:
        leaderboard_path = DIR_MODELS / target / "leaderboard.json"
        if not leaderboard_path.exists():
            logging.warning(f"Leaderboard not found for target '{target}', skipping.")
            continue
            
        with open(leaderboard_path, 'r', encoding='utf-8') as f:
            leaderboard = json.load(f)
        
        best_run = max(leaderboard, key=lambda x: x.get('r2', -float('inf')))
        results.append({"Target": target, "Best R² Score": best_run['r2']})

    if not results:
        logging.error("No leaderboard results found for any target. Cannot generate summary plot.")
        return

    df = pd.DataFrame(results).sort_values("Best R² Score", ascending=False)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    palette = sns.color_palette("coolwarm_r", n_colors=len(df))
    sns.barplot(x="Best R² Score", y="Target", data=df, palette=palette, hue="Target", dodge=False, legend=False, ax=ax)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_title("Final Model Performance Comparison", fontsize=18, pad=20)
    ax.set_xlabel("Best R² Score on Test Set (Higher is Better)", fontsize=12)
    ax.set_ylabel("Target Metric", fontsize=12)
    
    # Add value labels
    for p in ax.patches:
        width = p.get_width()
        x_pos = width + 0.01 if width >= 0 else width - 0.01
        ha = 'left' if width >= 0 else 'right'
        ax.text(x_pos, p.get_y() + p.get_height() / 2., f'{width:.3f}', va='center', ha=ha)
    
    # Adjust x-axis limits
    min_r2, max_r2 = df["Best R² Score"].min(), df["Best R² Score"].max()
    ax.set_xlim(min(min_r2 * 1.3, -0.2), max(max_r2 * 1.3, 0.4))
    
    plt.tight_layout()
    output_path = DIR_REPORT / "figures" / "00_final_summary_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ Final summary plot saved to: {output_path}")

def main():
    """Main function to generate visualizations for available targets."""
    available_targets = discover_available_targets()
    
    if not available_targets:
        logging.error("No trained models found. Please run models training first.")
        return
    
    logging.info(f"Found {len(available_targets)} trained models: {available_targets}")
    
    # Generate visualizations for each target
    for target in available_targets:
        try:
            visualizer = Visualizer(target)
            visualizer.run_essential_visualizations()
        except FileNotFoundError as e:
            logging.error(f"Could not generate plots for target '{target}' due to missing files: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while generating plots for target '{target}': {e}", exc_info=True)
    
    # Generate final comparison chart
    generate_final_summary_plot(available_targets)
    
    logging.info("✅ Visualization generation complete!")

if __name__ == "__main__":
    main()