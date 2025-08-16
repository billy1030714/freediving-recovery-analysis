"""
visualize_07.py – 視覺化圖表生成腳本

本腳本為整個管線的視覺化核心。

核心流程：
1.  載入由 04_models 和 06_predict 產出的所有相關檔案
    (模型, 數據集, 預測結果, 元數據)。
2.  針對每一個目標 (ERS, rmssd_post)，生成一套完整的、用於報告的
    高品質視覺化圖表，並存為獨立的 .png 檔案。
3.  產出的圖表包括：
    -   數據集組成圖 (Dataset Composition)
    -   預測值 vs. 真實值散佈圖 (Predicted vs. Actual)
    -   可靠度曲線 (Calibration Curve)
    -   Bland-Altman 一致性分析圖
    -   特徵重要性圖 (Feature Importance)
    -   SHAP Summary Plot
"""
# (請貼在 visualize_07.py 的最頂部)

import json
import logging
import warnings
from pathlib import Path
from datetime import timedelta # <-- 【修正】補上遺漏的導入

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split

# <-- 【修正】從 paths 導入所有需要的變數
from paths import DIR_MODELS, DIR_PREDICTIONS, DIR_REPORT, DIR_CONVERTED, get_daily_path 
from utils.data_loader import find_data_file, load_dataframe

# --- 【修正】統一且唯一的視覺化風格設定 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.bbox'] = 'tight'

# --- 全域設定 ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class Visualizer:
    """為單一目標生成一套完整的分析圖表。"""

    def __init__(self, target: str):
        self.target = target
        self.output_dir = DIR_REPORT / "figures" / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"===== [視覺化] target = {self.target} | 輸出至: {self.output_dir} =====")
        self._load_artifacts()

    # (請貼在 visualize_07.py 的 Visualizer class 中)

    def _load_artifacts(self):
        """載入所有必要的產出物，並以嚴謹的時間序列方式重現驗證集。"""
        # 模型相關
        model_dir = DIR_MODELS / self.target
        self.model = joblib.load(model_dir / "best_model.joblib")
        with open(model_dir / "dataset_card.json", 'r', encoding='utf-8') as f: self.card = json.load(f)
        with open(model_dir / "feature_schema.json", 'r', encoding='utf-8') as f: self.schema = json.load(f)
        
        self.feature_cols = self.schema['features']
        self.seed = self.card['random_seed']
        self.best_model_name = self.card['best_model_name']
        
        # 預測相關
        preds_path = DIR_PREDICTIONS / self.target / "preds.parquet"
        self.preds_df = pd.read_parquet(preds_path)

        # 原始數據
        source_data_path = Path(self.card['source_file'])
        self.data_path = source_data_path 
        df_all = load_dataframe(self.data_path)
        
        # ---【核心修正】重現嚴格的時間序列切分 (Time-Series Split) ---
        df_valid = df_all[df_all[self.target].notna()].copy()
        if self.target == 'rmssd_post': df_valid = df_valid[df_valid[self.target] > 0]
        
        real_df = df_valid[df_valid.get("dummy_flag", 0) == 0].copy()
        if 'end_apnea_time' not in real_df.columns:
            raise ValueError("數據中缺少 'end_apnea_time' 欄位，無法進行時間序列切分。")
        
        real_df['end_apnea_time'] = pd.to_datetime(real_df['end_apnea_time'])
        real_df_sorted = real_df.sort_values("end_apnea_time")
        
        split_idx = int(len(real_df_sorted) * 0.7)
        train_df = real_df_sorted.iloc[:split_idx]
        test_df = real_df_sorted.iloc[split_idx:]
        
        self.X_test = test_df[self.feature_cols]
        self.y_test = test_df[self.target]
        
        # 我們直接用儲存的預測結果來確保一致性，而不是重新預測
        # 但為了確保 y_pred 和 y_test 的 index 對得上，我們需要合併一下
        preds_on_real = self.preds_df[self.preds_df['dummy_flag'] == 0]
        test_df_with_preds = test_df.merge(preds_on_real, on='row_id', how='left')
        
        self.y_pred = test_df_with_preds['y_pred'].values
        # 重新校準 y_test，確保順序與 y_pred 完全一致
        self.y_test = test_df_with_preds[self.target].values
        
        logging.info("所有必要的產出物已成功載入，並以時間序列方式重現了驗證集。")

    def _save_fig(self, filename: str, title: str):
        """統一的存圖函數。"""
        if title:
            plt.title(title, fontsize=16)
        path = self.output_dir / filename
        # plt.tight_layout() # 移到 run_all_visualizations 中，在 suptitle 後調用
        plt.savefig(path, dpi=600) # <-- 【修正】DPI 提升至 600
        plt.close()
        logging.info(f"圖表已儲存: {filename}")

    def run_all_visualizations(self):
        """執行所有圖表的生成。"""
        self.plot_dataset_composition()
        self.plot_predicted_vs_actual()
        self.plot_calibration_curve()
        self.plot_bland_altman()
        self.plot_feature_importance()
        self.plot_shap_summary()
        self.plot_model_leaderboard()
        self.plot_typical_recovery_curves()

    # (請貼在 visualize_07.py 的 Visualizer class 中)
    def plot_dataset_composition(self):
        df_full = load_dataframe(self.data_path)
        real_count = (df_full.get('dummy_flag', 0) == 0).sum()
        dummy_count = (df_full.get('dummy_flag', 0) == 1).sum()
        total = real_count + dummy_count
        if total == 0: return

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=['Real Samples', 'Dummy Samples'], y=[real_count, dummy_count], palette=['#347893', '#77A9BE'])
        
        # ---【新增】為長條圖加上比例與數值 ---
        for p in ax.patches:
            height = p.get_height()
            percentage = f'{100 * height / total:.1f}%'
            ax.annotate(f'{int(height)}\n({percentage})',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, -20), textcoords='offset points',
                        color='white', fontsize=14, weight='bold')

        ax.set_ylabel('Count')
        self._save_fig("01_dataset_composition.png", f'Dataset Composition ({self.target})')
        
    def plot_predicted_vs_actual(self):
        from sklearn.metrics import r2_score # 局部導入
        plt.figure(figsize=(8, 8))
        
        # ---【修正】在函數內部重新計算 R2，確保圖文一致 ---
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
        cal_df = pd.DataFrame({'y_true': self.y_test, 'y_pred': self.y_pred})
        cal_df['pred_bin'] = pd.cut(cal_df['y_pred'], bins=10)
        bin_means = cal_df.groupby('pred_bin', as_index=False, observed=True).mean()
        
        plt.plot(bin_means['y_pred'], bin_means['y_true'], 'o-', label='Model Calibration')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', label='Perfect Calibration') # <--【修正】補上紅色虛線
        
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
        
        # 確保在計算時忽略 NaN 值
        valid_indices = ~np.isnan(diffs) & ~np.isnan(avg)
        diffs, avg = diffs[valid_indices], avg[valid_indices]
        if len(diffs) == 0: return

        mean_diff, std_diff = np.mean(diffs), np.std(diffs, ddof=1)
        lower_limit, upper_limit = mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff
        
        sns.scatterplot(x=avg, y=diffs, alpha=0.6)
        plt.axhline(mean_diff, color='r', linestyle='--')
        plt.axhline(upper_limit, color='g', linestyle='--')
        plt.axhline(lower_limit, color='g', linestyle='--')
        
        # ---【修正】使用 ax.text 進行精準標註 ---
        ax = plt.gca()
        text_x_pos = ax.get_xlim()[1] # 將文字放在圖表右側
        ax.text(text_x_pos, mean_diff, f'  Mean Bias: {mean_diff:.3f}', va='center', ha='left', color='r')
        ax.text(text_x_pos, upper_limit, f'  95% Limits of Agreement', va='center', ha='left', color='g')
        ax.text(text_x_pos, lower_limit, f'  95% Limits of Agreement', va='center', ha='left', color='g')

        plt.xlabel('Average of (Prediction + Actual)')
        plt.ylabel('Difference (Prediction - Actual)')
        plt.grid(True)
        self._save_fig("04_bland_altman_plot.png", f'Bland-Altman Plot ({self.target})')

    def plot_feature_importance(self):
        if self.best_model_name not in ['xgb', 'rf', 'ridge']: return
        
        model = self.model.named_steps['model']
        
        if self.best_model_name in ['xgb', 'rf']:
            importances = pd.Series(model.feature_importances_, index=self.feature_cols).sort_values(ascending=False).head(10)
            palette = 'viridis'
            xlabel = 'Importance Score'
            title = f'{self.best_model_name.upper()} Feature Importance'
        else: # ridge
            coefs = pd.Series(model.coef_, index=self.feature_cols)
            importances = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)
            palette = 'coolwarm'
            xlabel = 'Coefficient Magnitude'
            title = f'{self.best_model_name.upper()} Coefficient Magnitude'

        plt.figure(figsize=(10, 8))
        # ---【修正】簡化 barplot 調用，移除不標準的 hue 參數 ---
        sns.barplot(x=importances.values, y=importances.index, palette=palette)
        plt.xlabel(xlabel)
        plt.ylabel('Features')
        self._save_fig(f"05_feature_importance_{self.best_model_name}.png", title)

    def plot_shap_summary(self):
        try:
            X_full = load_dataframe(self.data_path)[self.feature_cols]
            X_full_imputed = self.model.named_steps["imputer"].transform(X_full)
            explainer = shap.Explainer(self.model.named_steps["model"], X_full_imputed)
            shap_values = explainer(X_full_imputed)
            
            plt.figure()
            # ---【修正】明確傳遞 feature_names 參數 ---
            shap.summary_plot(shap_values, X_full, feature_names=self.feature_cols, show=False)
            
            # 移除自動生成的標題，交給 _save_fig 統一處理
            plt.gca().set_title('') 
            
            self._save_fig(f"06_shap_summary.png", f'SHAP Summary Plot ({self.target})')
        except Exception as e:
            logging.warning(f"SHAP 分析失敗 for target {self.target}: {e}")

    def plot_model_leaderboard(self):
        """新增：繪製模型競賽排行榜的長條圖。"""
        board_path = DIR_MODELS / self.target / "leaderboard.json"
        if not board_path.exists():
            logging.warning(f"找不到 Leaderboard 檔案: {board_path}")
            return
            
        board_df = pd.read_json(board_path).sort_values("r2", ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        sns.barplot(x="r2", y="name", data=board_df, ax=ax1, palette="viridis")
        ax1.set_title(f'R-squared (R²) - Higher is Better')
        ax1.set_xlabel('R² Score')
        ax1.set_ylabel('Model')
        ax1.set_xlim(left=max(-1, board_df['r2'].min() - 0.1))

        sns.barplot(x="mae", y="name", data=board_df, ax=ax2, palette="plasma")
        ax2.set_title(f'Mean Absolute Error (MAE) - Lower is Better')
        ax2.set_xlabel('MAE')
        ax2.set_ylabel('')
        
        fig.suptitle(f'Model Comparison for {self.target}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_fig(f"99_model_leaderboard.png", "") # 不再需要內部標題

    def plot_typical_recovery_curves(self):
        """新增：繪製典型恢復曲線圖 (高/中/低 ERS)。"""
        if self.target != 'ERS':
            return # 此圖僅為 ERS 目標繪製

        # 找出高/中/低 ERS 分數的事件
        df_full = load_dataframe(self.data_path)
        df_real = df_full[df_full.get("dummy_flag", 0) == 0].dropna(subset=['ERS'])
        if len(df_real) < 3: return

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
                hr_df = pd.read_csv(hr_path, parse_dates=['Time'])
                
                t0 = pd.to_datetime(event['end_apnea_time'])
                recovery_df = hr_df[(hr_df['Time'] >= t0) & (hr_df['Time'] <= t0 + timedelta(seconds=120))]
                
                time_since_apnea = (recovery_df['Time'] - t0).dt.total_seconds()
                plt.plot(time_since_apnea, recovery_df['HR'], label=f"{label} (ERS: {event['ERS']:.2f})", alpha=0.8)

            except Exception as e:
                logging.warning(f"無法繪製事件 {event['row_id']} 的恢復曲線: {e}")

        plt.xlabel('Time Since Apnea End (seconds)')
        plt.ylabel('Heart Rate (bpm)')
        plt.title('Typical Heart Rate Recovery Curves')
        plt.legend()
        plt.grid(True)
        self._save_fig("98_typical_recovery_curves.png", "")

def main():
    """主執行函數，為每個目標生成所有圖表。"""
    targets = ["ERS", "rmssd_post"]
    for target in targets:
        try:
            visualizer = Visualizer(target)
            visualizer.run_all_visualizations()
        except FileNotFoundError as e:
            logging.error(f"無法為目標 '{target}' 產生圖表，缺少必要檔案: {e}")
        except Exception as e:
            logging.error(f"為目標 '{target}' 產生圖表時發生未預期錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    main()