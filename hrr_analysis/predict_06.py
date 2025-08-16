"""
predict_06.py – 模型預測與解釋報告產生腳本 (API 更新最終版)

【本次更新】:
- 修正 scikit-learn 的 FutureWarning，改用最新的 `root_mean_squared_error` 函數。
"""
# --- 穩定化設定與導入函式庫 ---
import os
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
import json, logging, warnings, argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib, numpy as np, pandas as pd, shap
from lime.lime_tabular import LimeTabularExplainer
# --- 【修正】導入新的 RMSE 函數 ---
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import percentileofscore

# --- 本地模組 ---
from paths import DIR_MODELS, DIR_PREDICTIONS
from utils.data_loader import find_data_file, load_dataframe

# --- 全域設定 ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- 核心類別 (Predictor 中的 save_results 方法有修改) ---
class Predictor:
    """專責執行預測任務的類別。"""
    def __init__(self, target: str):
        self.target = target; self.model_dir = DIR_MODELS / self.target
        self.output_dir = DIR_PREDICTIONS / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipe, self.features, self.df = None, None, None
        self.target_distribution = []
        
    def load_artifacts(self):
        logging.info(f"===== [預測階段] target = {self.target} =====")
        model_path = self.model_dir / "best_model.joblib"
        schema_path = self.model_dir / "feature_schema.json"
        dist_path = self.model_dir / "target_distribution.json"
        
        self.pipe = joblib.load(model_path)
        with open(schema_path, 'r', encoding='utf-8') as f: self.features = json.load(f)['features']
        
        if dist_path.exists():
            with open(dist_path, 'r', encoding='utf-8') as f: self.target_distribution = json.load(f)['training_data_distribution']
            logging.info(f"成功載入含有 {len(self.target_distribution)} 筆數據的目標分佈檔案。")
        else:
            logging.warning(f"找不到目標分佈檔案: {dist_path}，將無法計算百分位排名。")

        data_path = find_data_file()
        self.df = load_dataframe(data_path)
        logging.info(f"成功載入模型、特徵與來自 {data_path.name} 的數據。")

    def run_predictions(self) -> pd.DataFrame:
        X = self.df[[col for col in self.features if col in self.df.columns]]
        self.df['y_pred'] = self.pipe.predict(X)
        if self.target in self.df.columns:
            self.df['y_true'] = self.df[self.target]
            self.df['abs_err'] = abs(self.df['y_pred'] - self.df['y_true'])
        if self.target_distribution and self.target == "ERS":
            self.df['percentile_rank'] = self.df['y_pred'].apply(lambda x: percentileofscore(self.target_distribution, x))
            logging.info("成功計算個人化百分位排名。")
        return self.df

    def save_results(self, df_with_preds: pd.DataFrame):
        cols_to_save = ["row_id", "dummy_flag", "y_pred", "y_true", "abs_err", "percentile_rank"]
        final_cols = [c for c in cols_to_save if c in df_with_preds.columns]
        out_df = df_with_preds[final_cols]
        out_df.to_csv(self.output_dir / "preds.csv", index=False)
        out_df.to_parquet(self.output_dir / "preds.parquet", index=False)
        logging.info(f"預測結果已儲存至: {self.output_dir / 'preds.csv'}")
        
        real_preds = out_df[out_df["dummy_flag"] == 0]
        metrics = {}
        if not real_preds.empty and 'y_true' in real_preds.columns:
            evaluation_df = real_preds.dropna(subset=['y_true'])
            if not evaluation_df.empty:
                metrics = {
                    "r2": r2_score(evaluation_df['y_true'], evaluation_df['y_pred']),
                    "mae": mean_absolute_error(evaluation_df['y_true'], evaluation_df['y_pred']),
                    # --- 【修正】使用最新的 scikit-learn 函數 ---
                    "rmse": root_mean_squared_error(evaluation_df['y_true'], evaluation_df['y_pred']),
                }
        
        data_profile = {"total_rows": len(df_with_preds), "real_rows": len(real_preds)}
        output_data = {"metrics_real_only": metrics, "data_profile": data_profile}
        with open(self.output_dir / "metrics.json", "w", encoding='utf-8') as f: json.dump(output_data, f, indent=2, ensure_ascii=False)
        logging.info(f"評估指標已儲存至: {self.output_dir / 'metrics.json'}")

    def execute(self) -> dict:
        self.load_artifacts(); df_with_preds = self.run_predictions(); self.save_results(df_with_preds)
        return {"target": self.target, "pipe": self.pipe, "features": self.features, "df_with_preds": df_with_preds}

class Explainer:
    """專責執行模型解釋任務的類別。"""
    def __init__(self, predictor_results: dict):
        self.target = predictor_results['target']
        self.pipe = predictor_results['pipe']
        self.features = predictor_results['features']
        self.df = predictor_results['df_with_preds']
        self.output_dir = DIR_PREDICTIONS / self.target / "explain"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"===== [解釋階段] target = {self.target} =====")
        
    def _save_fig(self, filename: str):
        path = self.output_dir / filename
        plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
        logging.info(f"解釋圖表已儲存: {path}")

    def run_all_explanations(self):
        X = self.df[self.features]
        X_imputed = self.pipe.named_steps["imputer"].transform(X)
        try:
            explainer = shap.Explainer(self.pipe.named_steps["model"], X_imputed)
            shap_values = explainer(X_imputed)
            plt.figure(); shap.summary_plot(shap_values, X, show=False)
            self._save_fig(f"shap_summary_{self.target}.png")
        except Exception as e: logging.warning(f"SHAP 分析失敗: {e}")
        try:
            real_preds = self.df.dropna(subset=['y_true'])[(self.df["dummy_flag"] == 0)]
            if not real_preds.empty and 'abs_err' in real_preds.columns:
                worst_case_idx = real_preds['abs_err'].idxmax()
                worst_case_loc = X.index.get_loc(worst_case_idx)
                worst_case_instance = X_imputed[worst_case_loc:worst_case_loc+1][0]
                explainer_lime = LimeTabularExplainer(X_imputed, feature_names=self.features, mode="regression", random_state=42)
                exp = explainer_lime.explain_instance(worst_case_instance, self.pipe.predict, num_features=10)
                fig = exp.as_pyplot_figure(); self._save_fig(f"lime_worst_case_{self.target}.png")
        except Exception as e: logging.warning(f"LIME 分析失敗: {e}")

def main(args):
    """主執行函數，為每個目標依序執行預測與解釋。"""
    targets = ["ERS", "rmssd_post"]
    for target in targets:
        try:
            predictor = Predictor(target)
            predictor_results = predictor.execute()
            if not args.fast:
                explainer = Explainer(predictor_results)
                explainer.run_all_explanations()
        except Exception as e:
            logging.error(f"為目標 '{target}' 處理時發生未預期錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="06_predict - 預測與解釋報告產生腳本")
    parser.add_argument("--fast", action="store_true", help="啟用快速模式，僅執行預測，跳過耗時的 SHAP/LIME 解釋")
    args = parser.parse_args()
    main(args)