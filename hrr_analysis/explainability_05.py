"""
explainability_05.py – 模型可解釋性分析 (XAI) 腳本

本腳本為流程的第五步，旨在深入剖析已訓練好的模型，理解其決策行為。

核心流程：
1.  載入由 04_models 產出的最佳模型 (`best_model.joblib`) 及其相關元數據。
2.  使用共用的 data_loader 讀取與訓練時相同的特徵數據。
3.  應用四種主流 XAI 技術，生成視覺化圖表並儲存：
    -   SHAP (SHapley Additive exPlanations): 提供全域與區域性的特徵貢獻度。
    -   LIME (Local Interpretable Model-agnostic Explanations): 解釋單一樣本的預測結果。
    -   Permutation Importance: 透過打亂特徵順序來評估特徵的重要性。
    -   Partial Dependence Plots (PDP): 觀察單一特徵與模型預測之間的關係。
"""
# --- 穩定化設定與導入函式庫 ---
import os
# 將此設定放在所有科學計算庫導入之前，以確保穩定
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0") 
import json
import logging
import warnings
from pathlib import Path

# 設定 Matplotlib 後端為 'Agg'，使其可以在沒有 GUI 的環境下執行並存檔
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

# --- 本地模組 ---
from paths import DIR_MODELS, DIR_EXPLAINABILITY
from utils.data_loader import find_data_file, load_dataframe # <-- 導入共用模組

# --- 全域設定 ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'PingFang TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.dpi"] = 140

# --- 模型解釋核心類 ---
class ModelExplainer:
    """封裝針對單一模型的完整可解釋性分析流程。"""

    def __init__(self, target: str):
        self.target = target
        self.model_dir = DIR_MODELS / self.target
        self.output_dir = DIR_EXPLAINABILITY / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 預先載入所有必要的產出物
        self._load_artifacts()

    def _load_artifacts(self):
        """載入模型、數據與元數據。"""
        logging.info(f"===== [載入中] target = {self.target} =====")
        
        # 1. 載入元數據
        card_path = self.model_dir / "dataset_card.json"
        schema_path = self.model_dir / "feature_schema.json"
        with open(card_path, 'r', encoding='utf-8') as f: self.card = json.load(f)
        with open(schema_path, 'r', encoding='utf-8') as f: self.schema = json.load(f)
        
        self.feature_cols = self.schema['features']
        self.seed = self.card['random_seed']
        self.data_path = Path(self.card['source_file'])
        
        # 2. 載入模型
        model_path = self.model_dir / "best_model.joblib"
        self.model = joblib.load(model_path)
        
        # 3. 載入數據並重現訓練/測試集
        df_all = load_dataframe(self.data_path)
        df_valid = df_all[df_all[self.target].notna()].copy()
        if self.target == 'rmssd_post': df_valid = df_valid[df_valid[self.target] > 0]
        
        X = df_valid[self.feature_cols]
        y = df_valid[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.X_imputed = self.model.named_steps["imputer"].transform(X)

    def run_all_explanations(self):
        """依序執行所有可解釋性分析方法。"""
        logging.info("開始生成 SHAP 分析圖...")
        self.generate_shap_plots()

        logging.info("開始生成 LIME 分析圖...")
        self.generate_lime_plot()

        logging.info("開始生成 Permutation Importance 分析圖...")
        self.generate_permutation_importance()

        logging.info("開始生成 PDP 分析圖...")
        self.generate_pdp()
        logging.info(f"===== [完成] {self.target} 的所有分析圖表已儲存至 {self.output_dir} =====")

    def _save_fig(self, filename: str):
        """統一的存圖函數。"""
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logging.info(f"圖表已儲存: {path}")

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
            logging.warning(f"SHAP 分析失敗: {e}")

    def generate_lime_plot(self):
        try:
            X_train_imputed = self.model.named_steps["imputer"].transform(self.X_train)
            explainer = LimeTabularExplainer(
                training_data=X_train_imputed, feature_names=self.feature_cols,
                mode="regression", random_state=self.seed
            )
            # 解釋測試集中間的一個樣本
            mid_idx = len(self.X_test) // 2
            instance = self.model.named_steps["imputer"].transform(self.X_test.iloc[mid_idx:mid_idx+1])[0]
            
            exp = explainer.explain_instance(
                instance, self.model.named_steps["model"].predict, 
                num_features=min(10, len(self.feature_cols))
            )
            fig = exp.as_pyplot_figure()
            fig.suptitle(f"LIME 解釋 (單一樣本) - {self.target}")
            self._save_fig(f"lime_instance_{self.target}.png")
        except Exception as e:
            logging.warning(f"LIME 分析失敗: {e}")

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
            plt.xlabel("效能下降量 (Performance Decrease)")
            self._save_fig(f"permutation_importance_{self.target}.png")
        except Exception as e:
            logging.warning(f"Permutation Importance 分析失敗: {e}")
            
    def generate_pdp(self):
        try:
            # 我們只為最重要的前 4 個特徵繪製 PDP
            importances = pd.Series(
                self.model.named_steps["model"].feature_importances_, 
                index=self.feature_cols
            ).sort_values(ascending=False)
            top_features = importances.head(4).index.tolist()

            PartialDependenceDisplay.from_estimator(
                self.model, self.X_imputed, top_features, 
                feature_names=self.feature_cols, n_cols=2
            )
            fig = plt.gcf()
            fig.set_size_inches(12, 8)
            fig.suptitle(f"Partial Dependence Plots - {self.target}", fontsize=16)
            plt.subplots_adjust(top=0.9)
            self._save_fig(f"pdp_top4_{self.target}.png")
        except Exception as e:
            logging.warning(f"PDP 分析失敗: {e}")

# --- 主執行流程 ---
def main():
    """主執行函數，為每個目標建立 Explainer 並執行分析。"""
    targets_str = os.environ.get("TARGETS", "ERS,rmssd_post").strip()
    targets = [t.strip() for t in targets_str.split(",") if t.strip()]
    
    for target in targets:
        try:
            explainer = ModelExplainer(target)
            explainer.run_all_explanations()
        except FileNotFoundError as e:
            logging.error(f"無法為目標 '{target}' 進行分析，缺少必要檔案: {e}")
        except Exception as e:
            logging.error(f"為目標 '{target}' 分析時發生未預期錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    main()