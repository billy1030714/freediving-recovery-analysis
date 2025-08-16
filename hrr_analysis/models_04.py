"""
models_04.py – 模型訓練、評估與版本化腳本 (新增儲存目標分佈)

【本次更新】:
- 在儲存產出物時，額外儲存一份 `target_distribution.json`。
- 此檔案包含了「真實訓練數據」的目標值 (y_real_train)，將作為下游 06_predict 腳本計算百分位排名的依據。
"""

# --- 導入函式庫 ---
import argparse, json, logging, os, warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple
import joblib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# --- 本地模組 ---
from paths import DIR_MODELS
from utils.data_loader import find_data_file, load_dataframe

# --- 全域設定與版本凍結 ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
RANDOM_SEED = 42
MODEL_CONFIG = {
    "xgb": {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "reg_alpha": 0.0, "random_state": RANDOM_SEED, "n_jobs": -1, "tree_method": "hist"},
    "rf": {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 2, "random_state": RANDOM_SEED, "n_jobs": -1},
    "ridge": {"alpha": 1.0, "random_state": RANDOM_SEED}
}

class ModelTrainer:
    def __init__(self, target_name: str, seed: int):
        self.target = target_name
        self.seed = seed
        self.output_dir = DIR_MODELS / self.target
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_df = pd.DataFrame()
        self.y_real_train = pd.Series() # <-- 初始化
        logging.info(f"===== [初始化] ModelTrainer for target: {self.target} =====")

    def run(self, df: pd.DataFrame, data_path: Path):
        try:
            df_valid, feature_cols = self._prepare_data(df)
            X_train, X_test, y_train, y_test, meta = self._split_data(df_valid, feature_cols)
            board, best_model_info = self._train_and_evaluate(X_train, X_test, y_train, y_test)
            self._evaluate_on_dummies(best_model_info["pipeline"], feature_cols)
            self._save_artifacts(best_model_info, board, feature_cols, meta, data_path)
            logging.info(f"===== [成功] Target '{self.target}' 訓練完成。 =====")
        except Exception as e:
            logging.error(f"為目標 '{self.target}' 訓練時發生未預期錯誤: {e}", exc_info=True)

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        if self.target not in df.columns: raise KeyError(f"目標欄位 '{self.target}' 不存在。")
        df_valid = df[df[self.target].notna()].copy()
        if self.target == "rmssd_post": df_valid = df_valid[df_valid[self.target] > 0]
        if df_valid.empty: raise ValueError("過濾後無可用數據。")
        if 'end_apnea_time' in df_valid.columns: df_valid['end_apnea_time'] = pd.to_datetime(df_valid['end_apnea_time'])
        drop_cols = {"ERS", "rmssd_post", "row_id", "date", "end_apnea_time", "dummy_flag", self.target}
        feature_cols = [c for c in df_valid.select_dtypes(include=np.number).columns if c not in drop_cols]
        logging.info(f"數據準備完成: {len(df_valid)} 筆有效樣本, {len(feature_cols)} 個特徵。")
        return df_valid, feature_cols

    def _split_data(self, df_valid: pd.DataFrame, feature_cols: List[str]) -> tuple:
        logging.info("執行時間序列切分...")
        meta = {"split_strategy": "time_series_70_30"}
        real_df = df_valid[df_valid.get("dummy_flag", 0) == 0]
        self.dummy_df = df_valid[df_valid.get("dummy_flag", 0) == 1]
        if 'end_apnea_time' not in real_df.columns: raise ValueError("缺少 'end_apnea_time' 欄位。")
        
        real_df_sorted = real_df.sort_values("end_apnea_time")
        split_idx = int(len(real_df_sorted) * 0.7)
        if split_idx < 1 or split_idx >= len(real_df_sorted): raise ValueError(f"真實數據量 ({len(real_df_sorted)}) 過少。")
        
        real_train_df = real_df_sorted.iloc[:split_idx]
        real_test_df = real_df_sorted.iloc[split_idx:]
        
        X_real_train, y_real_train = real_train_df[feature_cols], real_train_df[self.target]
        self.y_real_train = y_real_train # <-- 儲存 y_real_train 以供後續儲存
        X_test, y_test = real_test_df[feature_cols], real_test_df[self.target]
        
        X_train = pd.concat([X_real_train, self.dummy_df[feature_cols]], axis=0)
        y_train = pd.concat([y_real_train, self.dummy_df[self.target]], axis=0)
        
        meta.update({"n_train_real": len(X_real_train), "n_train_dummy": len(self.dummy_df), "n_test_real": len(X_test), "dummies_in_training": not self.dummy_df.empty})
        logging.info(f"數據分割完成: 訓練集 {len(X_train)} 筆, 驗證集 {len(X_test)} 筆。")
        return X_train, X_test, y_train, y_test, meta

    def _train_and_evaluate(self, X_train, X_test, y_train, y_test) -> tuple:
        # --- DEBUG PROBE (ARCHIVED FOR FUTURE USE) ---
        # if self.target == 'ERS':
        #     logging.info("===== 偵錯探針：檢查送入 ERS 模型的訓練數據 =====")
        #     logging.info(f"y_train (ERS) 的統計描述:\n{y_train.describe().to_string()}")
        #     logging.info(f"X_train 的欄位:\n{X_train.columns.tolist()}")
        #     logging.info("==================== 偵錯結束 ====================")
        
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
            logging.info(f"[評估] {name}: R²={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}")
            if metrics["r2"] > best_model_info["r2"]:
                best_model_info = metrics.copy()
                best_model_info["pipeline"] = pipe
                
        logging.info(f"[最佳模型] 本次競賽冠軍為: {best_model_info['name']} (R² = {best_model_info['r2']:.4f})")
        return leaderboard, best_model_info

    def _evaluate_on_dummies(self, model_pipeline: Pipeline, feature_cols: List[str]):
        # (此函數保持不變)
        if self.dummy_df.empty: logging.info("數據集中無合成樣本，跳過 Dummy-only 評估。"); return
        logging.info("執行 Dummy-only 評估...")
        X_dummy, y_dummy_true = self.dummy_df[feature_cols], self.dummy_df[self.target]
        y_dummy_pred = model_pipeline.predict(X_dummy)
        metrics = {"r2": r2_score(y_dummy_true, y_dummy_pred), "mae": mean_absolute_error(y_dummy_true, y_dummy_pred), "rmse": np.sqrt(mean_squared_error(y_dummy_true, y_dummy_pred)), "n_dummy_samples": len(self.dummy_df)}
        logging.info(f"[評估@Dummies] R²={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}")
        with open(self.output_dir / "dummy_evaluation.json", "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2, ensure_ascii=False)
        logging.info(f"Dummy-only 評估結果已儲存至: {self.output_dir / 'dummy_evaluation.json'}")

    def _save_artifacts(self, best_model_info, leaderboard, feature_columns, dataset_metadata, data_path):
        # (既有儲存邏輯保持不變)
        best_model_path = self.output_dir / "best_model.joblib"; joblib.dump(best_model_info["pipeline"], best_model_path)
        if self.target == "ERS": joblib.dump(best_model_info["pipeline"], self.output_dir / "model.joblib")
        with open(self.output_dir / "leaderboard.json", "w", encoding="utf-8") as f: json.dump(leaderboard, f, indent=2, ensure_ascii=False)
        feature_schema = {"schema_version": "1.0", "feature_count": len(feature_columns), "features": feature_columns}
        with open(self.output_dir / "feature_schema.json", "w", encoding="utf-8") as f: json.dump(feature_schema, f, indent=2, ensure_ascii=False)
        dataset_card = {"card_version": "1.0", "source_file": str(data_path), "target_variable": self.target, "best_model_name": best_model_info["name"], "evaluation_metrics": {k: v for k, v in best_model_info.items() if k != 'pipeline'}, "dataset_split": dataset_metadata, "random_seed": self.seed}
        with open(self.output_dir / "dataset_card.json", "w", encoding="utf-8") as f: json.dump(dataset_card, f, indent=2, ensure_ascii=False)
        
        # --- 【新增】儲存真實訓練數據的目標分佈 ---
        dist_path = self.output_dir / "target_distribution.json"
        dist_data = {
            "target": self.target,
            "training_data_distribution": self.y_real_train.tolist()
        }
        with open(dist_path, "w", encoding="utf-8") as f:
            json.dump(dist_data, f)
        logging.info(f"真實訓練數據的目標分佈已儲存至: {dist_path}")
        logging.info(f"所有產出物已成功儲存至: {self.output_dir}")

def main():
    # (此函數保持不變)
    targets_str = os.environ.get("TARGETS", "ERS,rmssd_post").strip()
    targets = [t.strip() for t in targets_str.split(",") if t.strip()]
    try:
        data_path = find_data_file()
        df = load_dataframe(data_path)
        for target in targets:
            trainer = ModelTrainer(target_name=target, seed=RANDOM_SEED)
            trainer.run(df, data_path)
    except Exception as e:
        logging.error(f"主流程發生未預期錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    main()