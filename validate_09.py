# validate_09.py

import logging
import json
from pathlib import Path

# --- Local Modules ---
# 直接從您現有的腳本中導入核心組件
from hrr_analysis.models_04 import ModelTrainer, RANDOM_SEED
from paths import DIR_FEATURES, DIR_REPORT
from utils.data_loader import load_dataframe

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def run_validation_experiment(data_path: Path, target: str) -> dict:
    """
    針對給定的數據源和目標，執行一次完整的模型訓練與評估。

    Args:
        data_path (Path): 輸入的特徵檔案路徑 (e.g., features_ml.parquet)。
        target (str): 要預測的目標變數 (e.g., "ERS")。

    Returns:
        dict: 包含最佳模型名稱和 R² 分數的結果字典。
    """
    logging.info(f"--- Running experiment for target '{target}' with data '{data_path.name}' ---")
    try:
        # 載入數據
        df = load_dataframe(data_path)
        
        # 準備一個簡化版的 ModelTrainer，只取其核心的 run 方法
        # 注意：為了重用邏輯，我們傳入一個假的 data_path 給 run 方法，因為我們已經載入了數據
        # 這樣做可以避免修改您原始的 ModelTrainer
        trainer = ModelTrainer(target_name=target, seed=RANDOM_SEED)
        
        # 執行核心的數據準備、切分、訓練與評估流程
        df_valid, feature_cols = trainer._prepare_data(df)
        X_train, X_test, y_train, y_test, _ = trainer._split_data(df_valid, feature_cols)
        _, best_model_info = trainer._train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # 只返回我們最關心的指標
        return {
            "model_name": best_model_info.get("name"),
            "r2_score": best_model_info.get("r2")
        }

    except Exception as e:
        logging.error(f"Experiment failed for {data_path.name} on target {target}: {e}", exc_info=True)
        return {"model_name": "error", "r2_score": -999}

def main():
    """主執行函式，協調對照實驗並產出最終報告。"""
    logging.info("===== Starting Augmentation Validation Experiment =====")
    
    # 定義我們的實驗組與對照組的數據源
    original_data_path = DIR_FEATURES / "features_ml.parquet"
    augmented_data_path = DIR_FEATURES / "features_ml_aug.parquet"
    target_to_validate = "ERS"  # 我們專注於驗證對 ERS 的影響

    # 檢查檔案是否存在
    if not original_data_path.exists() or not augmented_data_path.exists():
        logging.error("One or both required data files are missing. Please ensure both 'features_ml.parquet' and 'features_ml_aug.parquet' exist.")
        return

    # --- 執行實驗 ---
    # 對照組：僅使用原始數據
    results_original = run_validation_experiment(original_data_path, target_to_validate)
    
    # 實驗組：使用增強後的數據
    results_augmented = run_validation_experiment(augmented_data_path, target_to_validate)

    # --- 產出結論 ---
    logging.info("===== Augmentation Validation Results =====")
    
    report_lines = [
        "# Augmentation Effectiveness Validation Report\n",
        f"- **Target Variable**: `{target_to_validate}`",
        f"- **Validation Method**: Time-Series Split (70/30) on real data.\n",
        "## Comparison of Model Performance\n",
        "| Training Data         | Best Model      | Validation R² Score |",
        "|:----------------------|:----------------|:--------------------|",
        f"| Original Data Only    | `{results_original['model_name']}`      | **{results_original['r2_score']:.4f}** |",
        f"| With Augmented Data   | `{results_augmented['model_name']}`      | **{results_augmented['r2_score']:.4f}** |",
        "\n"
    ]

    r2_original = results_original['r2_score']
    r2_augmented = results_augmented['r2_score']
    
    if r2_augmented > r2_original:
        improvement = ((r2_augmented - r2_original) / abs(r2_original)) if r2_original != 0 else float('inf')
        conclusion = f"**Conclusion**: Data augmentation is **effective**. The model trained with augmented data showed a **{improvement:.2%} improvement** in R² score on the unseen real-data validation set."
        report_lines.append(conclusion)
        logging.info("✅ Augmentation is EFFECTIVE.")
    else:
        conclusion = "**Conclusion**: Data augmentation did **not** improve model performance on the validation set for this run. Further analysis on the augmentation strategy may be needed."
        report_lines.append(conclusion)
        logging.info("⚠️ Augmentation did NOT improve performance.")
        
    logging.info(f"Original R²: {r2_original:.4f} | Augmented R²: {r2_augmented:.4f}")

    # --- 儲存報告 ---
    report_path = DIR_REPORT / "augmentation_validation_report.md"
    report_content = "\n".join(report_lines)
    
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        logging.info(f"Validation report saved to: {report_path}")
    except Exception as e:
        logging.error(f"Failed to save validation report: {e}")

    # 同時將結果打印到終端機
    print("\n" + "="*50)
    print("      Augmentation Validation Summary")
    print("="*50)
    print(report_content)
    print("="*50 + "\n")


if __name__ == "__main__":
    main()