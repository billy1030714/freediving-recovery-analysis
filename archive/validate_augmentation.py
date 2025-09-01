# validate_augmentation.py
import logging
import json
from pathlib import Path

# --- Local Modules ---
# Import core training utilities
from hrr_analysis.models_04 import ModelTrainer, RANDOM_SEED
from paths import DIR_FEATURES, DIR_REPORT
from utils.data_loader import load_dataframe

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def run_validation_experiment(data_path: Path, target: str) -> dict:
    """
    Run model training and evaluation for a given dataset and target.
    Args:
        data_path (Path): Feature file path (e.g., features_ml.parquet).
        target (str): Target variable to predict (e.g., "ERS").
    Returns:
        dict: Best model name and R² score.
    """
    logging.info(f"--- Running experiment for target '{target}' with data '{data_path.name}' ---")
    try:
        # Load data
        df = load_dataframe(data_path)

        # Initialize trainer
        trainer = ModelTrainer(target_name=target, seed=RANDOM_SEED)

        # Prepare data
        df_valid, feature_cols = trainer._prepare_data(df)
        X_train, X_test, y_train, y_test = trainer.split_data(df_valid, feature_cols)

        # Train & evaluate
        best_model_info = trainer._train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Return key metrics
        return {
            "model_name": best_model_info.get("name"),
            "r2_score": best_model_info.get("r2")
        }
    except Exception as e:
        logging.error(f"Experiment failed for {data_path.name} on target {target}: {e}", exc_info=True)
        return {"model_name": "error", "r2_score": -999}


def main():
    """Run augmentation validation experiment and generate report."""
    logging.info("===== Starting Augmentation Validation Experiment =====")

    # Define data sources
    original_data_path = DIR_FEATURES / "features_ml.parquet"
    augmented_data_path = DIR_FEATURES / "features_ml_aug.parquet"
    target_to_validate = "ERS"

    # Check if files exist
    if not original_data_path.exists() or not augmented_data_path.exists():
        logging.error("Missing required data files: 'features_ml.parquet' and/or 'features_ml_aug.parquet'.")
        return

    # Run experiments
    results_original = run_validation_experiment(original_data_path, target_to_validate)
    results_augmented = run_validation_experiment(augmented_data_path, target_to_validate)

    # Generate conclusions
    logging.info("===== Augmentation Validation Results =====")

    report_lines = [
        "# Augmentation Effectiveness Validation Report\n",
        f"- Target Variable: {target_to_validate}",
        f"- Validation Method: Time-Series Split (70/30) on real data.\n",
        "## Comparison of Model Performance\n",
        "| Training Data | Best Model | Validation R² Score |",
        "|:----------------------|:----------------|:--------------------|",
        f"| Original Data Only | {results_original['model_name']} | {results_original['r2_score']:.4f} |",
        f"| With Augmented Data | {results_augmented['model_name']} | {results_augmented['r2_score']:.4f} |",
        "\n"
    ]

    r2_original = results_original['r2_score']
    r2_augmented = results_augmented['r2_score']

    if r2_augmented > r2_original:
        improvement = ((r2_augmented - r2_original) / abs(r2_original)) if r2_original != 0 else float('inf')
        conclusion = f"Conclusion: Data augmentation is effective. The model trained with augmented data showed a {improvement:.2%} improvement in R² score on the unseen real-data validation set."
        report_lines.append(conclusion)
        logging.info("✅ Augmentation is EFFECTIVE.")
    else:
        conclusion = "Conclusion: Data augmentation did not improve model performance on the validation set. Further analysis on the augmentation strategy may be needed."
        report_lines.append(conclusion)
        logging.info("⚠️ Augmentation did NOT improve performance.")

    logging.info(f"Original R²: {r2_original:.4f} | Augmented R²: {r2_augmented:.4f}")

    # Save report
    report_path = DIR_REPORT / "augmentation_validation_report.md"
    report_content = "\n".join(report_lines)

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        logging.info(f"Validation report saved to: {report_path}")
    except Exception as e:
        logging.error(f"Failed to save validation report: {e}")

    # Print summary
    print("\n" + "=" * 50)
    print(" Augmentation Validation Summary")
    print("=" * 50)
    print(report_content)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()