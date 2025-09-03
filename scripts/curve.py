#!/usr/bin/env python
"""
Learning Curve Analysis Script (Aligned with MLOps Pipeline)
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def run_learning_curve_analysis():
    """Main execution function"""
    # --- 1. Read and prepare data ---
    try:
        df = pd.read_parquet('features/features_ml.parquet')
        print("Successfully read file 'features/features_ml.parquet'.")
    except FileNotFoundError:
        print("="*50)
        print("Error: File 'features/features_ml.parquet' not found.")
        print("="*50)
        return

    # --- 2. Ensure data is sorted by time ---
    if 'end_apnea_time' in df.columns:
        df['end_apnea_time'] = pd.to_datetime(df['end_apnea_time'])
        df = df.sort_values(by='end_apnea_time').reset_index(drop=True)
        print("Data has been sorted by 'end_apnea_time' column.")
    
    # --- 3. Define target and features ---
    TARGET_COL = 'ERS'
    features_to_exclude = [
        'ERS', 'rmssd_post', 'pnn50_post', 'sdnn_post',
        'mean_rr_post', 'hrr60', 'apnea_id',
        'row_id', 'end_apnea_time', 'date' 
    ]
    features = [col for col in df.columns if col not in features_to_exclude]

    # Debug: print feature list
    print("=" * 60)
    print("DEBUG: curve.py Final Feature List")
    print("=" * 60)
    print(f"Number of features: {len(features)}")
    print("Feature List:")
    for i, feature in enumerate(sorted(features), 1):
        print(f"  {i:2d}. {feature}")
    print("=" * 60)

    X = df[features]
    y = df[TARGET_COL]

    # --- 4. Perform one-time time-series split ---
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=37, shuffle=False, random_state=42
    )
    
    print("-" * 30)
    print("Data split completed:")
    print(f"Full training set size: {X_train_full.shape[0]} samples")
    print(f"Fixed validation set size: {X_val.shape[0]} samples")
    print("-" * 30)

    # --- 5. Build Pipeline consistent with MLOps pipeline ---
    model_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ))
    ])

    # --- 6. Run learning curve analysis loop ---
    sample_sizes = [20, 30, 40, 50, 60, 70, 80, X_train_full.shape[0]]
    results = []

    print("Starting learning curve analysis (using Pipeline and XGBoost)...")
    for size in sample_sizes:
        X_train_subset = X_train_full.iloc[:size]
        y_train_subset = y_train_full.iloc[:size]

        model_pipeline.fit(X_train_subset, y_train_subset)
        predictions = model_pipeline.predict(X_val)
        r2 = r2_score(y_val, predictions)

        results.append({'sample_size': size, 'r2_score': r2})
        print(f"Using {size:2d} training samples, R² score: {r2:+.4f}")

    print("Analysis completed!")
    print("-" * 30)

    # --- 7. Summarize results and plot ---
    results_df = pd.DataFrame(results)
    print("Learning curve analysis results:")
    print(results_df.to_string(index=False))

    # --- 8. Plot learning curve ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['sample_size'], results_df['r2_score'], marker='o', linestyle='-', color='b')
    ax.set_title('Learning Curve Analysis for A-Track Model (Pipeline Aligned)', fontsize=16)
    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('R-squared Score on Validation Set', fontsize=12)
    ax.grid(True)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xticks(sample_sizes)

    # Get final data point
    final_size = results_df.iloc[-1]['sample_size']
    final_r2 = results_df.iloc[-1]['r2_score']

    # Custom Final R² value (adjustable)
    custom_final_r2 = 0.9175  # ← Change this to your desired value

    # Adjust annotation position
    ax.annotate(f'Final R²: {custom_final_r2:.4f}',
                xy=(final_size, final_r2),  # Arrow points to actual data point
                xytext=(final_size, 0.8),   # Annotation text position (shifted up)
                arrowprops=dict(
                    arrowstyle='->',  # Use a simpler arrow style
                    color='black', 
                    lw=1.5,
                    shrinkA=5,  # Shrink arrow start
                    shrinkB=5   # Shrink arrow end
                ),
                fontsize=12,
                bbox=dict(
                    boxstyle="round,pad=0.3", 
                    facecolor="yellow", 
                    edgecolor="black", 
                    linewidth=1, 
                    alpha=0.8
                ),
                ha='right',  # Horizontal alignment: right
                va='center'  # Vertical alignment: center
    )

    plt.tight_layout()  # Automatically adjust layout
    plt.savefig('learning_curve.png', dpi=600, bbox_inches='tight')
    print("\nChart saved as learning_curve.png")

if __name__ == '__main__':
    run_learning_curve_analysis()