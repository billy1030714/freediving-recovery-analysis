"""
mean_rr_post Full Predictability Analysis
Analyze why mean_rr_post performs relatively well in Track B
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from utils.data_loader import find_data_file, load_dataframe
from scipy.stats import pearsonr

# Set chart style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def analyze_mean_rr_post():
    """Full predictability analysis of mean_rr_post"""
    
    print("=" * 60)
    print("mean_rr_post Full Predictability Analysis")
    print("=" * 60)
    
    # 1. Load model and data
    print("\n1. Loading model and data...")
    model = joblib.load('models/mean_rr_post/best_model.joblib')
    with open('models/mean_rr_post/feature_schema.json', 'r') as f:
        schema = json.load(f)
    features = schema['features']
    
    df = load_dataframe(find_data_file())
    df_valid = df[df['mean_rr_post'].notna()].copy()
    df_valid = df_valid[df_valid['mean_rr_post'] > 0]  # filter positive values
    
    print(f"Valid samples: {len(df_valid)}")
    print(f"Number of features: {len(features)}")
    
    # 2. Rebuild time series split
    print("\n2. Rebuilding validation set...")
    df_valid['end_apnea_time'] = pd.to_datetime(df_valid['end_apnea_time'])
    df_sorted = df_valid.sort_values('end_apnea_time')
    split_idx = int(len(df_sorted) * 0.7)
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['mean_rr_post']
    X_test = test_df[features]
    y_test = test_df['mean_rr_post']
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # 3. Model performance analysis
    print("\n3. Model performance analysis...")
    model_type = type(model.named_steps['model']).__name__
    print(f"Best model: {model_type}")
    
    y_pred_full = model.predict(X_test)
    r2_full = r2_score(y_test, y_pred_full)
    mae_full = mean_absolute_error(y_test, y_pred_full)
    
    print(f"Full model R²: {r2_full:.4f}")
    print(f"Full model MAE: {mae_full:.2f}")
    
    # 4. Feature importance analysis
    print("\n4. Feature importance analysis...")
    
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)
        print("Tree-based model feature importance (Top 10):")
        print(feature_imp.head(10))
        
    elif hasattr(model.named_steps['model'], 'coef_'):
        coefs = model.named_steps['model'].coef_
        feature_imp = pd.Series(np.abs(coefs), index=features).sort_values(ascending=False)
        print("Linear model coefficient importance (Top 10):")
        print(feature_imp.head(10))
    
    # 5. Permutation importance analysis
    print("\n5. Permutation importance analysis...")
    try:
        perm_importance = permutation_importance(
            model, X_test, y_test, 
            n_repeats=10, random_state=42, n_jobs=1
        )
        perm_imp = pd.Series(
            perm_importance.importances_mean, 
            index=features
        ).sort_values(ascending=False)
        
        print("Permutation importance (Top 10):")
        print(perm_imp.head(10))
        
        if 'HRbaseline' in perm_imp.index:
            print(f"\nHRbaseline permutation importance: {perm_imp['HRbaseline']:.6f}")
            
    except Exception as e:
        print(f"Permutation importance analysis failed: {e}")
    
    # 6. Predictive power of HRbaseline alone
    print("\n6. HRbaseline standalone prediction analysis...")
    
    if 'HRbaseline' in features:
        # Create a simple model using only HRbaseline
        simple_model = LinearRegression()
        
        # Handle missing values
        X_train_simple = train_df[['HRbaseline']].fillna(train_df['HRbaseline'].median())
        X_test_simple = test_df[['HRbaseline']].fillna(test_df['HRbaseline'].median())
        
        simple_model.fit(X_train_simple, y_train)
        y_pred_simple = simple_model.predict(X_test_simple)
        r2_simple = r2_score(y_test, y_pred_simple)
        mae_simple = mean_absolute_error(y_test, y_pred_simple)
        
        print(f"Using only HRbaseline R²: {r2_simple:.4f}")
        print(f"Using only HRbaseline MAE: {mae_simple:.2f}")
        print(f"HRbaseline contribution: {r2_simple/r2_full*100:.1f}%")
        
        # Correlation in test set
        baseline_values = test_df['HRbaseline'].fillna(test_df['HRbaseline'].median())
        correlation, p_value = pearsonr(baseline_values, y_test)
        print(f"Test set HRbaseline vs mean_rr_post correlation: {correlation:.4f} (p = {p_value:.6f})")
        
    else:
        print("HRbaseline not found in feature set")
    
    # 7. Physiological mechanism validation
    print("\n7. Physiological mechanism validation...")
    
    if 'HRbaseline' in df_valid.columns:
        # Calculate theoretical RR interval
        df_valid_clean = df_valid.dropna(subset=['HRbaseline', 'mean_rr_post'])
        df_valid_clean['theoretical_rr'] = 60000 / df_valid_clean['HRbaseline']
        
        # Correlation across full dataset
        total_correlation, total_p_value = pearsonr(df_valid_clean['theoretical_rr'], df_valid_clean['mean_rr_post'])
        print(f"Full dataset: Theoretical RR vs Actual RR correlation: {total_correlation:.4f} (p = {total_p_value:.6f})")
        
        # Statistical summary
        print("\nStatistical summary:")
        print("HRbaseline statistics:")
        print(df_valid_clean['HRbaseline'].describe())
        print("\nmean_rr_post statistics:")
        print(df_valid_clean['mean_rr_post'].describe())
        print("\nTheoretical RR statistics:")
        print(df_valid_clean['theoretical_rr'].describe())
        
        # 8. Visualization analysis
        print("\n8. Generating visualization charts...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('mean_rr_post Predictability Analysis', fontsize=16)
        
        # 8.1 Predicted vs Actual
        axes[0,0].scatter(y_test, y_pred_full, alpha=0.6, s=30)
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[0,0].set_xlabel('Actual mean_rr_post')
        axes[0,0].set_ylabel('Predicted mean_rr_post')
        axes[0,0].set_title(f'Full Model Prediction (R² = {r2_full:.3f})')
        axes[0,0].grid(True)
        
        # 8.2 HRbaseline vs mean_rr_post
        if 'HRbaseline' in features:
            test_baseline = test_df['HRbaseline'].fillna(test_df['HRbaseline'].median())
            axes[0,1].scatter(test_baseline, y_test, alpha=0.6, s=30)
            axes[0,1].set_xlabel('HRbaseline')
            axes[0,1].set_ylabel('mean_rr_post')
            axes[0,1].set_title(f'Baseline HR vs RR Interval (r = {correlation:.3f})')
            axes[0,1].grid(True)
        
        # 8.3 Theoretical vs Actual RR
        axes[1,0].scatter(df_valid_clean['theoretical_rr'], df_valid_clean['mean_rr_post'], alpha=0.6, s=30)
        axes[1,0].plot([df_valid_clean['theoretical_rr'].min(), df_valid_clean['theoretical_rr'].max()], 
                      [df_valid_clean['theoretical_rr'].min(), df_valid_clean['theoretical_rr'].max()], 'r--')
        axes[1,0].set_xlabel('Theoretical RR Interval (60000/HR)')
        axes[1,0].set_ylabel('Actual mean_rr_post')
        axes[1,0].set_title(f'Theoretical vs Actual (r = {total_correlation:.3f})')
        axes[1,0].grid(True)
        
        # 8.4 Feature importance
        if 'feature_imp' in locals():
            top_features = feature_imp.head(8)
            axes[1,1].barh(range(len(top_features)), top_features.values[::-1])
            axes[1,1].set_yticks(range(len(top_features)))
            axes[1,1].set_yticklabels(top_features.index[::-1])
            axes[1,1].set_xlabel('Importance')
            axes[1,1].set_title('Top 8 Feature Importance')
            axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('mean_rr_post_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Charts saved as 'mean_rr_post_analysis.png'")
    
    # 9. Conclusion summary
    print("\n" + "=" * 60)
    print("Conclusion of Analysis:")
    print("=" * 60)
    
    if 'HRbaseline' in features and r2_simple/r2_full > 0.7:
        print(f"✅ Predictability of mean_rr_post mainly comes from baseline HR ({r2_simple/r2_full*100:.1f}%)")
        print(f"✅ This reflects the mathematical relation RR = 60000/HR (r = {total_correlation:.3f})")
        print(f"✅ It belongs to baseline stability prediction rather than dynamic recovery prediction")
    else:
        print("❓ Source of predictability requires further analysis")
    
    print(f"\nFull model performance: R² = {r2_full:.4f}, MAE = {mae_full:.2f}")
    
    return {
        'r2_full': r2_full,
        'r2_simple': r2_simple if 'r2_simple' in locals() else None,
        'correlation': total_correlation if 'total_correlation' in locals() else None,
        'model_type': model_type
    }

if __name__ == "__main__":
    results = analyze_mean_rr_post()
    print("\nAnalysis complete!")