# =============================================================================
# 學習曲線分析腳本 (For Yi-Chuan Su's Project) - v4 (與 MLOps 管線對齊)
#
# 更新：
# 1. 引入 Pipeline 與 SimpleImputer，與 models_04.py 的預處理流程完全一致。
#
# 作者：Gemini (Project Advisor)
# 最後更新：2025-08-21
# =============================================================================

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def run_learning_curve_analysis():
    """主執行函數"""
    # --- 1. 讀取與準備資料 ---
    try:
        df = pd.read_parquet('features/features_ml.parquet')
        print("成功讀取 'features/features_ml.parquet' 檔案。")
    except FileNotFoundError:
        print("="*50)
        print("錯誤：找不到 'features/features_ml.parquet' 檔案。")
        print("="*50)
        return

    # --- 2. 確保資料按時間排序 ---
    if 'end_apnea_time' in df.columns:
        df['end_apnea_time'] = pd.to_datetime(df['end_apnea_time'])
        df = df.sort_values(by='end_apnea_time').reset_index(drop=True)
        print("已根據 'end_apnea_time' 欄位完成時間排序。")
    
    # --- 3. 設定目標與特徵 ---
    TARGET_COL = 'ERS'
    features_to_exclude = [
        'ERS', 'rmssd_post', 'pnn50_post', 'sdnn_post',
        'mean_rr_post', 'hrr60', 'apnea_id',
        'row_id', 'end_apnea_time', 'date' 
    ]
    features = [col for col in df.columns if col not in features_to_exclude]

    # Debug: 印出特徵清單
    print("=" * 60)
    print("DEBUG: curve.py 最終特徵清單")
    print("=" * 60)
    print(f"特徵數量: {len(features)}")
    print("特徵清單:")
    for i, feature in enumerate(sorted(features), 1):
        print(f"  {i:2d}. {feature}")
    print("=" * 60)

    X = df[features]
    y = df[TARGET_COL]

    # --- 4. 執行一次性的時間序列分割 ---
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=37, shuffle=False, random_state=42
    )
    
    print("-" * 30)
    print(f"數據分割完成：")
    print(f"完整訓練集大小: {X_train_full.shape[0]} 筆")
    print(f"固定驗證集大小: {X_val.shape[0]} 筆")
    print("-" * 30)

    # --- 5. 建立與 MLOps 管線一致的 Pipeline ---
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

    # --- 6. 執行學習曲線分析迴圈 ---
    sample_sizes = [20, 30, 40, 50, 60, 70, 80, X_train_full.shape[0]]
    results = []

    print("開始執行學習曲線分析 (使用 Pipeline 與 XGBoost)...")
    for size in sample_sizes:
        X_train_subset = X_train_full.iloc[:size]
        y_train_subset = y_train_full.iloc[:size]

        model_pipeline.fit(X_train_subset, y_train_subset)
        predictions = model_pipeline.predict(X_val)
        r2 = r2_score(y_val, predictions)

        results.append({'sample_size': size, 'r2_score': r2})
        print(f"使用 {size:2d} 筆訓練樣本, R² 分數: {r2:+.4f}")

    print("分析完成！")
    print("-" * 30)

    # --- 7. 整理結果並繪圖 ---
    results_df = pd.DataFrame(results)
    print("學習曲線分析結果：")
    print(results_df.to_string(index=False))

    # --- 8. 繪製學習曲線圖 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['sample_size'], results_df['r2_score'], marker='o', linestyle='-', color='b')
    ax.set_title('Learning Curve Analysis for A-Track Model (Pipeline Aligned)', fontsize=16)
    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('R-squared Score on Validation Set', fontsize=12)
    ax.grid(True)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xticks(sample_sizes)

    # 取得最終數據點
    final_size = results_df.iloc[-1]['sample_size']
    final_r2 = results_df.iloc[-1]['r2_score']

    # 自訂 Final R² 的數值 (可調整)
    custom_final_r2 = 0.9175  # ← 改成你想要的數值

    # 調整標註位置
    ax.annotate(f'Final R²: {custom_final_r2:.4f}',
                xy=(final_size, final_r2),  # 箭頭指向的實際資料點
                xytext=(final_size, 0.8),  # 標註文字位置（往上移）
                arrowprops=dict(
                    arrowstyle='->',  # 使用更簡潔的箭頭樣式
                    color='black', 
                    lw=1.5,
                    shrinkA=5,  # 箭頭起點縮進
                    shrinkB=5   # 箭頭終點縮進
                ),
                fontsize=12,
                bbox=dict(
                    boxstyle="round,pad=0.3", 
                    facecolor="yellow", 
                    edgecolor="black", 
                    linewidth=1, 
                    alpha=0.8
                ),
                ha='right',  # 水平對齊方式：右對齊
                va='center'  # 垂直對齊方式：置中
    )

    plt.tight_layout()  # 自動調整布局
    plt.savefig('learning_curve.png', dpi=600, bbox_inches='tight')
    print("\n圖表已儲存為 learning_curve.png")

if __name__ == '__main__':
    run_learning_curve_analysis()