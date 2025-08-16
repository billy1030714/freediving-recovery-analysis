"""
score_utils.py - 複合分數計算工具

本模組提供用於計算專案核心目標分數 (如 ECS) 的函式。
將此邏輯獨立出來，是為了確保在數據處理的各個階段 (特徵工程、數據增強) 
都能使用完全相同的標準來計算分數，遵守 DRY (Don't Repeat Yourself) 原則。
"""
import numpy as np
import pandas as pd
from typing import Optional

def compute_ecs(
    r60: Optional[float], 
    r90: Optional[float], 
    norm_slope: Optional[float]
) -> float:
    """
    計算 End-of-session Composite Score (ECS)。

    這是一個啟發式的加權平均，綜合了 60 秒恢復率、90 秒恢復率和標準化恢復斜率。
    - 處理缺失值：以合理的預設值或相關值進行填充。
    - 標準化：將各項指標轉換為 0-1 之間的數值。
    - 加權平均：給予各指標不同的權重，綜合為最終分數。

    Args:
        r60 (Optional[float]): 60 秒恢復率 (recovery_ratio_60s)。
        r90 (Optional[float]): 90 秒恢復率 (recovery_ratio_90s)。
        norm_slope (Optional[float]): 標準化恢復斜率 (normalized_slope)。

    Returns:
        float: 計算出的 ECS 分數，範圍在 0 到 1 之間。
    """
    # 處理缺失值
    val_r60 = 0.0 if pd.isna(r60) else float(r60)
    val_r90 = val_r60 if pd.isna(r90) else float(r90)  # 若 r90 為空，則假設其表現與 r60 相同
    val_ns = 0.0 if pd.isna(norm_slope) else float(norm_slope)

    # 將斜率轉換為 0-1 範圍的貢獻度 (假設 -0.8 的斜率表現為 100%)
    slope_contribution = min(1.0, abs(val_ns) / 0.8)

    # 加權計算 ECS
    ecs_score = 0.4 * val_r60 + 0.4 * val_r90 + 0.2 * slope_contribution
    
    return float(np.clip(ecs_score, 0.0, 1.0))