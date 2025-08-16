# utils/mi.py
# -*- coding: utf-8 -*-
"""
Mutual Information (MI) 工具：01–07 任一模組可直接使用
- 目標欄位預設為 ECS（可改 target_col）
- 避免洩漏：自動排除目標欄、composite/mi 分數欄、常見 timestamp/ID 等
- 支援：DataFrame 或檔案路徑（csv/parquet）作為輸入；輸出縮減版 features_ml、MI 排名 CSV、水平長條圖
- 回傳：top_features, ml_df, mi_df, weights（權重 Series）
- 可當模組 import，也可 CLI 執行：python -m utils.mi --input ...
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Tuple
from sklearn.feature_selection import mutual_info_regression


# --------- 內部工具 ---------
def _load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path.lower())[1]
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path.lower())[1]
    if ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    # 去除全常數欄（MI 無意義），但會在最後補 0 分方便檢視
    nunique = num.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    dropped = [c for c in num.columns if c not in keep]
    if dropped:
        num = num[keep]
        num.attrs["_dropped_constant"] = dropped  # 記錄下來待會補 0
    else:
        num.attrs["_dropped_constant"] = []
    return num

def _mi_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    # 缺值補 0；y 若有缺失用中位數
    Xn = X.fillna(0.0)
    yv = pd.to_numeric(y, errors="coerce")
    if not np.isfinite(yv).any():
        yv = pd.Series(np.zeros(len(y)), index=y.index, dtype=float)
    else:
        yv = yv.fillna(yv.median())

    try:
        scores = mutual_info_regression(Xn.values, yv.values)
    except Exception:
        scores = np.zeros(Xn.shape[1], dtype=float)

    mi = pd.Series(scores, index=Xn.columns, dtype=float).sort_values(ascending=False)

    # 將剛剛丟掉的常數欄補回 0 分
    dropped = X.attrs.get("_dropped_constant", [])
    for c in dropped:
        mi.loc[c] = 0.0
    # 以原始欄位順序（數值欄）排列
    num_cols = X.columns.tolist() + dropped
    return mi.reindex(num_cols, fill_value=0.0).sort_values(ascending=False)

def _normalize_0_1(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)

def _default_excludes(target_col: str) -> set[str]:
    return {
        target_col,
        "composite_score",
        "mi_composite_score",
        "mi_composite_score_pct",
        "mi_score", "score",  # 可能的舊欄
        "end_apnea_time",
        "start_time", "end_time", "timestamp", "time", "id", "subject_id", "uuid",
    }

def _pick_top_features(mi_series: pd.Series, top_n: int, excludes: set[str]) -> list[str]:
    top = [c for c in mi_series.index if c not in excludes][:top_n]
    # 關鍵欄位強制納入（若存在）
    for must in ("normalized_slope",):
        if (must in mi_series.index) and (must not in top):
            top.append(must)
    # 明確排除舊名
    top = [c for c in top if c not in ("hrr_slope", "composite_score")]
    return top

# --------- 核心 API ---------
def feature_selection_mi(
    features: pd.DataFrame | str,
    *,
    top_n: int = 10,
    target_col: str = "ECS",
    exclude_cols: Iterable[str] | None = None,
    # 可選輸出
    feature_output_path: str | None = None,   # 縮減版 features_ml.*（沿用舊名）
    mi_scores_csv_path: str | None = None,    # MI 排名 CSV
    mi_plot_path: str | None = None,          # MI 水平長條圖
    verbose: bool = True,
) -> Tuple[list[str], pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    計算 MI 並回傳：
      - top_features       : 依 MI 排名擇前 top_n（含必要強制欄位）
      - ml_df              : 只含 [end_apnea_time(若有), target_col, top_features, mi_composite_score(_pct)]
      - mi_df              : MI 排名表（columns=['feature','mi']）
      - weights            : top_features 的正規化權重 Series（和為 1）
    可輸出：縮減版 features_ml（CSV/Parquet）、MI 排名 CSV、與 PNG 圖
    """
    if isinstance(features, str):
        df = _load_table(features)
    else:
        df = features.copy()

    if target_col not in df.columns:
        raise ValueError(f"[MI] 找不到目標欄位 '{target_col}'。")

    excludes = _default_excludes(target_col)
    if exclude_cols:
        excludes.update(exclude_cols)

    # X / y
    y = df[target_col]
    X_all = df.drop(columns=[c for c in excludes if c in df.columns], errors="ignore")
    X_num = _safe_numeric(X_all)
    if X_num.shape[1] == 0:
        raise ValueError("[MI] 沒有可用的數值特徵。")

    mi_series = _mi_scores(X_num, y)

    # 輸出 MI 排名 CSV
    if mi_scores_csv_path:
        os.makedirs(os.path.dirname(mi_scores_csv_path), exist_ok=True)
        mi_series.rename("mi").to_csv(mi_scores_csv_path, header=True)
        if verbose:
            print(f"[MI] 已輸出 MI 排名 CSV：{mi_scores_csv_path}")

    # 輸出水平長條圖
    if mi_plot_path:
        os.makedirs(os.path.dirname(mi_plot_path), exist_ok=True)
        plt.figure(figsize=(8, max(4, 0.35 * len(mi_series))))
        plt.barh(range(len(mi_series)), mi_series.values)
        plt.yticks(range(len(mi_series)), mi_series.index)
        plt.gca().invert_yaxis()
        plt.title("Mutual Information Scores")
        plt.xlabel("MI")
        plt.tight_layout()
        plt.savefig(mi_plot_path, dpi=300)
        plt.close()
        if verbose:
            print(f"[MI] 已輸出 MI 圖：{mi_plot_path}")

    # 取前 top_n + 強制欄位
    top_features = _pick_top_features(mi_series, top_n, excludes)

    # 建立縮減版 ml_df
    cols_for_ml = []
    if "end_apnea_time" in df.columns:
        cols_for_ml.append("end_apnea_time")
    cols_for_ml.append(target_col)
    cols_for_ml += [c for c in top_features if c in df.columns]
    ml_df = df[cols_for_ml].copy()

    # 權重（僅針對 top_features）
    weights = mi_series[top_features].astype(float)
    if weights.sum() > 0:
        weights = weights / weights.sum()
        used = [c for c in top_features if c in ml_df.columns]
        ml_df["mi_composite_score"] = (ml_df[used] * weights[used]).sum(axis=1)
        ml_df["mi_composite_score_pct"] = _normalize_0_1(ml_df["mi_composite_score"]) * 100.0
    else:
        weights[:] = 0.0
        ml_df["mi_composite_score"] = 0.0
        ml_df["mi_composite_score_pct"] = 0.0
        if verbose:
            print("[MI] ⚠️ MI 分數為 0，已以 0 代替（可能資料量太少或全常數）。")

    # 若指定縮減版輸出
    if feature_output_path:
        _save_table(ml_df.round(6), feature_output_path)
        if verbose:
            print(f"[MI] ✅ 縮減版 features_ml 已輸出：{feature_output_path}")

    mi_df = mi_series.rename("mi").reset_index().rename(columns={"index": "feature"})
    return top_features, ml_df, mi_df, weights


# --------- 便利函式（讓 01–07 都可一行呼叫）---------
def run_mi_on_file(
    input_path: str,
    *,
    output_dir: str | None = None,
    top_n: int = 10,
    target_col: str = "ECS",
    exclude_cols: Iterable[str] | None = None,
    save_ml_name: str = "features_ml.csv",
    save_rank_name: str = "mi_scores.csv",
    save_plot_name: str = "mi_scores.png",
    verbose: bool = True,
):
    """
    讀檔→計算→一口氣輸出三個檔案到 output_dir（預設同資料夾）：
      - features_ml.csv（縮減版）
      - mi_scores.csv（MI 排名）
      - mi_scores.png（水平長條圖）
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    os.makedirs(output_dir, exist_ok=True)

    feat_out = os.path.join(output_dir, save_ml_name)
    rank_out = os.path.join(output_dir, save_rank_name)
    plot_out = os.path.join(output_dir, save_plot_name)

    top_features, ml_df, mi_df, weights = feature_selection_mi(
        features=input_path,
        top_n=top_n,
        target_col=target_col,
        exclude_cols=exclude_cols,
        feature_output_path=feat_out,
        mi_scores_csv_path=rank_out,
        mi_plot_path=plot_out,
        verbose=verbose,
    )
    return {
        "top_features": top_features,
        "weights": weights,
        "ml_path": feat_out,
        "rank_path": rank_out,
        "plot_path": plot_out,
        "ml_df": ml_df,
        "mi_df": mi_df,
    }


# --------- CLI 入口：python -m utils.mi ---------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HRR_project MI 工具（支援 csv/parquet）")
    p.add_argument("--input", required=True, help="輸入 features 檔（CSV/Parquet）")
    p.add_argument("--outdir", default=None, help="輸出資料夾（預設同輸入檔）")
    p.add_argument("--top_n", type=int, default=10, help="取前 N 個 MI 特徵")
    p.add_argument("--target", default="ECS", help="目標欄位（預設 ECS）")
    p.add_argument("--exclude", nargs="*", default=[], help="額外排除欄位（空白分隔）")
    p.add_argument("--ml_name", default="features_ml.csv", help="縮減版輸出檔名")
    p.add_argument("--rank_name", default="mi_scores.csv", help="MI 排名 CSV 檔名")
    p.add_argument("--plot_name", default="mi_scores.png", help="MI 圖檔名")
    p.add_argument("--quiet", action="store_true", help="安靜模式")
    return p

def _cli():
    ap = _build_argparser()
    args = ap.parse_args()
    res = run_mi_on_file(
        input_path=args.input,
        output_dir=args.outdir,
        top_n=args.top_n,
        target_col=args.target,
        exclude_cols=args.exclude,
        save_ml_name=args.ml_name,
        save_rank_name=args.rank_name,
        save_plot_name=args.plot_name,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print("[MI] 完成。重點：")
        print(" - Top features:", ", ".join(res["top_features"]))
        print(" - Weights:", res["weights"].round(4).to_dict())
        print(" - ML file:", res["ml_path"])
        print(" - Rank csv:", res["rank_path"])
        print(" - Plot png:", res["plot_path"])

if __name__ == "__main__":
    _cli()