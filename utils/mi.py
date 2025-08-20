# utils/mi.py
# -*- coding: utf-8 -*-
"""
Mutual Information (MI) Utility: Ready for use by any module from 01-08.

- Target column defaults to 'ERS' (configurable via `target_col`).
- Leakage Prevention: Automatically excludes the target column, composite/mi score columns,
  and common timestamp/ID columns.
- I/O Support: Accepts a DataFrame or a file path (csv/parquet) as input.
  Outputs a reduced `features_ml` file, an MI scores CSV, and a horizontal bar plot.
- Returns: A tuple of (top_features, ml_df, mi_df, weights).
- Usage: Can be imported as a module or run as a CLI script:
  `python -m utils.mi --input ...`
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Tuple
from sklearn.feature_selection import mutual_info_regression


# --------- Internal Utilities ---------
def _load_table(path: str) -> pd.DataFrame:
    """Loads a DataFrame from a CSV or Parquet file path."""
    ext = os.path.splitext(path.lower())[1]
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _save_table(df: pd.DataFrame, path: str) -> None:
    """Saves a DataFrame to a CSV or Parquet file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path.lower())[1]
    if ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Selects numeric columns and handles constant columns safely."""
    num = df.select_dtypes(include=[np.number]).copy()
    # Drop constant columns (meaningless for MI) but keep track of them.
    nunique = num.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    dropped = [c for c in num.columns if c not in keep]
    if dropped:
        num = num[keep]
        # Store them to add back with a score of 0 later
        num.attrs["_dropped_constant"] = dropped
    else:
        num.attrs["_dropped_constant"] = []
    return num

def _mi_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Calculates mutual information scores for features in X against target y."""
    # Fill NaNs in X with 0; fill NaNs in y with its median for robustness.
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

    # Add back the dropped constant columns with a score of 0.
    dropped = X.attrs.get("_dropped_constant", [])
    for c in dropped:
        mi.loc[c] = 0.0
    # Reindex to match the original order of numeric columns, then sort again.
    num_cols = X.columns.tolist() + dropped
    return mi.reindex(num_cols, fill_value=0.0).sort_values(ascending=False)

def _normalize_0_1(s: pd.Series) -> pd.Series:
    """Normalizes a pandas Series to the [0, 1] range."""
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)

def _default_excludes(target_col: str) -> set[str]:
    """Returns a set of default columns to exclude from feature selection."""
    return {
        target_col,
        "composite_score",
        "mi_composite_score",
        "mi_composite_score_pct",
        "mi_score", "score",  # Possible legacy columns
        "end_apnea_time", "row_id",
        "start_time", "end_time", "timestamp", "time", "id", "subject_id", "uuid",
    }

def _pick_top_features(mi_series: pd.Series, top_n: int, excludes: set[str]) -> list[str]:
    """Selects top N features based on MI scores, enforcing inclusion of essential features."""
    top = [c for c in mi_series.index if c not in excludes][:top_n]
    # Force inclusion of key features if they exist
    for must in ("normalized_slope",):
        if (must in mi_series.index) and (must not in top):
            top.append(must)
    # Explicitly exclude legacy names
    top = [c for c in top if c not in ("hrr_slope", "composite_score")]
    return top

# --------- Core API ---------
def feature_selection_mi(
    features: pd.DataFrame | str,
    *,
    top_n: int = 10,
    target_col: str = "ERS",
    exclude_cols: Iterable[str] | None = None,
    # Optional outputs
    feature_output_path: str | None = None,   # Path for the reduced features_ml.* file
    mi_scores_csv_path: str | None = None,    # Path for the MI scores CSV
    mi_plot_path: str | None = None,          # Path for the MI scores plot
    verbose: bool = True,
) -> Tuple[list[str], pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Computes MI and returns a tuple of results.

    Returns:
      - top_features: List of top N feature names (including mandatory ones).
      - ml_df: DataFrame containing only essential columns, top features, and the new MI composite score.
      - mi_df: DataFrame of MI scores and feature names.
      - weights: A pandas Series of normalized weights for the top features.
    """
    if isinstance(features, str):
        df = _load_table(features)
    else:
        df = features.copy()

    if target_col not in df.columns:
        raise ValueError(f"[MI] Target column '{target_col}' not found in the DataFrame.")

    excludes = _default_excludes(target_col)
    if exclude_cols:
        excludes.update(exclude_cols)

    # Prepare X and y
    y = df[target_col]
    X_all = df.drop(columns=[c for c in excludes if c in df.columns], errors="ignore")
    X_num = _safe_numeric(X_all)
    if X_num.shape[1] == 0:
        raise ValueError("[MI] No numeric features available for MI calculation.")

    mi_series = _mi_scores(X_num, y)

    # Export MI scores CSV if path is provided
    if mi_scores_csv_path:
        os.makedirs(os.path.dirname(mi_scores_csv_path), exist_ok=True)
        mi_series.rename("mi").to_csv(mi_scores_csv_path, header=True)
        if verbose:
            print(f"[MI] MI scores CSV exported to: {mi_scores_csv_path}")

    # Export MI plot if path is provided
    if mi_plot_path:
        os.makedirs(os.path.dirname(mi_plot_path), exist_ok=True)
        plt.figure(figsize=(8, max(4, 0.35 * len(mi_series))))
        plt.barh(range(len(mi_series)), mi_series.values)
        plt.yticks(range(len(mi_series)), mi_series.index)
        plt.gca().invert_yaxis()
        plt.title("Mutual Information Scores")
        plt.xlabel("MI Score")
        plt.tight_layout()
        plt.savefig(mi_plot_path, dpi=300)
        plt.close()
        if verbose:
            print(f"[MI] MI scores plot exported to: {mi_plot_path}")

    # Get top features
    top_features = _pick_top_features(mi_series, top_n, excludes)

    # Create the reduced DataFrame for ML
    cols_for_ml = []
    if "end_apnea_time" in df.columns:
        cols_for_ml.append("end_apnea_time")
    cols_for_ml.append(target_col)
    cols_for_ml += [c for c in top_features if c in df.columns]
    ml_df = df[cols_for_ml].copy()

    # Calculate weights and composite score
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
            print("[MI] ⚠️ MI scores sum to 0. Composite score set to 0. (Possible reason: low data volume or constant features).")

    # Export the reduced ML DataFrame if path is provided
    if feature_output_path:
        _save_table(ml_df.round(6), feature_output_path)
        if verbose:
            print(f"[MI] ✅ Reduced ML features file exported to: {feature_output_path}")

    mi_df = mi_series.rename("mi").reset_index().rename(columns={"index": "feature"})
    return top_features, ml_df, mi_df, weights


# --------- Convenience Wrapper ---------
def run_mi_on_file(
    input_path: str,
    *,
    output_dir: str | None = None,
    top_n: int = 10,
    target_col: str = "ERS",
    exclude_cols: Iterable[str] | None = None,
    save_ml_name: str = "features_ml.csv",
    save_rank_name: str = "mi_scores.csv",
    save_plot_name: str = "mi_scores.png",
    verbose: bool = True,
):
    """
    Convenience function to run the full MI pipeline on a file and save all artifacts.

    Reads a file, computes MI, and outputs three files to the specified directory:
      - features_ml.csv (reduced feature set)
      - mi_scores.csv (MI rankings)
      - mi_scores.png (horizontal bar plot)
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


# --------- CLI Entry Point: `python -m utils.mi` ---------
def _build_argparser() -> argparse.ArgumentParser:
    """Builds the argument parser for the CLI."""
    p = argparse.ArgumentParser(description="HRR_project MI Utility (supports csv/parquet)")
    p.add_argument("--input", required=True, help="Input features file (CSV/Parquet)")
    p.add_argument("--outdir", default=None, help="Output directory (defaults to the same as input)")
    p.add_argument("--top_n", type=int, default=10, help="Number of top features to select based on MI")
    p.add_argument("--target", default="ERS", help="Target column for MI calculation (default: ERS)")
    p.add_argument("--exclude", nargs="*", default=[], help="Additional columns to exclude (space-separated)")
    p.add_argument("--ml_name", default="features_ml.csv", help="Filename for the reduced ML features output")
    p.add_argument("--rank_name", default="mi_scores.csv", help="Filename for the MI scores CSV output")
    p.add_argument("--plot_name", default="mi_scores.png", help="Filename for the MI scores plot output")
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    return p

def _cli():
    """The main function for the command-line interface."""
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
        print("\n[MI] Done. Summary:")
        print(f" - Top features: {', '.join(res['top_features'])}")
        print(f" - Weights: {res['weights'].round(4).to_dict()}")
        print(f" - ML file: {res['ml_path']}")
        print(f" - Rank csv: {res['rank_path']}")
        print(f" - Plot png: {res['plot_path']}")

if __name__ == "__main__":
    _cli()