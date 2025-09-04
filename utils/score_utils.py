"""
score_utils.py - Core Recovery Score Computation Utilities

This module provides functions for computing the project's core Efficiency of Recovery Score (ERS).
The logic is separated out to ensure that the exact same standard is applied across all stages of data
processing, in compliance with the DRY (Don't Repeat Yourself) principle, and to facilitate unit testing.
"""
import numpy as np
import pandas as pd
from typing import Optional

def compute_ers(
    r60: Optional[float], 
    r90: Optional[float], 
    norm_slope: Optional[float]
) -> float:
    """
    Compute the Efficiency of Recovery Score (ERS).
    Integrates 60s/90s recovery ratios and normalized slope into a weighted score.
    Returns a value between 0 and 1.
    """
    # Step 1: Robustly handle missing values
    # If r60 is missing, treat it as the worst case (recovery ratio = 0)
    val_r60 = 0.0 if pd.isna(r60) else float(r60)

    # If r90 is missing, conservatively assume it equals r60
    val_r90 = val_r60 if pd.isna(r90) else float(r90)

    # If slope is missing, treat it as no recovery trend (contribution = 0)
    val_ns = 0.0 if pd.isna(norm_slope) else float(norm_slope)

    # Step 2: Convert slope into a contribution within the 0–1 range
    # Uses -0.8 bpm/second as physiological benchmark for ideal recovery slope
    # Formula: min(1.0, |slope| / 0.8) ensures contribution ∈ [0,1]
    slope_contribution = min(1.0, abs(val_ns) / 0.8)

    # Step 3: Compute the ERS score using weighted averaging
    ers_score = (0.4 * val_r60) + (0.4 * val_r90) + (0.2 * slope_contribution)
    
    # Step 4: Return the final score, ensured to lie within [0.0, 1.0]
    return float(np.clip(ers_score, 0.0, 1.0))