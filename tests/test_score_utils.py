"""
test_score_utils.py - score_utils 模組的單元測試
"""

import pytest
import numpy as np
from utils.score_utils import compute_ers 


@pytest.mark.parametrize("test_name, inputs, expected_output", [
    (
        "ideal_case", 
        {"r60": 0.8, "r90": 0.9, "norm_slope": -0.4}, 
        0.78
    ),
    (
        "missing_r60", 
        {"r60": None, "r90": 0.9, "norm_slope": -0.4}, 
        0.46
    ),
    (
        "missing_r90", 
        {"r60": 0.8, "r90": np.nan, "norm_slope": -0.4}, 
        0.74
    ),
    (
        "missing_slope", 
        {"r60": 0.8, "r90": 0.9, "norm_slope": None}, 
        0.68
    ),
    (
        "slope_contribution_capped", 
        {"r60": 0.5, "r90": 0.5, "norm_slope": -1.0}, 
        0.60
    ),
    (
        "all_inputs_zero", 
        {"r60": 0.0, "r90": 0.0, "norm_slope": 0.0}, 
        0.0
    ),
    (
        "all_inputs_missing", 
        {"r60": None, "r90": None, "norm_slope": None}, 
        0.0
    ),
    (
        "perfect_score",
        {"r60": 1.0, "r90": 1.0, "norm_slope": -0.8},
        1.0
    )
])
def test_compute_ers(test_name, inputs, expected_output):
    """
    Unified testing of various input combinations for the compute_ers function.
    """
    result = compute_ers(
        r60=inputs.get("r60"),
        r90=inputs.get("r90"),
        norm_slope=inputs.get("norm_slope")
    )
    assert result == pytest.approx(expected_output)