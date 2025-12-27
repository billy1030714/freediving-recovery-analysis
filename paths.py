# HRR_project/paths.py

from __future__ import annotations

from pathlib import Path
from datetime import date
from typing import Final
import os


def find_project_root(
    anchor: str = ".project_root",
    start_path: Path | None = None
) -> Path:
    """
    Locate the project root directory by searching for an anchor file.

    Parameters
    ----------
    anchor : str
        Name of the anchor file that defines the project root.
        This file must exist directly under the root directory.
    start_path : Path | None
        Directory to start searching from.
        If None, the directory of this file is used.

    Returns
    -------
    Path
        Absolute path to the project root directory.

    Raises
    ------
    FileNotFoundError
        If the anchor file cannot be found in any parent directory.
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    for parent in (start_path, *start_path.parents):
        if (parent / anchor).is_file():
            return parent

    raise FileNotFoundError(
        f"Project root not found: anchor file '{anchor}' does not exist."
    )


# -----------------------------------------------------------------------------
# Project root (resolved once, treated as immutable)
# -----------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = find_project_root()


# -----------------------------------------------------------------------------
# Directory structure (pure definitions, no I/O)
# -----------------------------------------------------------------------------
DIR_APPLE_HEALTH: Final[Path] = PROJECT_ROOT / "apple_health_export"
DIR_CONVERTED: Final[Path] = PROJECT_ROOT / "converted"
DIR_FEATURES: Final[Path] = PROJECT_ROOT / "features"
DIR_MODELS: Final[Path] = PROJECT_ROOT / "models"
DIR_EXPLAINABILITY: Final[Path] = PROJECT_ROOT / "explainability"
DIR_PREDICTIONS: Final[Path] = PROJECT_ROOT / "predictions"
DIR_REPORT: Final[Path] = PROJECT_ROOT / "report"
DIR_UTILS: Final[Path] = PROJECT_ROOT / "utils"
DIR_NOTEBOOKS: Final[Path] = PROJECT_ROOT / "notebooks"


# -----------------------------------------------------------------------------
# Key input files
# -----------------------------------------------------------------------------
XML_FILENAME: Final[str] = os.getenv("XML_FILENAME", "export.xml")
FILE_APPLE_HEALTH_XML: Final[Path] = DIR_APPLE_HEALTH / XML_FILENAME


def get_daily_path(
    directory: Path,
    prefix: str,
    date_obj: date,
    suffix: str
) -> Path:
    """
    Generate a standardized date-based file path.

    Naming convention:
        {prefix}_YYYYMMDD{suffix}

    Parameters
    ----------
    directory : Path
        Base directory for the file.
    prefix : str
        Semantic identifier of the data type (e.g., 'features', 'events').
    date_obj : date
        Date used for filename generation.
    suffix : str
        File suffix including extension (e.g., '.csv', '.parquet').

    Returns
    -------
    Path
        Full path to the generated file.
    """
    date_str = date_obj.strftime("%Y%m%d")
    return directory / f"{prefix}_{date_str}{suffix}"