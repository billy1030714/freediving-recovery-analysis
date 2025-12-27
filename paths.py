# HRR_project/paths.py

from pathlib import Path
from datetime import date
import os

def find_project_root(anchor_file: str = ".project_root") -> Path:
    """
    ROBUST PROJECT ROOT DETECTION:
    
    Algorithm:
    1. Start from current file's directory (or cwd if __file__ unavailable)
    2. Search upward through parent directories
    3. Look for anchor file (.project_root) in each directory
    4. Return first match or raise FileNotFoundError
    
    This cross-platform approach works regardless of execution context.
    """
    try:
        start_path = Path(__file__).resolve().parent
    except NameError:
        start_path = Path.cwd()
    
    for parent in [start_path] + list(start_path.parents):
        if (parent / anchor_file).exists():
            return parent
            
    raise FileNotFoundError(
        f"Cannot find the project anchor file '{anchor_file}'. Please ensure it is located in the project root directory."
    )

PROJECT_ROOT = find_project_root()

# --- Main folder paths ---
DIR_APPLE_HEALTH = PROJECT_ROOT / "apple_health_export"
DIR_CONVERTED = PROJECT_ROOT / "converted"
DIR_FEATURES = PROJECT_ROOT / "features"
DIR_MODELS = PROJECT_ROOT / "models"
DIR_EXPLAINABILITY = PROJECT_ROOT / "explainability"
DIR_PREDICTIONS = PROJECT_ROOT / "predictions"
DIR_REPORT = PROJECT_ROOT / "report"
DIR_UTILS = PROJECT_ROOT / "utils"
DIR_NOTEBOOKS = PROJECT_ROOT / "notebooks"

# --- Key input files ---
# determine by env
xml_filename = os.getenv("XML_FILENAME", "export.xml")
FILE_APPLE_HEALTH_XML = DIR_APPLE_HEALTH / xml_filename

# --- Helper function for dynamic path generation ---
def get_daily_path(directory: Path, data_type: str, date_obj: date, extension: str) -> Path:
    """Generate a standardized file path with date."""
    date_str = date_obj.strftime("%Y%m%d")
    filename = f"{data_type}_{date_str}{extension}"
    return directory / filename

# --- Self-check when this script is executed directly ---
if __name__ == '__main__':
    print(f"Project root directory is defined as: {PROJECT_ROOT}")
    assert "HRR_project" in str(PROJECT_ROOT), "The project root directory seems incorrect."
    
    print("\nAll main paths:")
    print(f"  Models directory: {DIR_MODELS}")
    print(f"  Explainability directory: {DIR_EXPLAINABILITY}")
    
    # Ensure folders exist for checking
    for dir_path in [DIR_MODELS, DIR_EXPLAINABILITY]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    assert DIR_EXPLAINABILITY.exists(), "Explainability directory check failed!"
    print("\nPath check complete, all main directories exist.")
    