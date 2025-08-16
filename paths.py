# HRR_project/paths.py

from pathlib import Path
from datetime import date
import os

def find_project_root(anchor_file: str = ".project_root") -> Path:
    """
    從當前檔案或工作目錄開始，向上尋找包含錨點檔案的目錄。
    這是目前業界最穩健的專案根目錄定位方法。
    """
    try:
        start_path = Path(__file__).resolve().parent
    except NameError:
        start_path = Path.cwd()
    
    for parent in [start_path] + list(start_path.parents):
        if (parent / anchor_file).exists():
            return parent
            
    raise FileNotFoundError(
        f"找不到專案錨點檔案 '{anchor_file}'。請確認它位於專案根目錄中。"
    )

PROJECT_ROOT = find_project_root()

# --- 主要資料夾路徑 ---
DIR_APPLE_HEALTH = PROJECT_ROOT / "apple_health_export"
DIR_CONVERTED = PROJECT_ROOT / "converted"
DIR_FEATURES = PROJECT_ROOT / "features"
DIR_MODELS = PROJECT_ROOT / "models"
DIR_EXPLAINABILITY = PROJECT_ROOT / "explainability"  # <-- 【修正】補上這行
DIR_PREDICTIONS = PROJECT_ROOT / "predictions"
DIR_REPORT = PROJECT_ROOT / "report"
DIR_UTILS = PROJECT_ROOT / "utils"
DIR_NOTEBOOKS = PROJECT_ROOT / "notebooks"

# --- 關鍵輸入檔案 ---
FILE_APPLE_HEALTH_XML = DIR_APPLE_HEALTH / "export.xml"

# --- 動態路徑生成輔助函數 ---
def get_daily_path(directory: Path, data_type: str, date_obj: date, extension: str) -> Path:
    """生成帶有日期的標準化檔案路徑。"""
    date_str = date_obj.strftime("%Y%m%d")
    filename = f"{data_type}_{date_str}{extension}"
    return directory / filename

# --- 當此腳本被直接執行時，用來自我檢查 ---
if __name__ == '__main__':
    print(f"專案根目錄被定義為: {PROJECT_ROOT}")
    assert "HRR_project" in str(PROJECT_ROOT), "專案根目錄看起來不正確。"
    
    print("\n所有主要路徑:")
    print(f"  模型資料夾 (Models): {DIR_MODELS}")
    print(f"  可解釋性資料夾 (Explainability): {DIR_EXPLAINABILITY}")
    
    # 為了能執行檢查，先確保資料夾存在
    for dir_path in [DIR_MODELS, DIR_EXPLAINABILITY]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    assert DIR_EXPLAINABILITY.exists(), "可解釋性資料夾檢查失敗 (Explainability directory check failed)!"
    print("\n路徑檢查完畢，所有主要資料夾均存在。")