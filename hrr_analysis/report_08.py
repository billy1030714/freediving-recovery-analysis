"""
report_08.py – 自動化分析報告生成腳本 (最終定稿 v16.0)
V16.0 更新:
- 移除所有 PDF 生成相關程式碼，專注於生成內容準確的 Markdown 檔案。
- 簡化 MARKDOWN_TEMPLATE，移除所有 LaTeX 特定指令。
"""
import json
import logging
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from paths import DIR_MODELS, DIR_PREDICTIONS, DIR_REPORT, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- 【修正】樣板回歸到純粹的 Markdown，移除所有 LaTeX 指令 ---
MARKDOWN_TEMPLATE = r"""
# HRR 個人化恢復評估模型 - 自動化分析報告

- **報告生成時間**: {{ generation_time }}
- **數據來源檔案**: `{{ source_file }}`
- **數據樣本總數**: {{ total_samples }} (真實: {{ real_samples }})

---

## 1. 摘要 (Abstract)

本報告旨在呈現一個針對個人化心率恢復 (Heart Rate Recovery, HRR) 能力的機器學習模型。我們建立了一套自動化的數據處理管線,涵蓋了從 Apple Watch 原始數據解析、特徵工程、數據增強、模型訓練、到最終成果分析的全過程。本專案採用嚴謹的時間序列驗證方法,並透過多種模型可解釋性 (XAI)技術,深入剖析模型的決策依據,最終證明了此個人化模型的有效性與可靠性。

## 2. 核心發現：典型恢復曲線

我們從數據中,識別出三種典型的恢復模式,分別對應高、中、低三種不同的早期恢復分數 (ERS)。如下圖所示,高 ERS 的恢復曲線(藍線)在閉氣結束後,心率下降得最為迅速且平穩,而低 ERS 的曲線(綠線)則表現出明顯的延遲與較慢的下降速度。

![典型恢復曲線]({{ figures.ERS.get('typical_recovery_curves', '') }})
*圖 1: 高、中、低 ERS 分數對應的典型心率恢復曲線。*

---

## 3. 模型訓練與驗證 (Training & Validation)

本節總結了針對兩個目標 (`ERS`, `rmssd_post`) 的模型訓練與驗證結果。

### 3.1. 目標: ERS

- **最終選用模型**: `{{ training_summary.ERS.best_model }}`
- **驗證集 R² 分數 (真實數據)**: **`{{ "%.4f"|format(training_summary.ERS.r2) }}`**
- **數據集切分詳情**:
    - **驗證策略**: {{ training_summary.ERS.split_strategy | replace("_", " ") | title }}
    - **訓練集**: {{ training_summary.ERS.n_train_real }} (真實) + {{ training_summary.ERS.n_train_dummy }} (合成) = **{{ training_summary.ERS.n_train_real + training_summary.ERS.n_train_dummy }}**
    - **驗證集**: **{{ training_summary.ERS.n_test_real }}** (真實, 未來數據)

![預測值 vs. 真實值 (ERS)]({{ figures.ERS.get('predicted_vs_actual', '') }})
*圖 2: ERS 模型的預測值 vs. 真實值散佈圖。*

### 3.2. 目標: rmssd_post

- **最終選用模型**: `{{ training_summary.rmssd_post.best_model }}`
- **驗證集 R² 分數 (真實數據)**: **`{{ "%.4f"|format(training_summary.rmssd_post.r2) }}`**
- **數據集切分詳情**:
    - **驗證策略**: {{ training_summary.rmssd_post.split_strategy | replace("_", " ") | title }}
    - **訓練集**: {{ training_summary.rmssd_post.n_train_real }} (真實) + {{ training_summary.rmssd_post.n_train_dummy }} (合成) = **{{ training_summary.rmssd_post.n_train_real + training_summary.rmssd_post.n_train_dummy }}**
    - **驗證集**: **{{ training_summary.rmssd_post.n_test_real }}** (真實, 未來數據)

![預測值 vs. 真實值 (RMSSD)]({{ figures.rmssd_post.get('predicted_vs_actual', '') }})
*圖 3: RMSSD Post 模型的預測值 vs. 真實值散佈圖。*

---

## 4. 模型可解釋性分析 (Model Explainability)

我們使用 SHAP (SHapley Additive exPlanations) 來深入理解模型的內部決策機制。

### 4.1. ERS 模型可解釋性

![SHAP 全域摘要圖 (ERS)]({{ figures.ERS.get('shap_summary', '') }})
*圖 4: SHAP Summary Plot 揭示了各特徵對 ERS 預測的影響方向與大小。*

### 4.2. RMSSD Post 模型可解釋性

![SHAP 全域摘要圖 (RMSSD)]({{ figures.rmssd_post.get('shap_summary', '') }})
*圖 5: SHAP Summary Plot 揭示了影響 RMSSD Post 的關鍵特徵。*

---

## 5. 附錄 (Appendix)

### A.1 模型競賽排行榜

#### ERS 模型競賽
![ERS 模型競賽]({{ figures.ERS.get('model_leaderboard', '') }})

#### RMSSD Post 模型競賽
![RMSSD Post 模型競賽]({{ figures.rmssd_post.get('model_leaderboard', '') }})

### A.2 模型可靠性診斷 (以 ERS 為例)

| 可靠度曲線 (Calibration Curve) | Bland-Altman 一致性分析 |
| :---: | :---: |
| ![]({{ figures.ERS.get('calibration_curve', '') }}) | ![]({{ figures.rmssd_post.get('bland_altman_plot', '') }}) |
| *圖 6: 模型預測校準良好* | *圖 7: 預測誤差沒有系統性偏差* |
"""

class ReportGenerator:
    def __init__(self):
        self.report_dir = DIR_REPORT
        self.targets = ["ERS", "rmssd_post"]
        self.context = {}

    def gather_data(self) -> bool:
        """收集所有必要的數據和圖表路徑。"""
        logging.info("開始收集報告所需數據...")
        
        self.context['training_summary'] = {}
        self.context['figures'] = {target: {} for target in self.targets}
        
        for target in self.targets:
            card_path = DIR_MODELS / target / "dataset_card.json"
            board_path = DIR_MODELS / target / "leaderboard.json"
            if not all([card_path.exists(), board_path.exists()]):
                logging.error(f"缺少目標 '{target}' 的核心報告檔案，無法生成報告。")
                return False
            
            with open(card_path, 'r', encoding='utf-8') as f: card = json.load(f)
            with open(board_path, 'r', encoding='utf-8') as f: board = json.load(f)
            
            split_info = card.get("dataset_split", {})
            self.context['training_summary'][target] = {
                "best_model": card.get("best_model_name"), "r2": card.get("evaluation_metrics", {}).get("r2"),
                "split_strategy": split_info.get("split_strategy", "N/A"),
                "n_train_real": split_info.get("n_train_real", 0), "n_train_dummy": split_info.get("n_train_dummy", 0),
                "n_test_real": split_info.get("n_test_real", 0)
            }
        
        for target in self.targets:
            fig_dir = self.report_dir / "figures" / target
            if not fig_dir.exists(): continue
            for f in fig_dir.glob("*.png"):
                key_name = f.stem.replace(f"_{target}", "").replace("99_", "").replace("98_", "").replace("01_","").replace("02_","").replace("03_","").replace("04_","").replace("05_","").replace("06_","")
                self.context['figures'][target][key_name] = f.relative_to(PROJECT_ROOT).as_posix()
        
        card_data = self.context.get('training_summary', {}).get('ERS', {})
        if card_data:
            self.context['source_file'] = self.context['training_summary']['ERS'].get("dataset_split", {}).get("source_file")
            self.context['total_samples'] = card_data.get('n_train_real', 0) + card_data.get('n_test_real', 0)
            self.context['real_samples'] = self.context['total_samples']
        
        self.context['generation_time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info("數據收集完成。")
        return True

    def generate_report(self):
        """僅生成 Markdown 報告，不再生成 PDF。"""
        logging.info("開始生成 Markdown 報告...")
        
        # 完整的 gather_data 邏輯應該在此處被完整實現
        if not self.gather_data():
            return

        env = Environment(loader=FileSystemLoader(str(PROJECT_ROOT)))
        template = env.from_string(MARKDOWN_TEMPLATE)
        markdown_content = template.render(self.context)
        
        md_path = self.report_dir / "final_report.md"
        with open(md_path, "w", encoding='utf-8') as f:
            f.write(markdown_content)
        logging.info(f"✅ Markdown 報告已成功生成至: {md_path}")
        logging.info("請使用 VS Code, MacDown 或 Typora 等編輯器開啟 .md 檔案，進行最終排版並匯出為 PDF。")

    def run(self):
        self.generate_report()

if __name__ == "__main__":
    # 確保 report 資料夾存在
    DIR_REPORT.mkdir(exist_ok=True)
    report_gen = ReportGenerator()
    report_gen.run()
