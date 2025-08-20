# Personalized Freediving Recovery Analysis & ML Feasibility Study

This project originates from my dual background as a **Registered Nurse and an Advanced Freediving Instructor**. In both settings, I observed that assessing recovery status often relies on subjective feelings or simplistic formulas. To address this, this research establishes an **industrial-grade, automated data pipeline** to rigorously investigate the feasibility of creating a personalized recovery model using Apple Watch data.

The project evolved from a standard predictive modeling task into a deeper scientific inquiry, culminating in a **"Dual-Track" methodology** that separates **product-oriented algorithm design** from **rigorous scientific validation**.

## 🔬 Core Scientific Finding: The Predictability Boundary

The central outcome of this research is the clear, quantitative delineation of the **predictability boundary** for recovery metrics using wearable-based heart rate data. After implementing a strict, leak-free validation pipeline (our "Research Track"), the results conclusively show that neither short-term nor long-term recovery metrics can be reliably predicted from pre-dive features.

The R² scores for all target metrics are near or below zero, indicating that the models perform no better than a naive baseline. This "negative result" is the core scientific contribution, proving that the predictive signal for these complex phenomena is not present in the current feature set.

## 🎯 The Dual-Track Methodology

To reconcile the needs of algorithm design with the rigor of scientific validation, this project implemented a **"Dual-Track" strategy** controlled by environment variables.

### 🔬 Track A: Product Design (`TASK_TYPE="short_term"`)
- **Goal:** Design and validate the Early Recovery Score (ERS) algorithm
- **Method:** Train models with ERS components **intentionally included** as features
- **Purpose:** Use SHAP analysis to validate that chosen components are most important factors
- **Result:** Provides data-driven justification for the ERS formula

### 🔍 Track B: Research Validation (`TASK_TYPE="long_term"`)
- **Goal:** Test true predictability of recovery metrics under strict conditions
- **Method:** Train models with **extremely strict feature exclusion** (no post-dive features)
- **Purpose:** Quantify the honest predictability boundaries
- **Result:** Generates the core scientific findings showing prediction limitations

## 🛠 Tech Stack & Highlights

- **Dual-Track Experimental Design:** Sophisticated methodology separating algorithm validation from scientific testing
- **Automated MLOps Pipeline:** Fully automated, reproducible workflow from raw data to final report
- **Explainable AI (XAI):** Strategic use of SHAP, LIME, and permutation importance
- **Rigorous Time-Series Validation:** Strict 70/30 chronological split preventing data leakage
- **Professional Python Packaging:** Structured as installable package with proper dependency management

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Apple Health export data (`export.xml`)
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/HRR_project.git
cd HRR_project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and package
pip install -r requirements.txt
pip install -e .

# Create project anchor file
touch .project_root
```

### Basic Setup
1. Place your Apple Health `export.xml` in the `apple_health_export/` folder
2. Run the data preparation pipeline:
```bash
python hrr_analysis/cleaning_01.py
python hrr_analysis/features_02.py
```

## 📊 Reproducing the Research Results

### 🔬 Research Track (Main Scientific Findings)
This generates the core scientific results showing prediction limitations:

```bash
# Set environment for Research Track
export TASK_TYPE="long_term"
export TARGETS="ERS,rmssd_post,sdnn_post,pnn50_post,mean_rr_post,hrr60"

# Run the complete research pipeline
python hrr_analysis/models_04.py
python hrr_analysis/explainability_05.py
python hrr_analysis/predict_06.py
python hrr_analysis/visualize_07.py
python hrr_analysis/report_08.py
```

### 🎯 Product Track (Algorithm Validation)
This validates the ERS algorithm design:

```bash
# Set environment for Product Track
export TASK_TYPE="short_term"
export TARGETS="ERS"

# Run algorithm validation
python hrr_analysis/models_04.py
python hrr_analysis/explainability_05.py

# Generate final visualizations and report
python hrr_analysis/visualize_07.py
python hrr_analysis/report_08.py
```

## 🧪 Optional: Data Augmentation & Validation

```bash
# Generate synthetic samples for data balancing
python hrr_analysis/augmentation_03.py --input features/features.csv

# Validate augmentation effectiveness
python validate_09.py
```

## 📁 Project Structure

```
HRR_project/
├── .gitignore                      # Git ignore patterns
├── .project_root                   # Project anchor file
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation script
├── paths.py                        # Central path configuration
├── validate_09.py                  # Augmentation validation (optional)
│
├── assets/                         # Static assets
│   └── pipeline_diagram.png        # Architecture diagram
│
├── hrr_analysis/                   # Main analysis pipeline
│   ├── __init__.py
│   ├── cleaning_01.py              # Data cleaning and preprocessing
│   ├── features_02.py              # Feature engineering
│   ├── augmentation_03.py          # Data augmentation (optional)
│   ├── models_04.py                # Model training and evaluation
│   ├── explainability_05.py        # XAI analysis (optional)
│   ├── predict_06.py               # Prediction and explanation
│   ├── visualize_07.py             # Visualization generation
│   └── report_08.py                # Final report generation
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── data_loader.py              # Centralized data loading
│   ├── mi.py                       # Mutual information analysis
│   └── score_utils.py              # ERS computation utilities
│
├── tests/                          # Unit tests
│   ├── __init__.py
│   └── test_score_utils.py         # Score computation tests
│
├── apple_health_export/            # Apple Health data (gitignored)
│   └── export.xml                  # Raw Apple Health export
│
├── converted/                      # Processed data files (gitignored)
│   ├── hr_YYYYMMDD.csv            # Daily heart rate data
│   └── apnea_events_YYYYMMDD.csv  # Daily apnea events
│
├── features/                       # Engineered features (gitignored)
│   ├── features.csv               # Base feature dataset
│   ├── features_ml.parquet        # ML-ready format
│   ├── features_aug.csv           # Augmented dataset (optional)
│   └── features_ml_aug.parquet    # Augmented ML format (optional)
│
├── models/                         # Trained models (gitignored)
│   ├── ERS/                       # ERS target models
│   │   ├── best_model.joblib      # Best performing model
│   │   ├── dataset_card.json      # Model metadata
│   │   ├── feature_schema.json    # Feature specifications
│   │   ├── leaderboard.json       # Model comparison results
│   │   └── target_distribution.json # Training data distribution
│   └── [other_targets]/           # Other target metric models
│
├── explainability/                 # XAI analysis results (gitignored)
│   └── [target]/                  # Per-target explanations
│       ├── shap_summary_[target].png
│       ├── lime_instance_[target].png
│       ├── permutation_importance_[target].png
│       └── pdp_top4_[target].png
│
├── predictions/                    # Prediction results (gitignored)
│   └── [target]/                  # Per-target predictions
│       ├── preds.csv              # Prediction results
│       ├── preds.parquet          # Prediction results (optimized)
│       ├── metrics.json           # Evaluation metrics
│       └── explain/               # Prediction explanations
│
└── report/                         # Final reports and figures (gitignored)
    ├── final_report.md            # Complete analysis report
    └── figures/                   # Generated visualizations
        ├── 00_final_summary_comparison.png
        └── [target]/              # Per-target figures
            ├── 02_predicted_vs_actual.png
            ├── 05_feature_importance_[model].png
            ├── 06_shap_summary.png
            └── 99_model_leaderboard.png
```

## 🔧 Environment Variables Reference

| Variable | Values | Purpose | Default |
|----------|--------|---------|---------|
| `TASK_TYPE` | `"short_term"` / `"long_term"` | Controls dual-track methodology | `"long_term"` |
| `TARGETS` | Comma-separated target names | Specifies which metrics to model | `"ERS,rmssd_post"` |
| `DATA_FILE` | Path to feature file | Override default data file search | Auto-detected |

### Common Target Combinations

```bash
# Research Track - All recovery metrics
export TARGETS="ERS,rmssd_post,sdnn_post,pnn50_post,mean_rr_post,hrr60"

# Product Track - ERS validation only
export TARGETS="ERS"

# Focused analysis - HRV metrics only
export TARGETS="rmssd_post,sdnn_post,pnn50_post"
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test individual modules
python -m pytest tests/test_score_utils.py -v
```

## 📈 Expected Outputs

### Research Track Results
- **Low R² scores** (near zero) across all targets
- **Boundary quantification** of prediction limits
- **Scientific validation** of methodology

### Product Track Results
- **High R² score** for ERS (>0.95)
- **SHAP validation** of ERS components
- **Algorithm justification** through explainability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the intersection of clinical nursing practice and freediving instruction
- Built on the scientific principle that "negative results" are valuable contributions
- Designed with reproducibility and transparency as core values

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{su2024hrr,
  title={Personalized Freediving Recovery Analysis: A Machine Learning Feasibility Study},
  author={Su, Yi-Chuan},
  year={2024},
  url={https://github.com/yourusername/HRR_project}
}
```

---

**Note**: This project demonstrates how rigorous scientific methodology can simultaneously advance practical applications and fundamental understanding in wearable health technology.