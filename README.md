# Personalized Freediving Recovery Assessment

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[![CI Validation](https://github.com/billy1030714/freediving-recovery-analysis/actions/workflows/main.yml/badge.svg)](https://github.com/billy1030714/freediving-recovery-analysis/actions)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning pipeline for analyzing freediving recovery patterns using Apple Watch data, implementing a novel **dual-track validation strategy** that separates algorithm validation from predictive assessment.

## 🎯 Research Objective

This project establishes a rigorous validation framework for consumer wearable health monitoring, using freediving recovery as a physiological model. The core contribution is a **dual-track validation methodology** that scientifically delineates the capabilities and limitations of consumer-grade devices in personalized health assessment.

## 🔬 Key Innovation: Dual-Track Validation

The project's core innovation is a dual-track methodology that systematically separates algorithm validation from scientific predictability testing.

### Complete Dual-Track Comparison

| Track | Purpose | Feature Set | ERS R² Score | Performance Drop |
|-------|---------|-------------|:------------:|:---------------:|
| **Product Design** | Algorithm Validation | Pre-dive + **ERS Components** | **0.9175** | - |
| **Research Validation** | Predictability Assessment | **Only** Pre-dive Features | **-0.1015** | **1.0190** (102% decrease) |

This dramatic contrast proves that:
1. **ERS algorithm design is scientifically sound** (Product Track validates the components)
2. **Prediction boundaries are clearly defined** (Research Track shows limits without components)
3. The dual-track strategy successfully separates algorithm design from predictive assessment

## 📊 Complete Results Summary

### Product Track Results (short_term mode)
Validates the Early Recovery Score (ERS) as a descriptive tool. This is the core logic validated by our automated CI pipeline.

| Model | R² Score | MAE | RMSE |
|-------|:--------:|:---:|:----:|
| **XGBoost (Best)** | **0.9175** | **0.0341** | **0.0472** |
| Random Forest | 0.8204 | 0.0514 | 0.0697 |
| Ridge | 0.7226 | 0.0570 | 0.0867 |

### Research Track Results (long_term mode)
Tests true predictability using only pre-dive features. These results define the scientific boundaries of the wearable device.

| Target Metric | Best Model | Best R² Score | Predictability |
|---------------|------------|:-------------:|----------------|
| **mean_rr_post** | Ridge | **0.3114** | ✅ Limited but meaningful |
| **pnn50_post** | Ridge | **0.0371** | ⚠️ Minimal |
| **hrr60** | Random Forest | **-0.0380** | ❌ Not predictable |
| **sdnn_post** | Ridge | **-0.0732** | ❌ Not predictable |
| **ERS** | Random Forest | **-0.1015** | ❌ Not predictable |
| **rmssd_post** | Ridge | **-0.1138** | ❌ Not predictable |

## 🚀 Quick Start

### Prerequisites
```bash
# Install Poetry for dependency management
curl -sSL https://install.python-poetry.org | python3 -

# Clone and setup
git clone https://github.com/billy1030714/freediving-recovery-analysis.git
cd freediving-recovery-analysis
poetry install
```

### Pipeline Execution

For full-scale experiments with large datasets, DVC is used locally to manage the pipeline.

```bash
# Run the full automated pipeline for all stages
poetry run dvc repro
```

To run specific tracks manually:

```bash
# Product Track (validates ERS algorithm design - Core Validation)
export TASK_TYPE=short_term
export TARGETS=ERS
poetry run python hrr_analysis/02_features.py
poetry run python hrr_analysis/04_models.py
poetry run python hrr_analysis/08_report.py

# Research Track (strict predictability assessment)
export TASK_TYPE=long_term
export TARGETS="ERS,rmssd_post"
poetry run dvc repro
```

## 🏗️ Architecture & Pipeline

The project follows a modular pipeline structure managed by `dvc.yaml`.

| Stage | Script | Purpose |
|-------|--------|---------|
| **Clean** | `01_cleaning.py` | Raw data processing |
| **Features** | `02_features.py` | Feature engineering |
| **Train** | `04_models.py` | Model training & evaluation |
| **Explain** | `05_explainability.py` | Model interpretation |
| **Predict** | `06_predict.py` | Prediction generation |
| **Visualize** | `07_visualize.py` | Visualization creation |
| **Report** | `08_report.py` | Final report generation |

```
freediving-recovery-analysis/
├── hrr_analysis/           # Core pipeline scripts
├── converted/              # Processed data (gitignored)
├── features/               # Feature outputs (gitignored)
├── models/                 # Trained models (gitignored)
├── explainability/         # Model interpretation results (gitignored)
├── predictions/            # Prediction outputs (gitignored)
├── report/                 # Final reports (gitignored)
├── scripts/                # Validation and testing scripts
├── dvc.yaml                # DVC pipeline configuration
├── params.yaml             # Centralized parameters
└── paths.py                # Path configuration and project root detection
```

## 🔧 Technical Stack

**ML Framework:** XGBoost, scikit-learn, Random Forest, Ridge Regression

**MLOps & Automation:**
- **Poetry**: Dependency management & reproducible environments
- **DVC**: Pipeline version control (for local experiments)
- **GitHub Actions**: Continuous integration for core algorithm validation
- **MLflow**: Experiment tracking & model logging

**Configuration Architecture:**
- **paths.py**: Centralized path management and project root detection
- **params.yaml**: Model hyperparameters and pipeline configuration
- **Direct constants**: Critical validation constants embedded in scripts for simplicity

**Explainability:** SHAP, LIME, Permutation Importance

**Data Processing:** Pandas, NumPy, feature engineering pipelines

**Validation:** Time-series splits, strict data leakage prevention

## 🧪 Validation Strategy

### Time-Series Data Split
- **Training**: First 70% of data (chronologically)
- **Validation**: Last 30% of data (chronologically)
- **Rationale**: Prevents future information leakage

### Dual-Track Feature Exclusion
- **Research Track**: Strict exclusion of all post-dive features and ERS components
- **Product Track**: Inclusion of ERS components for algorithm validation
- **CI Validation**: Automated testing of Product Track core logic

## 🔍 Scientific Insights

### Boundary Definition
The research successfully quantified the limits of pre-dive feature-based prediction:
- **Baseline physiological state** (mean_rr_post): Moderately predictable (R² = 0.31)
- **Dynamic recovery metrics** (ERS, RMSSD, SDNN): Not predictable (R² < 0.05)
- **Recovery event characteristics** (HRR60): Not predictable (R² ≈ 0)

### Algorithm Validation
SHAP analysis in the Product Track confirms that `recovery_ratio_60s`, `recovery_ratio_90s`, and `normalized_slope` are the most important ERS components, scientifically validating the algorithm design.

## 🏥 Clinical Relevance

### Immediate Applications
- Post-exercise recovery assessment for athletes
- Training load optimization in breath-hold sports
- Safety monitoring during freediving activities

### Future Translation Potential
- OSA (Obstructive Sleep Apnea) severity assessment
- COPD exacerbation prediction using similar physiological patterns
- Personalized cardiovascular risk stratification

## 👤 Author

**Yi-Chuan Su (蘇翊銓)**

🎓 B.S. Nursing, China Medical University (2024)  
🏥 Registered Nurse (Taiwan)  
🏊 AIDA4 & Molchanovs W3 Freediving Instructor Assistant

📧 Contact: Via GitHub Issues

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- Freediving community for data collection support
- Open source ML community for robust tools and libraries
- Apple Health team for comprehensive data export capabilities
- scikit-learn, SHAP, and DVC communities for enabling reproducible ML research

---

*This project demonstrates how rigorous scientific methodology can simultaneously advance practical applications and fundamental understanding in wearable health technology.*