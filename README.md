# Personalized Freediving Recovery Assessment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg)](https://github.com/billy1030714/freediving-recovery-analysis/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning pipeline for analyzing freediving recovery patterns using Apple Watch data, implementing a novel **dual-track validation strategy** that separates algorithm validation from predictive assessment.

## 🎯 Research Objective

This project establishes a rigorous validation framework for consumer wearable health monitoring, using freediving recovery as a physiological model. The core contribution is a **dual-track validation methodology** that scientifically delineates the capabilities and limitations of consumer-grade devices in personalized health assessment.

## 🔬 Key Innovation: Dual-Track Validation

The project's core innovation is a dual-track methodology that systematically separates algorithm validation from scientific predictability testing, providing both practical insights and fundamental scientific boundaries.

### Complete Dual-Track Comparison

| Track | Purpose | Feature Set | ERS R² Score | Performance Drop |
|-------|---------|-------------|:------------:|:---------------:|
| **Product Design** | Algorithm Validation | Pre-dive + **ERS Components** | **0.9175** | - |
| **Research Validation** | Predictability Assessment | **Only** Pre-dive Features | **-0.1015** | **1.0190** (102% decrease) |

This dramatic contrast proves that:
1. **ERS algorithm design is scientifically sound** (Product Track validates the components)
2. **Prediction boundaries are clearly defined** (Research Track shows limits without components)
3. **The dual-track strategy successfully separates algorithm design from predictive assessment**

## 📊 Complete Results Summary

### Product Track Results (short_term mode)
Validates the Early Recovery Score (ERS) as a descriptive tool by including ERS components as features.

| Model          |  R² Score  | MAE    | RMSE   |
| -------------- | :--------: | ------ | ------ |
| **XGBoost (Best)** | **0.9175** | 0.0341 | 0.0472 |
| Random Forest  |   0.8204   | 0.0514 | 0.0697 |
| Ridge          |   0.7226   | 0.0570 | 0.0867 |

### Research Track Results (long_term mode)
Tests true predictability using only pre-dive features under strict leak-free conditions.

| Target Metric  | Best Model    | Best R² Score | Predictability |
| -------------- | ------------- | :-----------: | -------------- |
| **mean_rr_post** | Ridge       | **0.3114**    | ✅ Limited but meaningful |
| **pnn50_post**   | Ridge       | **0.0371**    | ⚠️ Minimal |
| **hrr60**        | Random Forest | **-0.0380**   | ❌ Not predictable |
| **sdnn_post**    | Ridge       | **-0.0732**   | ❌ Not predictable |
| **ERS**          | Random Forest | **-0.1015**   | ❌ Not predictable |
| **rmssd_post**   | Ridge       | **-0.1138**   | ❌ Not predictable |

## 🚀 Quick Start (MLOps Workflow)

### Prerequisites
```bash
# Install Poetry for dependency management
curl -sSL https://install.python-poetry.org | python3 -

# Clone and setup
git clone <repository-url>
cd HRR_project
poetry install
```

### Complete Pipeline Execution

#### Option 1: DVC Pipeline (Recommended)
```bash
# Run full automated pipeline
poetry run dvc repro
```

#### Option 2: Manual Execution
```bash
# Research Track (default)
export TASK_TYPE=long_term
export TARGETS="ERS,rmssd_post"
poetry run python hrr_analysis/04_models.py
poetry run python hrr_analysis/05_explainability.py
poetry run python hrr_analysis/06_predict.py --fast
poetry run python hrr_analysis/07_visualize.py
poetry run python hrr_analysis/08_report.py

# Product Track (for ERS validation)
export TASK_TYPE=short_term  
export TARGETS=ERS
poetry run python hrr_analysis/04_models.py
poetry run python hrr_analysis/05_explainability.py
poetry run python hrr_analysis/08_report.py
```

## 📈 Pipeline Stages

| Stage | Script | Purpose | Key Outputs |
|-------|--------|---------|-------------|
| **Clean** | `01_cleaning.py` | Raw data processing | Heart rate data, apnea events |
| **Features** | `02_features.py` | Feature engineering | ML-ready feature datasets |
| **Train** | `04_models.py` | Model training & evaluation | Trained models, performance metrics |
| **Explain** | `05_explainability.py` | Model interpretation | SHAP plots, feature importance |
| **Predict** | `06_predict.py` | Generate predictions | Predictions, percentile rankings |
| **Visualize** | `07_visualize.py` | Create visualizations | Performance plots, recovery curves |
| **Report** | `08_report.py` | Generate final report | Comprehensive markdown report |

## 🏗️ Architecture

```
HRR_project/
├── hrr_analysis/           # Core pipeline scripts (01-08)
│   ├── 01_cleaning.py     # Data cleaning and preprocessing
│   ├── 02_features.py     # Feature engineering pipeline
│   ├── 04_models.py       # Model training and evaluation
│   ├── 05_explainability.py  # Model interpretation
│   ├── 06_predict.py      # Prediction generation
│   ├── 07_visualize.py    # Visualization creation
│   └── 08_report.py       # Final report generation
├── utils/                  # Shared utilities and data loaders
├── apple_health_export/    # Raw Apple Health data (gitignored)
├── converted/             # Processed heart rate and apnea data (gitignored)
├── features/              # Feature engineering outputs (gitignored)
├── models/                # Trained models and metadata (gitignored)
├── explainability/        # Model interpretation results (gitignored)
├── predictions/           # Prediction outputs and metrics (gitignored)
├── report/                # Final visualizations and reports (gitignored)
├── dvc.yaml              # DVC pipeline configuration
├── params.yaml           # Centralized parameters
└── paths.py              # Path configuration
```

## 🔧 Configuration

### Key Parameters (`params.yaml`)

```yaml
# Data Processing
data_cleaning:
  hr_min: 30
  hr_max: 200

# Feature Engineering  
feature_engineering:
  base_window: [330, 150]
  peak_max_seconds: 120
  hrv_window: [180, 360]

# Model Training
training:
  random_seed: 42
  split_ratio: 0.7
  quality_gate_threshold: 0.05
```

### Environment Variables

- `TASK_TYPE`: `long_term` (research) or `short_term` (product)
- `TARGETS`: Comma-separated list of target variables
- `CI`: Automatically detected for CI/CD environments

## 📁 Key Outputs

### Models Directory (`models/`)
- `best_model.joblib`: Trained model pipeline
- `dataset_card.json`: Complete model metadata with dual-track information
- `feature_schema.json`: Feature specifications
- `target_distribution.json`: Training data distribution for percentile calculations
- `leaderboard.json`: Model comparison results

### Reports Directory (`report/`)
- `final_report.md`: Comprehensive dual-track analysis report
- `figures/`: Performance visualizations and model plots organized by target

## 🔍 Scientific Insights

### Boundary Definition
The research successfully quantified the limits of pre-dive feature-based prediction:
- **Baseline physiological state** (mean_rr_post): Moderately predictable (R² = 0.31)
- **Dynamic recovery metrics** (ERS, RMSSD, SDNN): Not predictable (R² < 0.05)
- **Recovery event characteristics** (HRR60): Not predictable (R² ≈ 0)

### Algorithm Validation
SHAP analysis in the Product Track confirms that `recovery_ratio_60s`, `recovery_ratio_90s`, and `normalized_slope` are the most important ERS components, scientifically validating the algorithm design.

## 🧪 Validation Strategy

### Time-Series Data Split
- **Training**: First 70% of data (chronologically)
- **Validation**: Last 30% of data (chronologically)
- **Rationale**: Prevents future information leakage

### Dual-Track Feature Exclusion
- **Research Track**: Strict exclusion of all post-dive features and ERS components
- **Product Track**: Inclusion of ERS components for algorithm validation
- **Logging**: Comprehensive logging of excluded columns for transparency

## 🚀 CI/CD Integration

### DVC Pipeline Benefits
- **Reproducibility**: Version-controlled data and experiments
- **Automation**: One-command full pipeline execution (`dvc repro`)
- **Dependency Tracking**: Automatic stage dependency resolution
- **CI Integration**: Automated pipeline validation in GitHub Actions

### Pipeline Dependencies
```yaml
stages:
  train:
    deps:
      - hrr_analysis/04_models.py
      - features/features_ml.parquet
      - params.yaml
    outs:
      - models/
```

## 🏥 Clinical Relevance

### Immediate Applications
- Post-exercise recovery assessment for athletes
- Training load optimization in breath-hold sports
- Safety monitoring during freediving activities

### Future Translation Potential
- OSA (Obstructive Sleep Apnea) severity assessment
- COPD exacerbation prediction using similar physiological patterns
- Personalized cardiovascular risk stratification

## 🔧 Technical Stack

**ML Framework:** XGBoost, scikit-learn, Random Forest, Ridge Regression

**MLOps & Automation:**
- **Poetry**: Dependency management & reproducible environments
- **DVC**: Data & pipeline version control
- **MLflow**: Experiment tracking & model logging
- **GitHub Actions**: Continuous integration & automated pipeline execution

**Explainability:** SHAP, LIME, Permutation Importance

**Data Processing:** Pandas, NumPy, feature engineering pipelines

**Validation:** Time-series splits, strict data leakage prevention

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