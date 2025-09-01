.. HRR Analysis Documentation master file

======================================================
Heart Rate Recovery Analysis for Freediving
======================================================

.. image:: https://img.shields.io/badge/Python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

**A rigorous dual-track validation methodology for personalized physiological monitoring using consumer wearables**

.. note::
   This project demonstrates the application of machine learning in sports physiology, 
   specifically addressing the gap between subjective perception and objective physiological data
   in freediving recovery patterns.

Quick Start
-----------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/billy1030714/freediving-recovery-analysis.git
   cd HRR_project
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the complete integration test to verify the MLOps architecture
   python scripts/integration_test.py

# ... (Table of Contents 保持不變) ...

Executive Summary
-----------------

Project Highlights
~~~~~~~~~~~~~~~~~~

* [cite_start]**N-of-1 Study Design**: 121 standardized measurements from a single subject [cite: 108, 138, 293]
* [cite_start]**Dual-Track Validation**: Novel methodology to establish capability boundaries [cite: 4, 108, 138, 293, 359]
* **Learning Curve Analysis**: Reveals dynamic performance patterns based on sample size
* [cite_start]**100% Reproducible**: Complete MLOps pipeline with version control [cite: 108, 299]

Key Innovation: Dual-Track Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project introduces a rigorous **dual-track validation methodology**:

[cite_start]**Track A: Indicator Validity (R² ≈ 0.92)** [cite: 4, 108, 139]
   Validates the Early Recovery Score (ERS) as an effective descriptive tool using 
   [cite_start]physiological features including recovery-phase data. [cite: 139]

[cite_start]**Track B: Prediction Boundary (R² ≈ -0.1)** [cite: 4, 108, 139]
   Confirms the impossibility of true prediction using only pre-dive baseline features,
   [cite_start]establishing clear capability boundaries for consumer wearables. [cite: 139, 298]

# ... (Technical Architecture 保持不變) ...

Learning Curve Insights
-----------------------

**Analysis of the learning curve** reveals dynamic performance expectations at different stages of data collection. This finding, while not part of the automated CI, provides a valuable heuristic for interpreting model performance:

.. list-table:: Performance Expectations by Sample Size
   :header-rows: 1
   :widths: 20 20 30 30

   * - Samples
     - Expected R²
     - Status
     - Interpretation
   * - < 50
     - 0.57-0.66
     - Unstable
     - Report Only
   * - 50-60
     - 0.66-0.87
     - Transition
     - Relaxed Threshold (R² ≥ 0.65)
   * - **60** - **≈ 0.88**
     - **Jump Point**
     - **Reliable Performance (R² ≥ 0.88)**
   * - > 60
     - > 0.88
     - Stable
     - Converged Performance

# ... (Clinical Implications 和 Author Information 保持不變, 建議將 email 換成真實的) ...