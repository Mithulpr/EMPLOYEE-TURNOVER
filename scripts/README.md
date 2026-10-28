# Workspace Scripts Directory

## Overview
This directory contains essential standalone research, benchmarking, and verification scripts for the Employee Turnover Prediction project.

## Preserved Core Scripts

1. **`run_track1_step37_explainable_extension.py`**
   - *Protected Baseline Artifact* (referenced by system integration test suite).
   - Generates and validates explainable AI extension pipelines.

2. **`run_track1_step37_followup_hardening.py`**
   - *Protected Baseline Artifact* (referenced by system integration test suite).
   - Scientific hardening and calibration verification script.

3. **`test_step37_scientific_hardening.py`**
   - *Protected Baseline Artifact* (referenced by system integration test suite).
   - Automated regression test suite for scientific hardening baselines.

4. **`generate_corrected_roc_chart.py`**
   - Research visualization script generating the continuous unrounded ROC-AUC curve (94.55%) with 300 DPI publication standards.

5. **`run_experiment_b_base_paper.py`**
   - Reimplementation benchmark of the published Talebi et al. (2025) GA + LightGBM baseline on the 2,969-profile dataset.

6. **`run_experiment_c_adaptive_lgbm.py`**
   - Controlled comparative benchmark evaluating adaptive LightGBM on the standardized cohort.

7. **`run_benchmark_reproduction_audit.py`**
   - Comprehensive multi-model benchmark reproduction audit comparing baseline algorithms against the Two-Stage Selective Classification architecture.

8. **`run_eda_validation.py`**
   - Exploratory data analysis and feature distribution validation tool.

9. **`evaluate_step32.py`**
   - Standalone evaluator for the Step 32 architecture.
