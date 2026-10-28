# Research Gap Extension: Talebi et al. (2025)

An empirical research extension addressing the future research directions in **Talebi et al. (2025)** (*Journal of Open Innovation: Technology, Market, and Complexity*):
1. Advanced HR-domain feature engineering (burnout interactions, compensation parity, supervisory continuity).
2. Dual-level SHAP explainability (tree log-odds + complete hybrid pipeline probability space `[0, 1]`).
3. Leakage-safe nested cross-validation across all 2,969 records.

---

## 1. Directory Structure

```
research_gap_extension/
├── features.py                 # HRFeatureEngineer transformer with zero-denominator guards
├── train.py                    # 5-fold nested cross-validation and candidate training engine
├── evaluate.py                 # Standalone retrospective holdout evaluation script
├── explain.py                  # Dual-level SHAP explainability generator
├── config.json                 # Experiment configuration and hyperparameters
├── experiment_manifest.json    # Software environment, seeds, and artifact SHA-256 hashes
├── feature_registry.json       # Mathematical formulas and domain rationale for all 40 features
├── metrics.json / metrics.csv  # Full tabular metrics from nested CV and retrospective holdout
├── ablation_results.csv        # 3-arm controlled feature ablation comparison
├── evaluation_results.json     # Detailed evaluation outputs on 594 test records
├── EXPERIMENT_REPORT.md        # Comprehensive academic research report
├── README.md                   # This reproduction guide
│
├── models/                     # Saved candidate model binaries & scalers
│   ├── hr_feature_engineer.joblib
│   ├── scaler.joblib
│   ├── catboost_base.joblib
│   ├── lightgbm_base.joblib
│   ├── anchor_catboost.joblib
│   ├── meta_logistic_regression.joblib
│   ├── boundary_specialist.joblib
│   └── pipeline_schema.json
│
├── shap/                       # High-resolution SHAP explainability visualizations
│   ├── shap_global_bar.png
│   ├── shap_beeswarm.png
│   ├── shap_dependence_overtime_satisfaction.png
│   ├── shap_dependence_manager_tenure.png
│   ├── shap_waterfall_high_risk.png
│   ├── shap_waterfall_low_risk.png
│   ├── shap_complete_pipeline_waterfall.png
│   └── shap_explainability_summary.json
│
└── tests/                      # Automated test suite
    └── test_extension.py
```

---

## 2. Exact Reproduction Commands

All commands are executed from the repository root: `e:\R & D anitgravity`

### Step A: Run 5-Fold Nested Cross-Validation & Candidate Training
```bash
python research_gap_extension/train.py
```
*(Executes 3-arm ablation across all 2,969 records and fits candidate models. Duration: ~197 seconds)*

### Step B: Run Retrospective Holdout Evaluation
```bash
python research_gap_extension/evaluate.py
```
*(Evaluates candidate on 594 test records with pure forward inference)*

### Step C: Generate Complete SHAP Explanations & Plots
```bash
python research_gap_extension/explain.py
```
*(Computes TreeExplainer log-odds attributions and complete pipeline probability attributions)*

### Step D: Run Comprehensive Automated Test Suite
```bash
python -m unittest research_gap_extension/tests/test_extension.py
```
*(Validates formulas, zero guards, target leakage, fold isolation, loading, SHAP additivity, math, and baseline immutability)*

---

## 3. Key Measured Results Summary

| Metric | Arm A: Baseline (30 feats) | Arm B: Enriched All (40 feats) | Arm C: Selected Screened (35 feats) |
| :--- | :---: | :---: | :---: |
| **Nested CV Accuracy** | 89.32 ± 1.29% | 87.98 ± 1.14% | 89.26 ± 1.02% |
| **Nested CV F1-Score** | 67.31 ± 4.22% | 61.33 ± 3.89% | 64.24 ± 3.52% |
| **Total Errors (2,969)**| 317 | 357 (+40 errors) | 319 (+2 errors) |
| **Status** | Historical Baseline | Overfitting / Noise | **Selected Research Extension** |

*Production Champion Preservation: Official Step 32 remains 100% untouched and active in `final_employee_turnover_system/`.*
