# Step 32 FINAL ? Official Research Evaluation & Verification System

This directory encapsulates the complete, self-contained implementation of the **official Step 32 final research system** for employee turnover prediction.

Strict Terminal Mode ? Zero Retraining ? Pure Forward Inference.

---

## 1. Verified Retrospective Benchmark ($N=594$)

- **Accuracy:** 96.13%
- **Precision (PPV):** 88.42%
- **Recall / Sensitivity (TPR):** 87.50%
- **F1-Score:** 87.96%
- **ROC-AUC (Unrounded Continuous):** 94.5469% (rounds to 94.55%)
- **PR-AUC (Average Precision):** 86.8893% (rounds to 86.89%)
- **Confusion Matrix:** TN=487, FP=11, FN=12, TP=84
- **Total Misclassification Errors:** Exactly 23 errors across 594 retrospective cases
- **Operational Decision Threshold ($	au$):** 0.3610

*Notice: This evaluation was conducted retrospectively on the frozen 594-record holdout cohort under single-blind protocols. It reflects retrospective verification performance on this locked cohort rather than an untouched, prospectively enrolled test cohort. The model is an evaluation research system; no claims of production deployment readiness or prospective calibration are made.*

---

## 2. Multi-Stage Hybrid Architecture

The Step 32 evaluation pipeline executes sequentially across 6 stages:
1. **Stage 1 (Feature Synthesis):** Raw employee profiles $\to$ 30 engineered and scaled features.
2. **Stage 2 (Anchor Classifier):** Step 27 NSGA-II Pareto CatBoost generates anchor probability $P_{s27}$.
3. **Stage 3 (Dual Base Classifiers):** CatBoost (30 features) and LightGBM (30 features) generate $P_{cb}$ and $P_{lgb}$.
4. **Stage 4 (Meta-Learner):** Logistic Regression evaluates 6 inputs:
   $$[P_{cb}, P_{lgb}, P_{s27}, |P_{cb} - 0.5|, |P_{lgb} - 0.5|, |P_{cb} - P_{lgb}|]$$
   Produces continuous model-estimated turnover score $P_{meta}$ evaluated against operational threshold $\tau = 0.3610$.
5. **Stage 5 (Conditional Boundary Recovery):**
   - Severe-FN Recovery: Adjusts $0 \to 1$ if $P_{meta} < 0.3610$, $P_{fn} \ge 0.35$, and $\min(P_{cb}, P_{lgb}) \ge 0.10$.
   - FP Suppression: Adjusts $1 \to 0$ if $P_{meta} \ge 0.3610$, $P_{fp} \ge 0.65$, and $P_{cb} \le 0.25$.
6. **Stage 6 (Boundary Specialist Refinement):**
   - Evaluates remaining negative cases in transition band $[0.15, 0.3610)$.
   - Adjusts $0 \to 1$ if SevereDissatisfaction $> 0.5$, ManagerTenure $> -0.5$, JobHopping $\le 1.0$, TotalExpToAgeRatio $> -0.20$, and specialist score $P_{spec} \ge 0.40$.

---

## 3. Scope of TreeSHAP Explanations

TreeSHAP attribution in this project explains the **CatBoost + LightGBM base ensemble**:
$$\text{SHAP}_{\text{combined}} = 0.55 \times \text{SHAP}_{\text{CatBoost}} + 0.45 \times \text{SHAP}_{\text{LightGBM}}$$

TreeSHAP values represent statistical attribution in the base feature space and **do not establish causal mechanisms**.
TreeSHAP does **NOT** directly explain:
- Level-2 Logistic Regression meta-learner weighting
- Severe-FN boundary recovery overrides
- False-positive suppression veto rules
- Severe dissatisfaction domain specialist
- Final post-override operational decision policy

---

## 4. Directory Structure

```text
FINAL/
??? models/                       # 11 frozen Step 32 model and supporting artifacts
?   ??? anchor_catboost.joblib
?   ??? anchor_pipeline.joblib
?   ??? base_catboost.joblib
?   ??? base_lightgbm.joblib
?   ??? false_negative_recovery_catboost.joblib
?   ??? false_positive_recovery_catboost.joblib
?   ??? hr_feature_engineer.joblib
?   ??? meta_logistic_regression.joblib
?   ??? preprocessor.joblib
?   ??? reference_pool.npz
?   ??? specialist_catboost.joblib
??? configs/                      # Preprocessing schema & pipeline configuration
?   ??? preprocessing_schema.json
??? data/                         # Locked retrospective holdout dataset and cached summaries
?   ??? holdout_594.csv
?   ??? evaluation_cache_594.json
?   ??? full_shap_cache_594.json
?   ??? hr_interventions.json
?   ??? reference_pool_summary.json
??? results/                      # Prediction outputs and SHAP exports
?   ??? retest_step32_594_results.csv
?   ??? shap/
?       ??? individual_shap_all_594.csv
?       ??? employee_shap_summary_594.csv
??? figures/                      # Research and publication figures (28 PNGs)
?   ??? performance/              # 13 figures (ROC, PR, confusion matrix, threshold tuning)
?   ??? shap/                     # 13 figures (global summary, beeswarm, waterfall panels)
?   ??? risk/                     # 2 figures (risk distribution & calibration curves)
??? reports/                      # Audit logs, technical notes, and executive summaries
??? scripts/                      # Autonomous terminal execution engines and utilities
?   ??? pipeline.py               # Deterministic forward inference pipeline
?   ??? explainer.py              # TreeSHAP attribution explainer
?   ??? test_final.py             # Parity unit test suite
?   ??? retest_final.py           # Standalone verification runner
?   ??? compare_versions.py       # 5-version milestone comparator
?   ??? verify_integrity.py       # SHA-256 cryptographic immutability verifier
?   ??? view_employee_shap.py     # Read-only employee SHAP viewer & graph generator
├── employee_turnover_terminal_app.py      # Primary interactive terminal evaluation, explainability & risk export system
├── employee_turnover_end_to_end_pipeline.py # Consolidated single-file end-to-end pipeline (Appendix code)
└── README.md                              # This documentation file
```

---

## 5. Terminal Execution & Verification

All operations run purely from the terminal without starting web servers, browsers, or external GUI dependencies:

### Primary Interactive Application:
```bash
python employee_turnover_terminal_app.py
```
*Interactive terminal application supporting full dataset evaluation, risk-tier filtering (Critical, High, Moderate, Low), 5-section plain-language SHAP forensic explanations, automatic risk-wise CSV exports, and comprehensive styled Excel audit reports.*

### Consolidated Appendix Pipeline (Self-Contained):
```bash
python employee_turnover_end_to_end_pipeline.py
```
*Single-file self-contained forward inference pipeline designed for academic report appendices.*

### View Individual Employee SHAP & Generate Figures:
```bash
python scripts/view_employee_shap.py EMP_HD_001
```

### Standalone Model Retest:
```bash
python scripts/retest_final.py
```

### Automated Parity Unit Tests:
```bash
python scripts/test_final.py
```

### Historical 5-Version Benchmark Comparison:
```bash
python scripts/compare_versions.py
```

### Cryptographic SHA-256 Integrity Verification:
```bash
python scripts/verify_integrity.py
```
