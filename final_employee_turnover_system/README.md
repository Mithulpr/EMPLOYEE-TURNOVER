# Explainable Employee Turnover Prediction System

An empirical machine learning research framework for retrospective employee turnover prediction, featuring local SHAP attribution decomposition, pre-inference career timeline invariant validation, statistical drift monitoring, and operational HR intervention tracking.

---

## 1. Project Overview & Research Objectives

The **Explainable Employee Turnover Prediction System** investigates proactive employee turnover prediction on a frozen retrospective cohort ($N=594$). By synthesizing structured HR attributes with nonlinear domain interactions, the research framework evaluates classification accuracy while maintaining local attribution interpretability.

### Core Contributions:
1. **Multi-Stage Stacking Architecture (FINAL Version):** Combines diverse gradient boosting architectures (CatBoost, LightGBM) with an anchor model (Step 27 Pareto NSGA-II) via a Level-2 Logistic Regression meta-learner, supplemented by conditional boundary recovery rules.
2. **Comprehensive Five-Version Progression:** Traces error reduction from the foundational single-model baseline (**Champion 4**, 30 errors) down to the official research champion (**FINAL**, 23 errors).
3. **Rigorous Nested Cross-Validation:** Demonstrates feature ablation across 3 arms on the complete cohort using nested cross-validation.
4. **Governance & Responsible AI:** Enforces logical timeline invariants, batch drift monitoring (PSI and Two-Sample KS with FDR correction), demographic parity audits (DIR = 0.951), and non-causal attribution disclosures.

---

## 2. Official Five-Version Comparison Matrix

All five official versions were evaluated on the **locked 594-record retrospective holdout cohort** under single-blind protocols:

| Priority | Official Name | Original Name | Architecture Description | Retrospective Accuracy | Total Errors | Confusion Matrix (TN, FP, FN, TP) | F1-Score | ROC-AUC |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **FINAL** | Step 32 | Stacking Meta-Learner + Boundary Specialist | **96.13%** | **23** | 487, 11, 12, 84 | **87.96%** | **94.55%** |
| **2** | **CHAMPION 1** | Step 31 v2 | Step 28 Backbone + Precision FN Boundary Recovery | **95.96%** | **24** | 486, 12, 12, 84 | **87.50%** | **94.55%** |
| **3** | **CHAMPION 2** | Step 30 | Dual Meta-Learner + Severe-FN Recovery Rule | **95.79%** | **25** | 487, 11, 14, 82 | **86.77%** | **94.55%** |
| **4** | **CHAMPION 3** | Step 28 | Logistic Meta-Learner over CatBoost + LightGBM + Step 27 | **95.62%** | **26** | 487, 11, 15, 81 | **86.17%** | **94.55%** |
| **5** | **CHAMPION 4** | Stage 6 / Review-2 | Tuned Single-Model CatBoost (Depth 6 / Balanced, $\tau=0.4850$) | **94.95%** | **30** | 487, 11, 19, 77 | **83.70%** | **92.32%** |

> **Methodological Disclosure:** The 594-record holdout dataset is retrospective and was frozen during development. Reported metrics reflect retrospective evaluation on this locked cohort rather than an untouched, prospectively enrolled test cohort. The model is an evaluation research system; no claims of prospective production deployment readiness are made.

---

## 3. Scope of TreeSHAP Explanations

TreeSHAP attribution in this project explains the **CatBoost + LightGBM base ensemble**:
$$\text{SHAP}_{\text{combined}} = 0.55 \times \text{SHAP}_{\text{CatBoost}} + 0.45 \times \text{SHAP}_{\text{LightGBM}}$$

TreeSHAP values represent statistical attribution in base feature space and **do not establish causal mechanisms**.
TreeSHAP does **NOT** directly explain:
- Level-2 Logistic Regression meta-learner
- False-negative boundary recovery overrides
- False-positive suppression veto rules
- Severe dissatisfaction domain specialist
- Final post-override operational decision policy

---

## 4. Standardized Project Structure

The project has been organized into a clean, modular structure with zero retraining and zero duplicate files:

```text
final_employee_turnover_system/
?
??? CHAMPIONS/                    # Historical Precursor Versions (Champions 1 - 4)
?   ??? CHAMPION_1_STEP31/        # Step 31 Candidate v2 (24 errors, 95.96% Acc)
?   ??? CHAMPION_2_STEP30/        # Step 30 Severe-FN Recovery (25 errors, 95.79% Acc)
?   ??? CHAMPION_3_STEP28/        # Step 28 Stacking Meta-Ensemble (26 errors, 95.62% Acc)
?   ??? CHAMPION_4_STAGE6/        # Stage 6 / Review-2 Primary Baseline (30 errors, 94.95% Acc)
?   ??? README.md                 # Precursor progression guide
?
??? FINAL/                        # Official Step 32 Final Research System
?   ??? models/                   # 11 frozen Step 32 model and supporting artifacts
?   ??? configs/                  # Preprocessing schema definition
?   ??? data/                     # Locked retrospective holdout dataset (594 rows)
?   ??? results/                  # Retest predictions & results/shap/ exports
?   ??? figures/                  # Publication figures (performance/, shap/, risk/)
?   ??? reports/                  # Comprehensive audit reports and summaries
?   ??? scripts/                  # Standalone terminal utilities & inference engines
?   ??? terminal_research_console.py  # Research evaluation console
?   ??? run_final.py              # Automated terminal evaluation runner
?   ??? README.md                 # FINAL model technical documentation
?
??? archive/                      # Preserved non-active assets (Zero-Deletion)
?   ??? ablation_study/           # 3-arm nested CV ablation research experiments
?   ??? legacy_web_app/           # Preserved Flask UI (app/, static/, templates/)
??? requirements.txt              # Standard Python dependencies
??? README.md                     # This master documentation file
```

---

## 5. Quickstart & Terminal Usage

The framework is configured strictly for **terminal execution** with zero GUI, browser, or server dependencies.

### 1. Execute FINAL System Evaluation:
```bash
cd FINAL
python run_final.py
```
This runs a pure deterministic forward inference pass across all 594 retrospective cases, verifying exact match with the canonical research benchmarks (96.13% Acc, 87.96% F1, 23 errors).

### 2. Run Comprehensive Research Audit:
```bash
cd FINAL
python terminal_research_console.py --evaluate
```

### 3. Evaluate Single Employee & Generate SHAP Graphs:
```bash
cd FINAL
python terminal_research_console.py --employee EMP_HD_001
# or
python scripts/view_employee_shap.py EMP_HD_001
```

### 4. Inspect Precursor Champion Versions:
```bash
# Champion 1 (Step 31 v2, 24 errors)
python CHAMPIONS/CHAMPION_1_STEP31/scripts/evaluate_champion1.py

# Champion 2 (Step 30 Severe-FN, 25 errors)
python CHAMPIONS/CHAMPION_2_STEP30/scripts/evaluate_champion2.py

# Champion 3 (Step 28 Stacking, 26 errors)
python CHAMPIONS/CHAMPION_3_STEP28/scripts/evaluate_champion3.py

# Champion 4 (Stage 6 Baseline, 30 errors)
python CHAMPIONS/CHAMPION_4_STAGE6/scripts/evaluate_champion4.py
```

### 5. Five-Version Comparative Matrix:
```bash
cd FINAL
python scripts/compare_versions.py
```

### 6. Cryptographic SHA-256 Immutability Check:
```bash
cd FINAL
python scripts/verify_integrity.py
```
