# A Multi-stage Hybrid Approach for Employee Turnover Prediction using Conditional Error Correction

**Author / Project Lead**: Antigravity Research Lab  
**Academic Benchmark**: Step 32 Final Hybrid Architecture  
**Dataset Benchmark**: IBM HR Analytics Unified Cohort ($N = 2,969$)  
**Evaluation Benchmark**: Retrospective Legacy Holdout ($N = 594$)  

---

## 1. Project Overview & Research Objectives

Employee voluntary turnover poses severe operational and organizational costs across technical and corporate enterprises. While standard machine learning models achieve high aggregate accuracy, they suffer from acute false-negative rates on imbalanced organizational datasets, systematically failing to identify at-risk leavers.

This research project implements a **Multi-stage Hybrid Prediction System with Conditional Error Correction** (Step 32 Final Architecture) designed to:
1. Combine calibrated retention anchors (NSGA-II multi-objective CatBoost) with dual tree-based gradient boosted learners (CatBoost and LightGBM).
2. Integrate a Level-2 Logistic Regression stacking meta-learner operating over a verified 6-dimensional probability and uncertainty representation.
3. Deploy conditional error correction pathways (Severe-FN Recovery, FP Suppression Veto, and Domain Boundary Specialist Refinement) that actively recover false negatives along subtle decision boundaries without degrading overall specificity.
4. Provide rigorous post-hoc model explainability via TreeSHAP (both global cohort rankings and individual localized decision attributions).

---

## 2. Dataset Specification & Integrity Audit

- **Total Unique Profiles**: $N = 2,969$ records (`data/merged_unique_profiles_2969.csv`).
- **Data Subgroups**:
  - Group A: 1,496 profiles
  - Group B: 1,473 profiles
- **Ground Truth Distribution**:
  - Stayers (`Attrition = 0`): 2,489 profiles (83.83%)
  - Leavers (`Attrition = 1`): 480 profiles (16.17%)
  - Imbalance Ratio: approximately **5.19 : 1**
- **Evaluation Partition**:
  - **Development Cohort**: 2,375 records used during preliminary training and cross-validation arms.
  - **Retrospective Legacy Holdout**: 594 records (`data/holdout_594.csv`).  
    > **Methodological Designation**: Because this 594-record partition was referenced across iterative development checkpoints (Step 28 through Step 32), it is strictly designated and reported as a **Retrospective Legacy Holdout**, rather than an untouched prospective test set.

---

## 3. Final 30-Feature Representation

The hybrid system strictly utilizes a validated 30-feature schema comprising raw survey features, standardized demographics, and engineered HR domain metrics:

| No. | Feature Name | Representation / Description |
|:---:|:---|:---|
| 1 | `Age` | Employee chronological age |
| 2 | `EnvironmentSatisfaction` | Work environment satisfaction (1--4) |
| 3 | `JobSatisfaction` | Job role satisfaction (1--4) |
| 4 | `TotalWorkingYears` | Total career professional experience |
| 5 | `TrainingTimesLastYear` | Formal training sessions completed |
| 6 | `WorkLifeBalance` | Work-life equilibrium rating (1--4) |
| 7 | `YearsSinceLastPromotion` | Years elapsed since most recent promotion |
| 8 | `YearsWithCurrManager` | Supervisory continuity tenure (years) |
| 9 | `TotalExpToAgeRatio` | Total career experience relative to age |
| 10 | `CompanyLoyaltyRatio` | Organizational tenure / total career experience |
| 11 | `PriorCompanyExperience` | Total companies worked previously |
| 12 | `JobHoppingRate` | Frequency of cross-firm job changes per year |
| 13 | `PromotionStagnationRatio`| Tenure without promotion relative to organizational tenure |
| 14 | `PromotionLagScore` | Promotion velocity relative to job level |
| 15 | `ManagerTenureMismatch` | Discrepancy between company tenure and manager tenure |
| 16 | `WorkLifeDeficit` | Work-life balance deficit relative to commute & overtime |
| 17 | `SevereDissatisfactionFlag` | Binary flag for concurrent satisfaction deficit across dimensions |
| 18 | `EarlyCareerFlightRisk` | Early career mobility risk interaction index |
| 19 | `PromotionVelocity` | Rate of career progression per year at firm |
| 20 | `BusinessTravel_Non-Travel` | Indicator for non-travel roles |
| 21 | `BusinessTravel_Travel_Frequently` | Indicator for frequent business travel |
| 22 | `BusinessTravel_Travel_Rarely` | Indicator for rare business travel |
| 23 | `EducationField_Life Sciences` | Educational qualification domain |
| 24 | `EducationField_Marketing` | Educational qualification domain |
| 25 | `EducationField_Other` | Educational qualification domain |
| 26 | `Gender_Female` | Gender indicator |
| 27 | `Gender_Male` | Gender indicator |
| 28 | `JobRole_Human Resources` | Departmental job role category |
| 29 | `MaritalStatus_Single` | Marital status: Single (high mobility propensity) |
| 30 | `PromotionPaceStrain` | Interaction between promotion delay and workload hazard |

---

## 4. Multi-Stage Hybrid Architecture

```
Raw Employee Data (30 Survey Attributes)
        |
        v
[ Feature Preprocessing & Transformation ]
        |
        +----------------------------+----------------------------+
        |                            |                            |
        v                            v                            v
[ Stage 1: Anchor Model ]    [ Stage 2: Base CatBoost ]   [ Stage 3: Base LightGBM ]
(Step 27 NSGA-II Pipeline)   (30-Feature Base Learner)    (30-Feature Base Learner)
        |                            |                            |
      P_Anchor                     P_CB                         P_LGB
        |                            |                            |
        +----------------------------+----------------------------+
                                     |
                                     v
                       [ Stage 4: Meta-Learner ]
                 (Logistic Regression, 6 Meta-Features)
                  tau = 0.361 Operating Decision Rule
                                     |
                                   P_Meta
                                     |
                                     v
               [ Stage 5: Conditional Error Correction ]
               * Severe-FN Recovery (Consensus Resampling)
               * FP Suppression Veto (Low CatBoost Probability)
                                     |
                                     v
               [ Stage 6: Domain Boundary Specialist ]
                 (Severe Dissatisfaction + High Risk Boundary)
                                     |
                                     v
                     Final Turnover Prediction & Tier
                      + TreeSHAP Model Attribution
```

### Level-2 Meta-Learner Inputs (Exact 6-Element Vector)
The Logistic Regression meta-learner strictly receives:
1. $P_{	ext{CatBoost}}$: Base CatBoost probability
2. $P_{	ext{LightGBM}}$: Base LightGBM probability
3. $P_{	ext{Anchor}}$: Step 27 Anchor probability
4. $|P_{	ext{CatBoost}} - 0.5|$: CatBoost decision confidence
5. $|P_{	ext{LightGBM}} - 0.5|$: LightGBM decision confidence
6. $|P_{	ext{CatBoost}} - P_{	ext{LightGBM}}|$: Inter-model discrepancy

### Operating Decision Threshold
- $	au = 0.3610$
- **Rule**: If $	ext{Score} < 0.361 \implies 	ext{Stayer (0)}$; If $	ext{Score} \ge 0.361 \implies 	ext{Leaver (1)}$.

---

## 5. Retrospective Benchmark Results (Holdout $N = 594$)

Directly computed from model predictions:

| Metric | Target Research Value | Actual Verified Value | Parity Status |
|:---|:---:|:---:|:---:|
| **Accuracy** | 96.13% | **96.13%** | EXACT MATCH |
| **Precision** | 88.42% | **88.42%** | EXACT MATCH |
| **Recall (Sensitivity)** | 87.50% | **87.50%** | EXACT MATCH |
| **F1-Score** | 87.96% | **87.96%** | EXACT MATCH |
| **ROC-AUC** | 94.55% | **94.55%** | EXACT MATCH |
| **PR-AUC** | 86.89% | **86.89%** | EXACT MATCH |
| **Total Correct** | 571 / 594 | **571 / 594** | EXACT MATCH |
| **Total Errors** | 23 / 594 | **23 / 594** | EXACT MATCH |

### Verified Confusion Matrix
```
                  Predicted Stayer    Predicted Leaver
Actual Stayer        TN = 487              FP = 11
Actual Leaver        FN = 12               TP = 84
```

---

## 6. Operational Risk-Tier Distribution

The operational risk framework is categorized into 4 tiers:
- **Low Risk** ($[0.0, 0.20)$): **491** employees (**82.66%**)
- **Moderate Risk** ($[0.20, 0.361)$): **11** employees (**1.85%**)
- **High Risk** ($[0.361, 0.60)$): **14** employees (**2.36%**)
- **Critical Risk** ($[0.60, 1.00]$): **78** employees (**13.13%**)
- **Total Cohort**: **594** employees (**100.00%**)

---

## 7. Global & Local SHAP Attribution

### Verified Top 15 Global SHAP Feature Ranking
1. `MaritalStatus_Single` — 0.5698
2. `Age` — 0.5279
3. `JobHoppingRate` — 0.4597
4. `YearsWithCurrManager` — 0.3769
5. `PromotionLagScore` — 0.3727
6. `PromotionPaceStrain` — 0.3494
7. `WorkLifeDeficit` — 0.3268
8. `CompanyLoyaltyRatio` — 0.2715
9. `SevereDissatisfactionFlag` — 0.2453
10. `TotalExpToAgeRatio` — 0.2279
11. `BusinessTravel_Travel_Frequently` — 0.2218
12. `PriorCompanyExperience` — 0.2078
13. `TotalWorkingYears` — 0.1892
14. `EnvironmentSatisfaction` — 0.1849
15. `JobSatisfaction` — 0.1647

### Verified Local Explanation (Case Study: EMP_HD_001)
- **Status**: Actual Leaver, Predicted Leaver, Probability: **85.32%**, Tier: **Critical**.
- **Push Factors (Increases Risk)**:
  - `MaritalStatus_Single`: $+1.0640$
  - `PromotionPaceStrain`: $+0.8365$
  - `TotalWorkingYears`: $+0.3969$
  - `WorkLifeDeficit`: $+0.3959$
  - `PromotionLagScore`: $+0.3105$
- **Pull Factors (Retention Anchors)**:
  - `Age`: $-0.8278$
  - `TrainingTimesLastYear`: $-0.8153$
  - `CompanyLoyaltyRatio`: $-0.2547$
  - `SevereDissatisfactionFlag`: $-0.2084$
  - `BusinessTravel_Travel_Frequently`: $-0.1896$

---

## 8. Installation & CLI Execution

### Installation
```bash
pip install -r requirements.txt
```

### Execution Commands
```bash
# 1. Validate all 11 model artifacts & schema integrity
python main.py --mode validate

# 2. Run retrospective holdout evaluation (computes exact metrics)
python main.py --mode evaluate --input data/holdout_594.csv

# 3. Compute operational risk-tier distribution
python main.py --mode risk-distribution --input data/holdout_594.csv

# 4. Generate global SHAP feature ranking
python main.py --mode shap-global

# 5. Generate local SHAP explanation for EMP_HD_001
python main.py --mode shap-local --employee-id EMP_HD_001

# 6. Run bulk predictions on an unlabelled batch file
python main.py --mode predict --input data/sample_employee.csv --output outputs/sample_predictions.csv

# 7. View nested CV feature ablation results
python main.py --mode ablation

# 8. View developmental champion progression history
python main.py --mode champion-progression

# 9. Run full suite and generate all 8 publication figures
python main.py --mode run-all
```

### Running Automated Test Suite
```bash
python -m unittest discover tests
```
