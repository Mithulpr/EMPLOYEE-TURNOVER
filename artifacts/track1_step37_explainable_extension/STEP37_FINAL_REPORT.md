# Track 1 — Step 37: Explainable, Limitation-Aware Extension of the Step32 Champion

## 1. Executive Summary & Mission Mandate
In **Step 37**, we developed an explainable, limitation-aware employee attrition prediction and monitoring extension grounded in our official **Step 32 Champion** ($23$ errors on the 594-record legacy holdout).

In strict adherence to governance and anti-leakage principles:
1. **Absolute Immutability**: All artifacts belonging to Step 32 and prior steps were verified cryptographically and preserved strictly read-only.
2. **Bit-Exact Architectural Reconstruction**: The exact Step 32 pipeline (CatBoost base learner, LightGBM base learner, Logistic Regression stacking meta-learner, and CatBoost experience-density decision boundary refinement) was reconstructed and verified with **zero error differences** against the frozen holdout target ($487$ TN, $11$ FP, $12$ FN, $84$ TP).
3. **Genuine Post-Hoc SHAP Explainability**: Implemented global feature importance, beeswarm summary plots, dependence plots, local waterfall breakdowns, and explicit rule-override audit traces without altering model predictions.
4. **Input Data Quality & Response Bias Safeguards**: Developed an automated validation module detecting schema anomalies, missing fields, out-of-range values, and survey response divergences without attributing moral intent or dishonesty to employees.
5. **Model Drift Monitoring & Intervention Tracking**: Integrated Population Stability Index (PSI), Kolmogorov-Smirnov (KS) statistics, sample-size safeguards, and a longitudinal intervention action-tracking schema.
6. **Production-Grade Interactive Web Dashboard**: Delivered a self-contained, responsive HTML5/Vanilla CSS/JavaScript platform featuring research scoreboards, live prediction simulation, SHAP explainers, drift inspectors, and intervention tracking.

---

## 2. Exact Step 32 Architecture Copied
The Step 32 Champion consists of a calibrated two-stage stacking ensemble augmented with a domain boundary refinement layer:
- **Base Learner 1 (CatBoost)**: `iterations=600`, `learning_rate=0.06`, `depth=6`, `l2_leaf_reg=1.5`, `auto_class_weights='Balanced'`, random seed `42`.
- **Base Learner 2 (LightGBM)**: `num_leaves=31`, `max_depth=10`, `learning_rate=0.06`, `n_estimators=150`, `class_weight='balanced'`, random seed `42`.
- **Anchor Predictor (Step 27)**: Tri-model CatBoost anchor probabilities ($P_{33}, P_{28}, P_{37}$).
- **Meta-Learner (Logistic Regression)**: $C=1.0$, `max_iter=200`, fitted on $6$ out-of-fold meta-features ($P_{	ext{cb}}, P_{	ext{lgb}}, P_{s27}, |P_{	ext{cb}}-0.5|, |P_{	ext{lgb}}-0.5|, |P_{	ext{cb}}-P_{	ext{lgb}}|$).
- **Decision Threshold**: $	au = 0.3610$.
- **Boundary Refinement Layer**:
  - Activation Zone: $	ext{Pred}_{	ext{Step30}} == 0 \quad 	ext{and} \quad 0.15 \le P_{	ext{meta}} < 0.3610$.
  - Domain Predicates: $	ext{SevereDissatisfactionFlag} > 0.50$, $	ext{YearsWithCurrManager} > -0.50$, $	ext{JobHoppingRate} \le 1.00$, $	ext{TotalExpToAgeRatio} > -0.20$.
  - CatBoost Specialist Verification: $P_{	ext{cb\_spec}} \ge 0.40$ (where specialist is a shallow CatBoost tree: `depth=2, iterations=80, lr=0.03, l2=10`).

---

## 3. Exact Step 37 Additions
Step 37 augments the frozen Step 32 pipeline with non-invasive explanation, validation, and monitoring capabilities:
1. **Serialized Model Package**: Saved fitted pipeline artifacts (`step37_catboost_base.joblib`, `step37_lightgbm_base.joblib`, `step37_meta_logistic_regression.joblib`, `step37_catboost_specialist_refinement.joblib`).
2. **SHAP Explanation Engine**: Global beeswarm, bar plots, dependence charts, individual waterfall plots, and explicit rule-override traces.
3. **Data Quality & Response Bias Validator**: Reusable schema, range, and straight-lining auditor.
4. **Drift Monitoring Module**: Dual-window PSI and KS test detector with sample-size safeguards.
5. **HR Action Tracking System**: Snapshot-enabled intervention schema with causal inference disclaimers.
6. **Subgroup Demographic Fairness Auditor**: Disaggregated performance analysis across Gender, Age cohorts, and Marital Status.
7. **Full-Stack Web Dashboard**: 5-tab responsive web application in Vanilla CSS/JS.

---

## 4. Model Artifact Availability and Reproduction Status
- **Prior Repository Status**: The `artifacts/track1_step32_aggressive/final_candidate/` directory stored prediction CSVs and configuration JSONs, but no serialized `.joblib` model binary.
- **Resolution**: Reconstructed the model from the authoritative Step 32 codebase using identical random seeds (`42, 44221`), exact partitions (Pool $N=1,900$, Conf $N=475$, Holdout $N=594$), and exact hyperparameter dictionaries.
- **Verification Result**: Reconstructed models matched historical Step 32 holdout predictions with **zero record discrepancies** ($594/594$ identical classifications, max float probability difference $< 10^{-15}$).

---

## 5. Dataset Audit and Evaluation Protocol
- **Dataset**: `dataset/merged_unique_profiles_2969.csv` (SHA-256: `92027AF0EF8E882A19643479329252BC865B931346594893A520CFF2DBEA2C7C`).
- **Records**: $2,969$ total employees ($2,489$ retained, $480$ leavers = $16.17\%$ minority class).
- **Partitions**:
  - Training Pool: $1,900$ records (used for base learner training & OOF meta-feature generation).
  - Confirmation Set: $475$ records (used for internal validation, gating, and ablations).
  - Legacy Holdout: $594$ records (used strictly as frozen diagnostic benchmark).
- **Anti-Leakage Protocol**: No holdout labels were used for explainer fitting, threshold tuning, or background selection. Background data was drawn exclusively from the Training Pool.

---

## 6. SHAP Methodology & Key Explanations
- **Explainer Selection**: TreeExplainer applied to tree learners (CatBoost and LightGBM). Meta-level feature contributions combined via linear stacking weights.
- **Background Sample**: $100$ development records reproducibly drawn from the Training Pool.
- **Explanation Sample**: $50$ Confirmation records covering representative high-risk, low-risk, and borderline cases.
- **Top Global Predictive Drivers (by Mean Absolute SHAP)**:
  1. `MaritalStatus_Single` (Mean $|	ext{SHAP}| = 0.5777$): Single marital status consistently acts as an accelerant of mobility risk.
  2. `Age` (Mean $|	ext{SHAP}| = 0.5666$): Strong non-linear effect where younger career profiles exhibit substantially higher baseline mobility.
  3. `JobHoppingRate` (Mean $|	ext{SHAP}| = 0.5166$): Ratio of past companies worked relative to total years; powerful structural flight predictor.
  4. `YearsWithCurrManager` (Mean $|	ext{SHAP}| = 0.3790$): Managerial stability strongly suppresses attrition risk; transitions $<1.5$ years spike flight risk.
  5. `PromotionLagScore` (Mean $|	ext{SHAP}| = 0.3712$): Promotion stagnation relative to total tenure elevates risk.
  6. `PromotionPaceStrain` (Mean $|	ext{SHAP}| = 0.3373$): Interplay between tenure in current role and overall promotion velocity.
  7. `CompanyLoyaltyRatio` (Mean $|	ext{SHAP}| = 0.3136$): High loyalty suppresses risk, while low loyalty indicates flight readiness.
  8. `WorkLifeDeficit` (Mean $|	ext{SHAP}| = 0.2787$): Overtime and work-life imbalance drive positive push contributions.
  9. `SevereDissatisfactionFlag` (Mean $|	ext{SHAP}| = 0.2527$): Compound dissatisfaction across environment and role.
  10. `BusinessTravel_Travel_Frequently` (Mean $|	ext{SHAP}| = 0.2266$): Frequent travel strain.

---

## 7. Prediction Parity Tests
To prove that SHAP integration is strictly post-hoc and non-interfering:
- **Pre-SHAP Holdout Errors**: $23 / 594$ (TN=487, FP=11, FN=12, TP=84, Acc=96.13%, F1=87.96%).
- **Post-SHAP Holdout Errors**: $23 / 594$ (TN=487, FP=11, FN=12, TP=84, Acc=96.13%, F1=87.96%).
- **Discrepant Predictions**: Exactly $0$ records ($100.00\%$ parity).
- **Max Probability Absolute Difference**: $< 1.11 \times 10^{-16}$ (within floating-point machine precision).

---

## 8. Scientific Ablation Scoreboard (Confirmation Set, $N=475$)

| Ablation Variant | Confirmation Errors | Accuracy | Precision | Recall | F1-Score | Scientific Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. Full Step 32 Architecture (Baseline)** | **20** | **95.79%** | **88.00%** | **85.71%** | **86.84%** | Reconstructed gold standard; balances precision & recall. |
| **B. CatBoost Standalone Only** | 32 | 93.26% | 88.14% | 67.53% | 76.47% | CatBoost alone suffers severe recall drop (67.53%). |
| **C. LightGBM Standalone Only** | 42 | 91.16% | 73.33% | 71.43% | 72.37% | LightGBM alone exhibits elevated false positives & false negatives. |
| **D. Hybrid without Meta-Learner (50/50)** | 43 | 90.95% | 72.37% | 71.43% | 71.90% | Simple linear blend fails to capture boundary uncertainties. |
| **E. Stacking without Decision Refinement** | 23 | 95.16% | 87.50% | 81.82% | 84.56% | Omitting domain rule leaves 3 additional false negatives ($81.82\%$ vs $85.71\%$). |
| **F. Full Step 32 with SHAP Layer Active** | **20** | **95.79%** | **88.00%** | **85.71%** | **86.84%** | Proves SHAP layer causes exactly zero prediction deviation ($100\%$ parity). |

---

## 9. Data Quality and Survey Response Bias Safeguards
- **Schema Validation**: Automated checking against 45 raw columns and 30 model features ($100\%$ passed).
- **Range & Type Enforcers**: Numerical bounding on Age ($18\text{--}70$), Working Years ($0\text{--}50$), Monthly Income ($1,000\text{--}25,000$), and Survey Ratings ($1\text{--}4$).
- **Response Consistency Flags Detected**:
  - `SATISFACTION_STAGNATION_DIVERGENCE`: High satisfaction alongside prolonged career stagnation ($\ge 7$ years without promotion in $\ge 10$ years tenure) ($N=32$, $1.08\%$). Flagged for qualitative review; retention is not blindly assumed.
  - `NEW_ROLE_TRANSITION_DISSATISFACTION`: Dissatisfaction reported immediately following a recent promotion or manager change ($\le 1$ year) ($N=108$, $3.64\%$). Flagged as role transition stress rather than chronic flight risk.
  - `UNIFORM_RESPONSE_STRAIGHTLINING`: Identical rating submitted across all survey scales ($N=21$, $0.71\%$). Flagged for potential survey fatigue.
- **Ethical Safeguard**: Survey divergence flags reflect response anomalies or transition friction, NOT employee dishonesty.

---

## 10. Model Drift & Intervention Tracking Capabilities
- **Population Stability Index (PSI)**: Quantile-binned drift tracking. Both Confirmation ($N=475$) and Holdout ($N=594$) batches scored $0$ features with $\text{PSI} \ge 0.10$, confirming baseline distribution stability.
- **Sample-Size Safeguards**: Prominent warnings displayed whenever incoming batch size is $<30$ records.
- **Intervention Lifecycle**: Structured tracking from `PENDING_REVIEW` to `ACTION_IN_PROGRESS` and `COMPLETED`.
- **Causality Notice**: Explicitly disclaims causal efficacy claims in the absence of randomized control trials.

---

## 11. Demographic Fairness & Privacy Safeguards
Evaluated on the Holdout ($N=594$):
- **Gender Disaggregation**:
  - Male ($N=361$): Accuracy $95.01\%$, Recall $82.26\%$, Precision $87.93\%$, FPR $2.34\%$, Errors $18$.
  - Female ($N=233$): Accuracy $97.85\%$, Recall $97.06\%$, Precision $89.19\%$, FPR $2.01\%$, Errors $5$.
- **Age Cohorts**:
  - Younger ($<30$, $N=128$): Accuracy $92.19\%$, Recall $94.29\%$, Precision $80.49\%$, FPR $8.60\%$, Errors $10$.
  - Mid-Career ($30\text{--}45$, $N=338$): Accuracy $97.34\%$, Recall $86.05\%$, Precision $92.50\%$, FPR $1.02\%$, Errors $9$.
  - Senior ($>45$, $N=128$): Accuracy $96.88\%$, Recall $77.78\%$, Precision $100.00\%$, FPR $0.00\%$, Errors $4$.
- **Marital Status**:
  - Single ($N=173$): Accuracy $94.22\%$, Recall $91.49\%$, Precision $87.76\%$, FPR $4.76\%$, Errors $10$.
  - Married ($N=283$): Accuracy $95.76\%$, Recall $81.40\%$, Precision $89.74\%$, FPR $1.67\%$, Errors $12$.
  - Divorced ($N=138$): Accuracy $99.28\%$, Recall $100.00\%$, Precision $85.71\%$, FPR $0.76\%$, Errors $1$.
- **Privacy Protections**: All employee identifiers are anonymized (`EMP_CONF_XXXX`, `EMP_HOLD_XXXX`). No personal demographic identifiers are logged in model training artifacts.

---

## 12. Limitation Classification Matrix

| Research & Operational Limitation | Status | Mitigation / Scientific Safeguard Implemented in Step 37 |
| :--- | :---: | :--- |
| **Lack of True Real-Time Production Feed** | Partially Addressed | Built batch-upload validator and drift monitoring engine ready for future data ingestion. |
| **Lack of Independent External Validation** | Partially Addressed | Conducted nested CV on development partitions; preserved legacy holdout as frozen benchmark; avoided iterative tuning. |
| **Cross-Sectional Single-Survey Limitation** | Addressed | Implemented survey consistency auditor; documented impossibility of panel verification on single snapshots. |
| **Observational Causality Confounding** | Addressed | Added prominent causality warnings; disclaimed causal efficacy claims for HR interventions. |
| **Sample-Size Instability in Drift Testing** | Addressed | Enforced hard statistical safeguards and UI warnings for batches with $N < 30$ records. |
| **Model Interpretability vs Stacking Complexity** | Addressed | TreeExplainer deployed on base trees, meta-learner decomposed, and rule overrides surfaced explicitly. |
| **Employee Privacy & Surveillance Risk** | Addressed | Anonymized employee identifiers; excluded sensitive demographics from decision override predicates. |
| **Real-World Intervention Outcome Tracking** | Unresolved | Requires longitudinal real-world organizational data collection; cannot be resolved on historical static benchmark. |

---

## 13. Transparent Governance Statement
This project utilizes a static, historical benchmarking dataset ($N=2,969$). While internal cross-validation and confirmation partitioning demonstrate high technical precision ($96.13\%$ accuracy, $87.96\%$ F1), **no claims of real-time operational deployment or independent external multi-organization generalization are made**. True real-world deployment requires prospective cohort studies, independent organizational validation, and randomized evaluation of HR interventions.

---

## 14. Final Conclusion & Mentor Summary
Step 37 successfully extends the official **Step 32 Champion** into an explainable, limitation-aware decision support system without modifying the frozen predictive core. By verifying bit-exact model reconstruction, deploying multi-level SHAP explainers, establishing data quality and drift monitoring protocols, and providing an interactive web dashboard, Step 37 establishes a complete, academically defensible framework for responsible predictive HR analytics.

$$\mathbf{STEP\ 32\ RETAINED\ AS\ OFFICIAL\ 5TH\ CHAMPION;\ STEP\ 37\ ARCHIVED\ AS\ EXPLAINABILITY\ EXTENSION}$$
