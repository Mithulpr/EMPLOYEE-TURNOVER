# Track 1 — Step 36: Aggressive, Controlled Fine-Tuning of the Step35 XGBoost Replacement Framework

## Executive Summary & Mission Mandate
In **Step 36**, we conducted a systematic, 60-round multi-stage optimization campaign designed to aggressively refine the CatBoost + XGBoost hybrid architecture established in Step 35, investigating whether it could match or surpass the official **Step 32 Champion** ($23$ errors on the 594-record legacy holdout) without reintroducing LightGBM.

All operations adhered strictly to absolute immutability and anti-leakage governance:
1. **Zero Data Snooping**: Optimization was conducted strictly inside fold-isolated cross-validation on the development data ($N=2,375$: Training Pool $N=1,900$, Confirmation $N=475$). The 594-record legacy holdout was held completely closed until after the candidate was frozen.
2. **Absolute Immutability**: All artifacts belonging to Step 32, Step 35, Step 30, Step 31, Step 33, and Step 34 were verified and preserved as read-only.
3. **No LightGBM**: LightGBM was completely absent from all Step 36 training, inference, and prediction pipelines.

---

## 1. Master Comparative Scoreboard

| Benchmark / Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Errors ($N=594$) | Confusion Matrix (TN, FP, FN, TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Paper Target (Talebi et al., 2025)** | 92.00% | 89.00% | 85.00% | 87.00% | 94.00% | — | — |
| **Step 31 Candidate v2 (4th Champion)** | 95.96% | 87.50% | 87.50% | 87.50% | 94.55% | 24 | TN=486, FP=12, FN=12, TP=84 |
| **Step 32 Champion (5th Champion — Baseline)** | **96.13%** | **88.42%** | **87.50%** | **87.96%** | **94.55%** | **23** | **TN=487, FP=11, FN=12, TP=84** |
| **Step 35 XGBoost Replacement Baseline** | 95.62% | 87.23% | 85.42% | 86.32% | 94.25% | 26 | TN=486, FP=12, FN=14, TP=82 |
| **Step 36 Fine-Tuned Candidate** | **95.79%** | **88.17%** | **85.42%** | **86.77%** | **94.23%** | **25** | **TN=487, FP=11, FN=14, TP=82** |

---

## 2. Multi-Stage Optimization Campaign (60 Rounds Completed)

The 60 rounds were organized across six distinct thematic phases:

### Phase Summary:
1. **Phase A (Rounds 1–5: Baseline Reproduction & Diagnostics)**: Verified Step 35 baseline parity, standalone CatBoost/XGBoost baselines, and uncalibrated equal blends. Confirmed that standalone models suffer from acute recall or precision deficiencies without stacking.
2. **Phase B (Rounds 6–15: CatBoost Optimization & Regularization)**: Explored tree depth ($d=4$ to $7$), iterations ($500$ to $800$), L2 regularization ($1.5$ to $8.0$), random strength, feature subsampling (`rsm=0.80`), and asymmetric class weights. Discovered that $d=6$ with moderate regularization (lambda 1.5 to 3.0) provides the optimal trade-off between margin separation and leaf stability.
3. **Phase C (Rounds 16–30: XGBoost Replacement Optimization)**: Investigated shallow vs deep XGBoost architectures, L1/L2 penalties (`reg_alpha=1.0`, `reg_lambda=12`), split penalties (gamma 0.5 to 2.0), `max_delta_step` for imbalanced updates, and stochastic subsampling. Confirmed that a deep, conservative architecture ($d=5$, $n=200$, lr=0.03, sub=0.70, col=0.85, lambda=10, gamma=1.0) maximizes consensus synergy with CatBoost.
4. **Phase D (Rounds 31–40: Meta-Learning, Calibration & Stacking)**: Evaluated probability blending, logit-space stacking, Ridge-regularized logistic stacking ($C$ 0.20 to 2.0), interaction meta-features, confidence-margin weighting, and isotonic/Platt calibration. Validated that L2-regularized logistic stacking on linear probabilities maintains the most stable out-of-fold generalization.
5. **Phase E (Rounds 41–50: Decision Boundary, Disagreement & Domain Refinement)**: Swept decision thresholds (tau in [0.34, 0.37]), cost-sensitive objectives, disagreement veto gates, consensus thresholds (min prob >= 0.10), and experience density ratios ($r_{th} > -0.20$). The default threshold tau=0.3610 with consensus minimum 0.10 and $r_{th} > -0.20$ demonstrated the most consistent development F1 score ($88.16\%$) and lowest confirmation errors ($18$).
6. **Phase F (Rounds 51–60: Ablations, Robustness & Final Synthesis)**: Executed component ablations, multi-seed stability across 4 and 8 seeds, and 1,000-iteration non-parametric bootstrap confidence intervals before freezing the candidate.

---

## 3. Scientific Ablation Scoreboard (Confirmation Set, $N=475$)

| Ablation Variant | Confirmation Errors | Accuracy | Precision | Recall | F1-Score | Scientific Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Step 32 Baseline (with LightGBM)** | 20 | 95.79% | 88.00% | 85.71% | 86.84% | Reference gold standard using LightGBM. |
| **CatBoost Standalone (no XGBoost)** | 32 | 93.26% | 88.14% | 67.53% | 76.47% | Severe recall collapse ($67.53\%$) without secondary booster. |
| **XGBoost Standalone (no CatBoost)** | 31 | 93.47% | 82.81% | 68.83% | 75.18% | Standalone XGBoost exhibits high boundary variance. |
| **Equal Probability Blend (50/50, No Meta)** | 35 | 92.63% | 80.88% | 71.43% | 75.86% | Unweighted linear blend fails to learn optimal confidence weighting. |
| **Pure Stacking (No Severe-FN Recovery)** | 19 | 96.00% | 88.16% | 87.01% | 87.58% | Omitting FN recovery layer leaves 1 additional false negative. |
| **Full Step 36 Fine-Tuned System (Winner)** | **17** | **96.42%** | **90.54%** | **87.01%** | **88.74%** | Complete hybrid system achieves peak precision (90.54%) and F1 (88.74%). |

---

## 4. Multi-Seed Stability & Bootstrap Robustness

### Multi-Seed Stability (8 Seeds: 42, 123, 2024, 3407, 34071, 34072, 34073, 34074)
- **Confirmation Errors**: 17.00 +/- 0.00
- **Confirmation Accuracy**: 96.42%
- **Confirmation Precision**: 90.54%
- **Confirmation Recall**: 87.01%
- **Confirmation F1-Score**: 88.74%

### 1,000-Iteration Non-Parametric Bootstrap 95% Confidence Intervals
- **Accuracy 95% CI**: [94.74%, 97.89%]
- **Precision 95% CI**: [82.69%, 96.78%]
- **Recall 95% CI**: [79.10%, 93.83%]
- **F1-Score 95% CI**: [82.92%, 93.43%]
- **Total Errors 95% CI**: [10.0, 25.0]

---

## 5. Post-Freeze Holdout Evaluation & Error Diagnostics

Following candidate freezing, the single permitted diagnostic benchmark on the 594-record legacy holdout yielded:
- **Accuracy**: 95.79%
- **Precision**: 88.17%
- **Recall**: 85.42%
- **F1-Score**: 86.77%
- **ROC-AUC**: 94.23%
- **Confusion Matrix**: TN=487, FP=11, FN=14, TP=82 (25 total errors)

### Comparative Error Analysis:
1. **Development Superiority**: On the Confirmation partition ($N=475$), the fine-tuned XGBoost architecture outperformed Step 32 by 3 fewer errors (17 vs 20), +2.54% precision (90.54% vs $88.00\%$), and +1.90% F1 score (88.74% vs $86.84\%$).
2. **Holdout Error Plateau**: On the legacy holdout ($N=594$), replacing LightGBM with XGBoost produced 25 errors (+2 errors relative to Step 32's 23 errors). Specifically, two borderline leavers that LightGBM scored marginally above the consensus recovery threshold received lower probability scores under XGBoost, preventing recovery without manually overfitting to holdout record IDs.
3. **Generalization Integrity**: No post-freeze tuning or error manipulation was performed.

---

## 6. Official Promotion Verdict

**STEP 32 RETAINED AS OFFICIAL CHAMPION (STEP 36 HOLDOUT ERRORS: 25 VS STEP 32: 23)**

### Governance & Promotion Rationale:
- **Predefined Standard**: Step 32 remains the official Champion unless the candidate strictly beats or matches all targets (23 or fewer errors, Accuracy >= 96.13%, Precision >= 88.42%, Recall >= 87.50%, F1 >= 87.96%) on actual post-freeze evaluation.
- **Outcome**: Step 36 confirmed that the CatBoost + XGBoost hybrid architecture achieves superior development generalization (17 errors, 90.54% precision, zero multi-seed variance across 8 seeds), but scored 25 errors on the legacy holdout.
- **Verdict**: In accordance with scientific integrity rules, **Step 32 remains the official 5th Champion**. Step 36 is preserved as a complete, fully documented, 60-round research milestone demonstrating that XGBoost provides a stable, zero-variance alternative to LightGBM in two-stage imbalanced meta-learning pipelines.
