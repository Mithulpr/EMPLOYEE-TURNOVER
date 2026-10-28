# Comprehensive 15-Candidate CatBoost Model-Family Search & Benchmark Report

**Execution Date**: 2026-08-25 12:07:04  
**Evaluator**: Antigravity Machine Learning Intelligence (Track 1 / Version 12)  
**Dataset Reference**: `dataset/merged_unique_profiles_2969.csv` (SHA-256: `92027AF0EF8E882A19643479329252BC865B931346594893A520CFF2DBEA2C7C`)  
**Evaluation Protocol**: Leak-Free 5-Fold OOF, 25-Fold Repeated Stratified Nested CV, Single-Pass 594-Row Development Holdout (`FINAL_DEVELOPMENT_HOLDOUT_COMPARISON`).

---

## 1. Executive Summary & 15-Candidate Screening

A systematic 15-candidate CatBoost model family search was executed across tree depth, learning rate, iterations, L2 leaf regularization, ordered vs plain boosting, border counts, random strength, bagging temperatures, and feature representations.

### Table 1: Complete 15-Candidate 5-Fold OOF Benchmark Summary ($N_{\text{train}} = 2,375$)

| Candidate_Name | Pareto_OOF_Accuracy | Pareto_OOF_Precision | Pareto_OOF_Recall | Pareto_OOF_F1 | Pareto_OOF_ROC_AUC | Pareto_OOF_Errors | Composite_Selection_Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Exact V2 CatBoost Reproduction | 90.65% | 73.96% | 65.10% | 69.25% | 89.42% | 222.0 | 0.7795 |
| 2. Shallow Regularized CatBoost (Depth 4) | 87.62% | 60.71% | 66.41% | 63.43% | 86.82% | 294.0 | 0.7289 |
| 3. Depth-5 Regularized CatBoost | 89.18% | 67.79% | 63.02% | 65.32% | 88.30% | 257.0 | 0.7488 |
| 4. Depth-7 Stronger-L2 CatBoost | 88.97% | 65.89% | 65.89% | 65.89% | 89.13% | 262.0 | 0.7514 |
| 5. Low-LR / High-Iteration CatBoost | 87.83% | 61.02% | 68.49% | 64.54% | 87.60% | 289.0 | 0.7371 |
| 6. High-LR / Low-Iteration CatBoost | 88.88% | 65.71% | 65.36% | 65.54% | 88.45% | 264.0 | 0.7482 |
| 7. Ordered-Boosting CatBoost | 85.05% | 53.16% | 63.54% | 57.89% | 83.57% | 355.0 | 0.6843 |
| 8. Plain Boosting with Border Tuning | 90.53% | 71.31% | 69.27% | 70.28% | 88.66% | 225.0 | 0.7815 |
| 9. Strong L2 Regularization (L2=12.0) | 89.09% | 70.10% | 56.77% | 62.73% | 87.73% | 259.0 | 0.7365 |
| 10. Controlled Random Strength / Bagging | 88.55% | 63.93% | 66.93% | 65.39% | 87.88% | 272.0 | 0.7449 |
| 11. Mild Class Weighting (PosWeight=3.2) | 90.36% | 72.33% | 65.36% | 68.67% | 89.34% | 229.0 | 0.7743 |
| 12. Moderate Class Weighting (PosWeight=4.5) | 89.18% | 68.19% | 61.98% | 64.94% | 88.32% | 257.0 | 0.7471 |
| 13. Stagnation-Aware Feature CatBoost | 88.38% | 65.70% | 58.85% | 62.09% | 87.20% | 276.0 | 0.7266 |
| 14. Mobility/Career-Velocity Feature CatBoost | 89.05% | 71.99% | 52.86% | 60.96% | 87.71% | 260.0 | 0.7299 |
| 15. Final Hybrid Regularized CatBoost | 87.71% | 60.60% | 68.49% | 64.30% | 87.12% | 292.0 | 0.7346 |

---

## 2. Error Diversity & Transition Forensics vs V2 Reference

| Candidate_Name | Total_OOF_Errors | Rescued_V2_FPs | Rescued_V2_FNs | New_FPs_Created | New_FNs_Created | Net_Error_Change_vs_V2 | Jaccard_Error_Similarity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Exact V2 CatBoost Reproduction | 222 | 42 | 0 | 0 | 17 | -25 | 0.7765 |
| 2. Shallow Regularized CatBoost (Depth 4) | 294 | 27 | 11 | 62 | 23 | 47 | 0.6295 |
| 3. Depth-5 Regularized CatBoost | 257 | 43 | 7 | 28 | 32 | 10 | 0.6417 |
| 4. Depth-7 Stronger-L2 CatBoost | 262 | 33 | 8 | 34 | 22 | 15 | 0.6799 |
| 5. Low-LR / High-Iteration CatBoost | 289 | 21 | 13 | 59 | 17 | 42 | 0.6594 |
| 6. High-LR / Low-Iteration CatBoost | 264 | 37 | 10 | 38 | 26 | 17 | 0.6431 |
| 7. Ordered-Boosting CatBoost | 355 | 35 | 25 | 120 | 48 | 108 | 0.4506 |
| 8. Plain Boosting with Border Tuning | 225 | 40 | 14 | 17 | 15 | -22 | 0.6918 |
| 9. Strong L2 Regularization (L2=12.0) | 259 | 56 | 3 | 19 | 52 | 12 | 0.5912 |
| 10. Controlled Random Strength / Bagging | 272 | 24 | 10 | 39 | 20 | 25 | 0.6961 |
| 11. Mild Class Weighting (PosWeight=3.2) | 229 | 45 | 7 | 11 | 23 | -18 | 0.6940 |
| 12. Moderate Class Weighting (PosWeight=4.5) | 257 | 43 | 5 | 24 | 34 | 10 | 0.6525 |
| 13. Stagnation-Aware Feature CatBoost | 276 | 48 | 7 | 36 | 48 | 29 | 0.5801 |
| 14. Mobility/Career-Velocity Feature CatBoost | 260 | 64 | 6 | 13 | 70 | 13 | 0.5364 |
| 15. Final Hybrid Regularized CatBoost | 292 | 32 | 18 | 73 | 22 | 45 | 0.5760 |

---

## 3. Repeated Stratified 5x5 Nested CV (25 Folds) on Top Finalists

| Candidate_Name | Accuracy | Precision | Recall | F1_Score | ROC_AUC | Mean_Errors |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 8. Plain Boosting with Border Tuning | 89.65% ± 1.33% | 67.15% ± 3.70% | 70.42% ± 6.28% | 68.66% ± 4.52% | 91.33% ± 2.43% | 61.5 ± 7.9 |
| 1. Exact V2 CatBoost Reproduction | 90.42% ± 1.67% | 71.31% ± 4.90% | 67.96% ± 7.34% | 69.52% ± 5.96% | 91.70% ± 2.22% | 56.9 ± 9.9 |
| 11. Mild Class Weighting (PosWeight=3.2) | 90.23% ± 1.53% | 70.38% ± 4.85% | 68.42% ± 6.59% | 69.27% ± 5.22% | 91.58% ± 2.27% | 58.0 ± 9.1 |
| 4. Depth-7 Stronger-L2 CatBoost | 88.83% ± 1.57% | 64.29% ± 4.47% | 69.79% ± 6.75% | 66.82% ± 5.03% | 90.66% ± 2.33% | 66.3 ± 9.3 |

---

## 4. Single-Pass Development Holdout Snapshot ($N=594$)

- **Frozen Winner**: `8. Plain Boosting with Border Tuning` (@ $\tau^* = 0.4500$)
- **Total Correct**: **539 / 594** (90.74% Accuracy)
- **Total Errors**: **55 / 594**
- **Confusion Matrix**: TP=67, FP=26, TN=472, FN=29
- **Accuracy**: **90.74%** (V2: 92.42%)
- **Precision**: **72.04%** (V2: 78.02%)
- **Recall**: **69.79%** (V2: 73.96%)
- **F1-Score**: **70.90%** (V2: 75.94%)
- **ROC-AUC**: **90.17%** (V2: 90.69%)

---

## 5. Master Comparison Against Base-Paper Targets

| Metric | Literature Base-Paper Target | Official V2 Champion | Winning CatBoost Candidate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | > 92.00% | **92.42%** | **90.74%** | FAIL |
| **Precision** | > 89.00% | 78.02% | **72.04%** | FAIL |
| **Recall** | > 85.00% | 73.96% | **69.79%** | FAIL |
| **F1-Score** | > 87.00% | 75.94% | **70.90%** | FAIL |
| **ROC-AUC** | > 94.00% | 90.69% | **90.17%** | FAIL |

---
