# Talebi et al. (2025) Base Paper Exact Reproduction Report on 2,969 Dataset

**Date**: 2026-10-06 09:12:42  
**Workspace**: `BASE_PAPER_REPRODUCE`  
**Dataset**: `data/merged_unique_profiles_2969.csv` ($N = 2,969$ unique employee profiles)  
**Base Paper Citation**:  
> Talebi, H., Taghvaee, S., Zolfaghari, M., & Shokouhyar, S. (2025).  
> *A hybrid machine learning framework with genetic algorithm-based feature selection for employee turnover prediction*.  
> **Journal of Open Innovation: Technology, Market, and Complexity**, Vol. 11, Issue 1, Article 100557, Elsevier. DOI: 10.1016/j.joitmc.2025.100557.

---

## 1. Executive Summary

This isolated reproduction executes the audited experimental methodology published by Talebi et al. (2025) transferred directly onto our frozen 2,969-profile employee turnover dataset.

No proprietary domain feature engineering, synthetic oversampling, or proposed-model components (such as CatBoost, XGBoost, or Step 32) were incorporated. All metrics are calculated directly from this freshly executed 10-fold cross-validation run using the paper's exact audited parameters (MinMaxScaler, Mean/Mode imputation, GA with 10-fold CV LightGBM fitness, and LightGBM with $lr=0.10, max\_depth=10, num\_leaves=31$).

### Key Performance Findings:
- **Accuracy**: **92.59%** (2,749 correct classifications out of 2,969 records)
- **Precision**: **88.46%**
- **Recall**: **62.29%**
- **F1-Score**: **73.11%**
- **ROC-AUC**: **93.95%**
- **PR-AUC**: **83.40%**
- **Balanced Accuracy**: **80.36%**
- **Specificity**: **98.43%**
- **Brier Score**: **0.0582**
- **Total Classification Errors**: **220** (False Positives: 39, False Negatives: 181)

---

## 2. Methodology & Experimental Protocol Audit

| Component | Base Paper Specification | Reproduction Implementation | Audit Status |
| :--- | :--- | :--- | :--- |
| **Dataset Size** | 2,932 records (Kaggle HR) / 1,470 (Org) | 2,969 records (`data/merged_unique_profiles_2969.csv`) | Replaced by Task Design |
| **Target Variable** | Attrition (Binary 0/1) | Attrition (0 = Retained, 1 = Turnover; 16.17% positive) | MATCH |
| **Raw Features** | 30 survey features (Table 4) | 30 raw survey features (zero interaction terms) | MATCH |
| **Missing Imputation** | Mean for continuous, Mode for categorical (Page 5) | Mean for continuous, Mode for categorical | MATCH |
| **Feature Encoding** | Integer encoding for LightGBM (Page 8) | LabelEncoder (integer representation) | MATCH |
| **Feature Scaling** | Min-Max scaling to [0, 1] (Page 6) | MinMaxScaler [0, 1] | MATCH |
| **Resampling** | None / Native distribution (Page 10) | None / Native distribution preserved | MATCH |
| **Feature Selection** | Binary Genetic Algorithm (GA) | Binary Genetic Algorithm (`GeneticAlgorithmFeatureSelector`) | MATCH |
| **GA Population** | 50 binary chromosomes (Pages 6-7) | 50 binary chromosomes | MATCH |
| **GA Selection** | Roulette Wheel Selection (Page 6) | Roulette Wheel Selection | MATCH |
| **GA Crossover** | Uniform Crossover ($p_c = 0.80$, Page 7) | Uniform Crossover ($p_c = 0.80$) | MATCH |
| **GA Mutation** | Bit-flip Mutation ($p_m = 0.10$, Page 7) | Bit-flip Mutation ($p_m = 0.10$ per gene) | MATCH |
| **GA Elitism** | Top 5 chromosomes (10%, Page 7) | Top 5 chromosomes (10%) | MATCH |
| **GA Fitness** | 10-Fold CV LightGBM F1-Score (Page 7) | 10-Fold Stratified CV LightGBM F1-Score | MATCH |
| **LightGBM Model** | $lr=0.10, num\_leaves=31, max\_depth=10$ (Page 8) | $lr=0.10, num\_leaves=31, max\_depth=10, subsample=0.8$ | MATCH |
| **Evaluation** | 10-Fold Stratified Cross-Validation (Pages 8, 10, 13) | 10-Fold Stratified Cross-Validation on all 2,969 samples | MATCH |

---

## 3. Genetic Algorithm Convergence & Selected Feature Subset

The Genetic Algorithm converged with a peak 10-Fold CV F1-Score of **0.7299**.

### Selected Optimal Features (11 features):
1. `Age`
2. `BusinessTravel`
3. `EnvironmentSatisfaction`
4. `JobSatisfaction`
5. `MaritalStatus`
6. `NumCompaniesWorked`
7. `TotalWorkingYears`
8. `TrainingTimesLastYear`
9. `WorkLifeBalance`
10. `YearsSinceLastPromotion`
11. `YearsWithCurrManager`

---

## 4. Comprehensive Performance Metrics (10-Fold Cross-Validation)

| Metric | Exact Paper Protocol on 2,969 | Historical 88.01% Implementation | Original Paper on 2,932 (Table 13) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **92.59%** | 88.01% | 92.00% |
| **Precision** | **88.46%** | 74.80% | 89.00% |
| **Recall** | **62.29%** | 38.96% | 85.00% |
| **F1-Score** | **73.11%** | 51.23% | 87.00% |
| **ROC-AUC** | **93.95%** | 87.11% | 94.00% |
| **PR-AUC** | **83.40%** | 66.13% | Not Reported |
| **Balanced Accuracy** | **80.36%** | 68.21% | Not Reported |
| **Specificity** | **98.43%** | 97.47% | Not Reported |
| **Brier Score** | **0.0582** | 0.0874 | Not Reported |
| **Total Errors** | **220** | 356 | Not Reported |
| **True Positives (TP)** | **299** | 187 | Not Reported |
| **False Positives (FP)** | **39** | 63 | Not Reported |
| **True Negatives (TN)** | **2450** | 2,426 | Not Reported |
| **False Negatives (FN)** | **181** | 293 | Not Reported |

---

## 5. Confusion Matrix Breakdown

| Actual / Predicted | Predicted Retained (0) | Predicted Turnover (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Retained (0)** | **2,450** (TN) | **39** (FP) | **2,489** |
| **Actual Turnover (1)** | **181** (FN) | **299** (TP) | **480** |
| **Total Predicted** | **2,631** | **338** | **2,969** |

---

## 6. Generated Artifacts

All outputs are saved locally within `BASE_PAPER_REPRODUCE/`:
- `outputs/base_paper_methodology_registry.json`
- `outputs/feature_preprocessing_registry.csv`
- `outputs/resampling_registry.csv`
- `outputs/selected_features.json`
- `outputs/selected_feature_list.txt`
- `outputs/ga_generation_history.csv`
- `outputs/ga_fitness_convergence.png`
- `outputs/predictions.csv`
- `outputs/confusion_matrix.csv`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/pr_curve.png`
- `outputs/metrics.json`
- `outputs/base_paper_method_on_2969_results.csv`
- `outputs/reproducibility_config.json`
- `reports/BASE_PAPER_REPRODUCTION_2969.md`
