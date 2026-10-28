# Talebi et al. (2025) Base Paper Reproduction Report on 2,969 Dataset

**Date**: 2026-10-05 23:20:13  
**Workspace**: `BASE_PAPER_REPRODUCE`  
**Dataset**: `data/merged_unique_profiles_2969.csv` ($N = 2,969$ unique employee profiles)  
**Base Paper Citation**:  
> Talebi, H., Taghvaee, S., Zolfaghari, M., & Shokouhyar, S. (2025).  
> *A hybrid machine learning framework with genetic algorithm-based feature selection for employee turnover prediction*.  
> **Journal of Open Innovation: Technology, Market, and Complexity**, Vol. 11, Issue 1, Article 100557, Elsevier. DOI: 10.1016/j.joitmc.2025.100557.

---

## 1. Executive Summary

This isolated reproduction executes the exact methodology published by Talebi et al. (2025) transferred directly onto our frozen 2,969-profile employee turnover dataset.

No proprietary domain feature engineering, synthetic oversampling, or proposed-model components (such as CatBoost, XGBoost, or Step 32) were incorporated. All metrics are calculated directly from this freshly executed 10-fold cross-validation run.

### Key Performance Findings:
- **Accuracy**: **88.01%** (2,613 correct classifications out of 2,969 records)
- **Precision**: **74.80%**
- **Recall**: **38.96%**
- **F1-Score**: **51.23%**
- **ROC-AUC**: **87.11%**
- **PR-AUC**: **66.13%**
- **Balanced Accuracy**: **68.21%**
- **Specificity**: **97.47%**
- **Brier Score**: **0.0874**
- **Total Classification Errors**: **356** (False Positives: 63, False Negatives: 293)

---

## 2. Methodology & Experimental Protocol

| Component | Base Paper Specification | Reproduction Implementation |
| :--- | :--- | :--- |
| **Dataset Size** | 2,932 records (Kaggle HR) / 1,470 (Org) | 2,969 records (`data/merged_unique_profiles_2969.csv`) |
| **Target Variable** | Attrition (Binary 0/1) | Attrition (0 = Retained, 1 = Turnover; 16.17% positive) |
| **Raw Features** | 30 survey features | 30 raw survey features (zero interaction terms) |
| **Preprocessing** | Categorical: LabelEncoder; Continuous: StandardScaler | LabelEncoder for categorical, StandardScaler for numeric |
| **Resampling** | None (Native class distribution) | None (Native class distribution preserved) |
| **Feature Selection** | Binary Genetic Algorithm (GA) | Binary Genetic Algorithm (`GeneticAlgorithmFeatureSelector`) |
| **GA Population** | 50 binary chromosomes | 50 binary chromosomes |
| **GA Selection** | Roulette Wheel Selection | Roulette Wheel Selection |
| **GA Crossover** | Uniform Crossover ($p_c = 0.80$) | Uniform Crossover ($p_c = 0.80$) |
| **GA Mutation** | Bit-flip Mutation ($p_m = 0.10$ per gene) | Bit-flip Mutation ($p_m = 0.10$ per gene) |
| **GA Elitism** | Top 5 chromosomes (10%) | Top 5 chromosomes (10%) |
| **Fitness Function** | LightGBM CV F1-Score | LightGBM Stratified CV F1-Score |
| **LightGBM Model** | $lr=0.08, num\_leaves=31, max\_depth=6$ | $lr=0.08, num\_leaves=31, max\_depth=6, subsample=0.8$ |
| **Evaluation** | 10-Fold Stratified Cross-Validation | 10-Fold Stratified Cross-Validation on all 2,969 samples |

---

## 3. Genetic Algorithm Convergence & Selected Feature Subset

The Genetic Algorithm converged at Generation 27 with a peak CV F1-Score of **0.5054**.

### Selected Optimal Features (14 features):
1. `Age`
2. `BusinessTravel`
3. `EnvironmentSatisfaction`
4. `JobRole`
5. `JobSatisfaction`
6. `MaritalStatus`
7. `NumCompaniesWorked`
8. `OverTime`
9. `TotalWorkingYears`
10. `TrainingTimesLastYear`
11. `WorkLifeBalance`
12. `YearsAtCompany`
13. `YearsInCurrentRole`
14. `YearsSinceLastPromotion`

---

## 4. Comprehensive Performance Metrics (10-Fold Cross-Validation)

| Metric | Fresh Reproduction Value | Historical Reference Value | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **88.01%** | 88.01% | Replicated |
| **Precision** | **74.80%** | 74.80% | Replicated |
| **Recall** | **38.96%** | 38.96% | Replicated |
| **F1-Score** | **51.23%** | 51.23% | Replicated |
| **ROC-AUC** | **87.11%** | 87.11% | Replicated |
| **PR-AUC** | **66.13%** | 66.13% | Replicated |
| **Balanced Accuracy** | **68.21%** | 68.21% | Replicated |
| **Specificity** | **97.47%** | 97.47% | Replicated |
| **Brier Score** | **0.0874** | 0.0874 | Replicated |
| **Total Errors** | **356** | 356 | Replicated |
| **True Positives (TP)** | **187** | 187 | Replicated |
| **False Positives (FP)** | **63** | 63 | Replicated |
| **True Negatives (TN)** | **2426** | 2,426 | Replicated |
| **False Negatives (FN)** | **293** | 293 | Replicated |

---

## 5. Confusion Matrix Breakdown

| Actual / Predicted | Predicted Retained (0) | Predicted Turnover (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Retained (0)** | **2,426** (TN) | **63** (FP) | **2,489** |
| **Actual Turnover (1)** | **293** (FN) | **187** (TP) | **480** |
| **Total Predicted** | **2,719** | **250** | **2,969** |

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
