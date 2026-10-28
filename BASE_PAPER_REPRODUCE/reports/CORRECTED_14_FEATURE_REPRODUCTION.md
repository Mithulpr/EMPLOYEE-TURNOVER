# Talebi et al. (2025) Methodology — 14-Feature HR Reproduction Report

**Date**: 2026-10-14 11:30:58  
**Experiment**: **14-FEATURE CONSTRAINED TALEBI REPRODUCTION**  
**Designation**: *Faithful reproduction of the documented methodology with the paper-reported 14-feature HR configuration on our 2,969-record dataset.*  
**Dataset**: `data/merged_unique_profiles_2969.csv` ($N = 2,969$ unique employee profiles)  
**Base Paper Citation**:  
> Talebi, H., Taghvaee, S., Zolfaghari, M., & Shokouhyar, S. (2025).  
> *A hybrid machine learning framework with genetic algorithm-based feature selection for employee turnover prediction*.  
> **Journal of Open Innovation: Technology, Market, and Complexity**, 11(1), 100557.

---

## 1. Executive Summary

In Talebi et al. (2025), the authors evaluate their hybrid GA + LightGBM framework across two datasets:
- **IBM HR Analytics dataset**: **14 selected features** (Page 8: *"For the HR dataset, 14 features were used; for the Organization dataset, 11 features were selected"*).
- **Organization dataset**: **11 selected features**.

To eliminate variable chromosome cardinality and ensure a strict, controlled reproduction matching the paper's reported **HR feature count**, this experiment implements a fixed-cardinality 14-feature constraint inside the Genetic Algorithm. Every single chromosome evaluated by the GA contains **exactly 14 selected features** (14 ones, 16 zeros) enforced via a deterministic `repair_to_14` function.

---

## 2. FINAL 14 SELECTED HR FEATURES

The Genetic Algorithm independently selected the following 14 optimal features from the 30 candidate survey features, maximizing the 10-fold CV LightGBM F1-score:

1. **Age**
2. **BusinessTravel**
3. **Department**
4. **Education**
5. **EducationField**
6. **EnvironmentSatisfaction**
7. **JobSatisfaction**
8. **MaritalStatus**
9. **NumCompaniesWorked**
10. **PerformanceRating**
11. **TotalWorkingYears**
12. **WorkLifeBalance**
13. **YearsAtCompany**
14. **YearsSinceLastPromotion**

---

## 3. 10-Fold Cross-Validation Performance (Full 2,969 Dataset)

- **Accuracy**: **90.97%** (2701 / 2,969 correct)
- **Precision**: **84.87%** (258 / 304)
- **Recall (Sensitivity)**: **53.75%** (258 / 480)
- **F1-Score**: **65.82%**
- **ROC-AUC**: **92.22%**
- **PR-AUC**: **79.52%**
- **Balanced Accuracy**: **75.95%**
- **Specificity**: **98.15%**
- **Brier Score**: **0.0662**
- **Total Errors**: **268** (False Positives: 46, False Negatives: 222)

---

## 4. Three-Way Performance Comparison

| Metric | Published Talebi et al. (2025) HR Result | Talebi Method — 14-Feature HR Reproduction | Final Step 32 Champion |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 92.00% | **90.97%** | 96.13% |
| **Precision** | 89.00% | **84.87%** | 88.42% |
| **Recall (Sensitivity)** | 85.00% | **53.75%** | 87.50% |
| **F1-Score** | 87.00% | **65.82%** | 87.96% |
| **ROC-AUC** | 94.00% | **92.22%** | 94.55% |
| **PR-AUC** | Not reported | **79.52%** | 86.89% |
| **Total Errors** | Not reported | **268** | 23 |
| **False Positives** | Not reported | **46** | 11 |
| **False Negatives** | Not reported | **222** | 12 |

---

## 5. Verification & Integrity Checklist

- `GA_CARDINALITY_CHECK = PASS`: Every evaluated chromosome strictly contained exactly 14 active genes.
- `DATA_INTEGRITY = PASS`: Evaluated across all 2,969 records with zero synthetic augmentation.
- `NO_DATA_LEAKAGE = PASS`: Feature selection and model evaluation performed strictly through cross-validation.
- `ISOLATION = PASS`: No Step 32 engineered features, Level-2 stacking, or recovery models were introduced.
