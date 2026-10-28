# Comprehensive Exploratory Data Analysis & Statistical Validation Report

## 1. Dataset Overview and Structural Validation
- **Source File**: `dataset/merged_unique_profiles_2969.csv`
- **Total Profile Records**: 2969 unique employee instances.
- **Initial Raw Column Count**: 45 features.
- **Cleaned Feature Count**: 31 columns (30 predictors + 1 target).
- **Exact Duplicate Rows**: 0 (0.00% exact duplicate rows).
- **Dataset Partition Breakdown**:
  - `Dataset_A`: 1,496 records (50.39%)
  - `Dataset_B`: 1,473 records (49.61%)

---

## 2. Target Variable Analysis (Attrition)
The target column `Attrition` displays moderate class imbalance:
- **Class 0 (Stayed / Retained)**: 2,489 employees (83.83%)
- **Class 1 (Turnover / Left)**: 480 employees (16.17%)
- **Imbalance Ratio**: 5.185 : 1 ($pprox 5.19 : 1$)

### Key Imbalance Implication:
A trivial majority classifier achieves an Accuracy of 83.83% but has a Recall and F1-score of 0.00%. Consequently, model evaluation must prioritize balanced optimization across **Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Balanced Accuracy**.

---

## 3. Data Hygiene & Critical Anomalies Remediated

### 3.1 MonthlyIncome 10x Scale Harmonization
- **Discovery**: In the raw dataset, records originating from `Dataset_A` had `MonthlyIncome` values scaled by an exact factor of 10 (mean = $64,966.05, median = $49,190.00, max = $199,990.00), whereas records from `Dataset_B` were recorded in standard individual monthly salary dollars (mean = $6,500.23, median = $4,908.00, max = $19,999.00).
- **Remediation**: Standardized `Dataset_A` monthly incomes by dividing by 10.0. Post-harmonization distributions between the two cohorts are statistically identical ($p > 0.90$).

### 3.2 Categorical Typo Rectification
- **Discovery**: In `BusinessTravel`, 7 records contained the string `'TravelRarely'` instead of standard `'Travel_Rarely'`.
- **Remediation**: Recoded `'TravelRarely'` to `'Travel_Rarely'`.

### 3.3 Identifier & Invariant Attribute Pruning
- **Identifiers Excluded**: `EmpID`, `EmployeeID`, `EmployeeNumber` (zero predictive value, high risk of identity memorization/overfitting).
- **Invariants Excluded**: `EmployeeCount` (constant=1), `Over18` (constant='Y'), `StandardHours` (constant=80).
- **Redundant Label Columns Dropped**: `AgeGroup`, `Education_Label`, `EnvironmentSatisfaction_Label`, `JobSatisfaction_Label`, `PerformanceRating_Label`, `WorkLifeBalance_Label`, `SalarySlab` (these text columns perfectly mirrored existing numeric columns).
- **Source Label Excluded**: `_dataset_source` removed to prevent partition-specific artifact learning.

---

## 4. Key HR Statistical Insights & Attrition Drivers

1. **Satisfaction Gradients**:
   - Employees reporting **JobSatisfaction = 1 (Low)** exhibit a turnover rate of **22.8%**, compared to **11.3%** for **JobSatisfaction = 4 (Very High)**.
   - Employees with **EnvironmentSatisfaction = 1 (Low)** exhibit an attrition rate of **25.3%**, compared to **13.4%** for high satisfaction.
   - Employees with **WorkLifeBalance = 1 (Bad)** have a turnover rate exceeding **31.2%**, representing a massive attrition multiplier.

2. **Tenure & Early Career Stagnation**:
   - Turnover peaks sharply during the first 1-3 years at the company (early career hazard period).
   - High ratio of `YearsSinceLastPromotion` relative to `YearsAtCompany` correlates strongly with employee flight risk.

3. **Compensation & Mobility**:
   - Turnover is disproportionately concentrated in lower salary brackets and lower job levels (JobLevel 1 and 2).
   - Employees with stock options (`StockOptionLevel > 0`) exhibit significantly lower attrition rates than employees with zero stock options.

---

## 5. Artifacts and Visualization Manifest
The following diagnostic figures have been generated and saved:
- `outputs/figures/target_distribution.png`: Binary class distribution and percentage breakdown.
- `outputs/figures/income_harmonization_comparison.png`: Before-and-after boxplots of MonthlyIncome harmonization.
- `outputs/figures/satisfaction_vs_attrition.png`: Attrition rates segmented by satisfaction metrics.
- `outputs/figures/tenure_and_promotion_vs_attrition.png`: Kernel density estimates of tenure and promotion intervals.
- `outputs/figures/correlation_matrix.png`: Heatmap of linear correlations across numeric attributes.
