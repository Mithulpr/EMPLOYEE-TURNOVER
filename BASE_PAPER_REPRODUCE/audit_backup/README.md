# BASE PAPER REPRODUCTION: Talebi et al. (2025) on 2,969 Dataset

This repository reproduces the exact Genetic Algorithm (GA) feature selection and LightGBM classification methodology published by Talebi et al. (2025), evaluated directly on our frozen 2,969-profile employee turnover dataset.

## Reference Base Paper
- **Authors**: H. Talebi, S. Taghvaee, M. Zolfaghari, S. Shokouhyar
- **Year**: 2025
- **Title**: *A hybrid machine learning framework with genetic algorithm-based feature selection for employee turnover prediction*
- **Journal**: *Journal of Open Innovation: Technology, Market, and Complexity*, 11(1), 100557
- **Source PDF**: `base_paper/1-s2.0-S2199853125000927-main.pdf`

---

## Directory Structure

```
BASE_PAPER_REPRODUCE/
├── base_paper/
│   └── 1-s2.0-S2199853125000927-main.pdf    # Base paper publication
├── data/
│   └── merged_unique_profiles_2969.csv      # Unified 2,969-profile dataset
├── outputs/                                 # Reproduction figures, tables & configs
│   ├── base_paper_methodology_registry.json
│   ├── feature_preprocessing_registry.csv
│   ├── resampling_registry.csv
│   ├── selected_features.json
│   ├── selected_feature_list.txt
│   ├── ga_generation_history.csv
│   ├── ga_fitness_convergence.png
│   ├── predictions.csv
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── metrics.json
│   ├── base_paper_method_on_2969_results.csv
│   └── reproducibility_config.json
├── reports/
│   └── BASE_PAPER_REPRODUCTION_2969.md      # Comprehensive academic report
├── src/
│   ├── __init__.py
│   └── data_loader.py                       # Self-contained data hygiene module
├── run_base_paper.py                        # Standalone execution pipeline
└── README.md                                # This documentation
```

---

## Methodology Protocol

1. **Dataset**: 2,969 unique employee profiles, 30 raw survey features, 16.17% turnover rate (480 positives, 2,489 negatives).
2. **Preprocessing**:
   - Categorical attributes: LabelEncoder.
   - Continuous numerical features: StandardScaler.
   - No synthetic oversampling (native distribution preserved).
   - Zero domain-engineered interaction terms.
3. **Genetic Algorithm Feature Selection**:
   - Chromosome representation: Binary mask across 30 raw features.
   - Population size: 50 chromosomes.
   - Selection mechanism: Roulette-wheel selection.
   - Crossover: Uniform crossover ($p_c = 0.80$).
   - Mutation: Bit-flip mutation ($p_m = 0.10$).
   - Elitism: Top 5 chromosomes preserved directly.
   - Fitness function: Stratified Cross-Validation F1-score of LightGBM.
   - Termination: Convergence with early stopping (10 generations patience).
4. **Classification Architecture**:
   - Algorithm: LightGBM (`learning_rate=0.08`, `num_leaves=31`, `max_depth=6`, `subsample=0.8`).
   - Evaluation Protocol: 10-Fold Stratified Cross-Validation on all 2,969 records.

---

## Execution Instructions

The reproduction pipeline uses strictly relative, self-contained paths and requires no external path dependencies.

### Command 1: Run directly from reproduction directory
```powershell
cd "E:\R&D final\BASE_PAPER_REPRODUCE"
python run_base_paper.py
```

### Command 2: Run from project root
```powershell
python "E:\R&D final\BASE_PAPER_REPRODUCE\run_base_paper.py"
```

---

## Fresh Reproduction Results (10-Fold Stratified CV on 2,969)

All metrics are freshly calculated from the execution run:

| Metric | Empirical Value | Historical Reference Target | Status |
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
| **Classification Errors** | **356 / 2,969** | 356 | Replicated |
| **Selected Features** | **14 Features** | 14 Features | Replicated |

### Optimal Selected 14 Features:
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
