# BASE PAPER REPRODUCTION: Talebi et al. (2025) on 2,969 Dataset

This repository reproduces the exact Genetic Algorithm (GA) feature selection and LightGBM classification methodology published by Talebi et al. (2025), evaluated directly on our frozen 2,969-profile employee turnover dataset.

## Reference Publication
- **Title**: *Developing a hybrid machine learning model for employee turnover prediction: Integrating LightGBM and genetic algorithms*
- **Authors**: Hojat Talebi, Amid Khatibi Bardsiri, Vahid Khatibi Bardsiri
- **Publication**: *Journal of Open Innovation: Technology, Market, and Complexity*, Volume 11, Issue 1 (2025), Article 100557
- **ISSN / DOI**: ISSN 2199-8531 / DOI 10.1016/j.joitmc.2025.100557
- **Source PDF**: `base_paper/1-s2.0-S2199853125000927-main.pdf`

---

## Directory Structure

```
BASE_PAPER_REPRODUCE/
├── audit_backup/                                # Preserved pre-audit state and historical runs
├── base_paper/
│   └── 1-s2.0-S2199853125000927-main.pdf       # Original publication PDF
├── data/
│   └── merged_unique_profiles_2969.csv         # Frozen 2,969-profile dataset
├── outputs/                                    # Generated artifacts, figures, and manifests
│   ├── base_paper_protocol_audit.md            # Exhaustive parameter-by-parameter audit
│   ├── base_paper_reproducibility_manifest.json # Complete machine-readable reproducibility manifest
│   ├── base_paper_methodology_registry.json
│   ├── base_paper_method_on_2969_results.csv
│   ├── base_paper_vs_final_comparison.png      # 4-panel visual comparison vs Step 32
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── feature_preprocessing_registry.csv
│   ├── ga_fitness_convergence.png
│   ├── ga_generation_history.csv
│   ├── metrics.json
│   ├── predictions.csv
│   ├── pr_curve.png
│   ├── reproducibility_config.json
│   ├── resampling_registry.csv
│   ├── roc_curve.png
│   ├── selected_features.json
│   └── selected_feature_list.txt
├── reports/
│   ├── BASE_PAPER_CORRECTION_LOG.md            # Detailed log of all audit rectifications
│   ├── BASE_PAPER_EXACT_REPRODUCTION_AUDIT.md  # Comprehensive formal reproduction report
│   └── BASE_PAPER_REPRODUCTION_2969.md         # Baseline experimental execution summary
├── src/
│   ├── __init__.py
│   └── data_loader.py                          # Self-contained data loading & hygiene module
├── run_base_paper.py                           # Standalone reproduction execution script
└── README.md                                   # This repository documentation
```

---

## Verified Experimental Protocol (Talebi et al., 2025 Exact)

1. **Preprocessing (Page 5, Section 3.1)**:
   - Missing-value imputation via mean for continuous variables and mode for categorical variables.
   - Categorical variables represented as integers for native LightGBM handling.
   - Continuous variables normalized strictly using **Min-Max scaling** to [0, 1].
   - **No resampling** (native class distribution preserved: 83.83% retained, 16.17% turnover).
2. **Genetic Algorithm Feature Selection (Pages 6-7, Section 3.2.1)**:
   - Population size: 50 binary chromosomes.
   - Selection: Roulette Wheel Selection (fitness proportional).
   - Crossover: Uniform crossover ($p_c = 0.80$).
   - Mutation: Bit-flip mutation ($p_m = 0.10$ per gene).
   - Elitism: Top 5 individuals (10% of population) carried over directly.
   - Fitness function: **10-Fold Stratified Cross-Validation LightGBM F1-Score**.
   - Stopping rule: Maximum generations with early stopping after 10 consecutive generations without improvement.
3. **Classifier Configuration (Page 8, Section 3.3)**:
   - Algorithm: LightGBM (GBDT, leaf-wise growth).
   - Grid-search parameters: `learning_rate = 0.10`, `max_depth = 10`, `num_leaves = 31`, `subsample = 0.80`, `n_estimators = 100`.
   - Decision threshold: 0.50 default binary classification.
4. **Evaluation Procedure (Pages 8, 10, 13)**:
   - 10-Fold Stratified Cross-Validation across all 2,969 employee profiles.

---

## Final Verified Reproduced Performance (10-Fold CV on 2,969 Profiles)

- **GA-Selected Features (11 features)**:
  `Age`, `BusinessTravel`, `EnvironmentSatisfaction`, `JobSatisfaction`, `MaritalStatus`, `NumCompaniesWorked`, `TotalWorkingYears`, `TrainingTimesLastYear`, `WorkLifeBalance`, `YearsSinceLastPromotion`, `YearsWithCurrManager`
- **Accuracy**: 92.59% (2,749 / 2,969 correct)
- **Precision**: 88.46% (299 / 338)
- **Recall (Sensitivity)**: 62.29% (299 / 480)
- **F1-Score**: 73.11%
- **ROC-AUC**: 93.95%
- **PR-AUC**: 83.40%
- **Balanced Accuracy**: 80.36%
- **Specificity**: 98.43% (2,450 / 2,489)
- **Total Errors**: 220 (False Positives: 39, False Negatives: 181)

---

## How to Execute

From within the `BASE_PAPER_REPRODUCE/` directory:

```bash
python run_base_paper.py
```
