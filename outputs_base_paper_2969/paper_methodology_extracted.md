# Authoritative Base Paper Methodology Extraction

**Source Document**: `base paper/1-s2.0-S2199853125000927-main.pdf`
**Full Citation**: Talebi, H., Taghvaee, S., Zolfaghari, M., & Shokouhyar, S. (2025). *A hybrid machine learning framework with genetic algorithm-based feature selection for employee turnover prediction*. Journal of Open Innovation: Technology, Market, and Complexity, 11(1), 100557.

## 1. Core Algorithmic Framework
- **Hybrid GA + LightGBM**: Combines binary Genetic Algorithm global feature optimization with LightGBM gradient boosting.
- **Genetic Algorithm Parameters (Section 3.2.1, Pages 6-7)**:
  - Population: 50 binary chromosomes
  - Fitness Function: 10-fold Cross-Validation F1-Score of LightGBM
  - Selection: Roulette Wheel Selection ($P(i) = f_i / \sum f_k$)
  - Crossover: Uniform Crossover (Crossover Rate $p_c = 0.80$)
  - Mutation: Bit-flip Mutation (Mutation Rate $p_m = 0.10$ per gene)
  - Elitism: Top 5 chromosomes (10%) carried over
  - Termination: 100 generations or early convergence (10 generations)

## 2. Published Benchmark Results (Table 13, Page 14)
| Method | Accuracy | Precision | Recall | F1 Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proposed Method (GA + LightGBM)** | **0.92** | **0.89** | **0.85** | **0.87** | **0.94** |
| Standalone LightGBM | 0.86 | 0.81 | 0.80 | 0.80 | 0.89 |
| Standalone XGBoost | 0.81 | 0.78 | 0.76 | 0.77 | 0.86 |
| Standalone Random Forest | 0.78 | 0.74 | 0.68 | 0.71 | 0.82 |
| Standalone Decision Tree | 0.76 | 0.75 | 0.70 | 0.73 | 0.81 |
| Standalone Logistic Regression | 0.75 | 0.72 | 0.65 | 0.68 | 0.80 |
