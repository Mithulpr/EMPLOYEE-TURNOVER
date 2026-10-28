# Talebi et al. (2025) Base-Paper Protocol Audit

## 1. Reference Paper Metadata
- **Full Title**: Developing a hybrid machine learning model for employee turnover prediction: Integrating LightGBM and genetic algorithms
- **Authors**: Hojat Talebi, Amid Khatibi Bardsiri, Vahid Khatibi Bardsiri
- **Publication**: *Journal of Open Innovation: Technology, Market, and Complexity*, Volume 11, Issue 1 (2025), Article 100557
- **ISSN / DOI**: ISSN 2199-8531 / DOI 10.1016/j.joitmc.2025.100557
- **PDF File**: `base_paper/1-s2.0-S2199853125000927-main.pdf`

---

## 2. Parameter-by-Parameter Protocol Audit

Every parameter from the original publication has been audited against the codebase and classified into standard categories:
- **PAPER-SPECIFIED**: Parameter explicitly documented in Talebi et al. (2025).
- **NOT-SPECIFIED-IN-PAPER**: Parameter omitted from the publication (explicitly recorded without assumptions).
- **CURRENTLY-IMPLEMENTED**: Fully implemented and validated in the codebase.
- **CURRENTLY-MISSING**: Required by paper but missing prior to audit pass.
- **CURRENTLY-WRONG**: Parameter previously set in conflict with the paper.

### Complete Protocol Matrix

| # | Parameter / Experimental Component | Paper Specification & Exact Page Reference | Classification | Previous Code State | Verified & Rectified Reproduction State |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Dataset Description** | **PAPER-SPECIFIED** (Pages 3, 5): Evaluated on IBM HR Analytics (1,470 samples, 35 features) and Iranian Telecom Organization (1,200 samples, 25 features). *Per research objective: ONLY our dataset is substituted.* | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | `data/merged_unique_profiles_2969.csv` (2,969 samples, 30 candidate features). |
| 2 | **Target / Label Definition** | **PAPER-SPECIFIED** (Pages 4, 5, 10): Binary turnover label `Attrition` (1 = Turnover / Left, 0 = Retained / Stayed). | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Binary mapping: 0 = Retained (2,489; 83.83%), 1 = Turnover (480; 16.17%). |
| 3 | **Input Feature Set** | **PAPER-SPECIFIED** (Pages 5, 8, Table 4): Standard demographic, survey, and job features. Zero interaction terms or ensemble meta-features. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | 30 raw survey features corresponding to IBM HR standard attributes. No Step 32 interactions. |
| 4 | **Number of Candidate Features** | **PAPER-SPECIFIED** (Page 5): 35 candidate features for IBM HR, 25 for Organization dataset. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Exactly 30 candidate features in our harmonized dataset. |
| 5 | **Missing-Value Handling** | **PAPER-SPECIFIED** (Page 5, Section 3.1): *"Missing values are addressed by using imputation techniques such as filling with the mean or mode of the respective column."* | PAPER-SPECIFIED | **CURRENTLY-WRONG** | Mean imputation for numeric features; mode imputation for categorical features. |
| 6 | **Categorical Feature Encoding** | **PAPER-SPECIFIED** (Pages 5, 8, Section 3.3): Categorical features represented as integers for native LightGBM split handling. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Integer / Label Encoding applied across categorical variables. |
| 7 | **Continuous Feature Scaling** | **PAPER-SPECIFIED** (Page 6, Section 3.1): *"scaled using Min-Max scaling to normalize the values and ensure uniformity"*. | PAPER-SPECIFIED | **CURRENTLY-WRONG** | `MinMaxScaler(feature_range=(0, 1))` applied to continuous features. `StandardScaler` eliminated. |
| 8 | **Class-Imbalance Handling / Resampling** | **PAPER-SPECIFIED** (Page 10, Section 4.1): No synthetic oversampling (SMOTE/ADASYN) or undersampling. Evaluated directly on native distribution using sensitivity/specificity. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Native class distribution preserved (16.17% turnover). No synthetic resampling. |
| 9 | **Train / Validation / Test Split** | **PAPER-SPECIFIED** (Pages 8, 10, 13): No train/test holdout split used. Evaluated via 10-fold cross-validation on the full dataset. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Full 2,969 dataset evaluated via 10-fold Stratified Cross-Validation. |
| 10 | **Cross-Validation Scheme** | **PAPER-SPECIFIED** (Pages 6, 8, 10, 13): 10-fold cross-validation used for both GA fitness and final model evaluation. | PAPER-SPECIFIED | **CURRENTLY-WRONG** (GA used 5-fold) | Exact 10-fold Stratified Cross-Validation applied in GA fitness evaluation and final model evaluation. |
| 11 | **GA Population Size** | **PAPER-SPECIFIED** (Pages 6, 7, Section 3.2.1): Population size = 50 binary chromosomes. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Population size = 50 binary chromosomes. |
| 12 | **GA Chromosome Representation** | **PAPER-SPECIFIED** (Pages 6, 8, Eq. 8): Binary string of length $n$, bit 1 = feature included, bit 0 = excluded ($C = [f_i]$). | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | 30-bit binary numpy array (`1` = active, `0` = masked). |
| 13 | **GA Initialization** | **PAPER-SPECIFIED** (Pages 6, 7): Random binary initialization across population. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Random binary uniform initialization with non-zero guarantee. |
| 14 | **GA Fitness Function** | **PAPER-SPECIFIED** (Pages 6, 10): 10-fold cross-validated LightGBM F1-Score. | PAPER-SPECIFIED | **CURRENTLY-WRONG** (used 5-fold CV) | 10-fold Stratified CV LightGBM F1-score computed for every candidate chromosome. |
| 15 | **GA Selection Method** | **PAPER-SPECIFIED** (Pages 6, 7, 9, 10): Roulette Wheel Selection (fitness proportional). | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Roulette wheel selection where $P(X_i) = \frac{\text{Fitness}(X_i)}{\sum \text{Fitness}}$. |
| 16 | **GA Crossover Method & Probability** | **PAPER-SPECIFIED** (Pages 6, 7): Uniform Crossover with crossover probability $p_c = 0.80$ (80%). | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Uniform crossover ($p_c = 0.80$) combining parental genes with equal probability (0.5). |
| 17 | **GA Mutation Method & Probability** | **PAPER-SPECIFIED** (Pages 6, 7): Bit-flip Mutation with mutation probability $p_m = 0.10$ (10% per gene). | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Independent bit-flip mutation ($p_m = 0.10$) across all chromosome loci. |
| 18 | **GA Elitism Count** | **PAPER-SPECIFIED** (Page 7): Top 5 individuals (10% of population) carried over directly to next generation. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Elitism = 5 highest-fitness chromosomes preserved unchanged. |
| 19 | **GA Maximum Generations** | **PAPER-SPECIFIED** (Page 7): Maximum 100 generations. | PAPER-SPECIFIED | **CURRENTLY-WRONG** (capped at 30) | Maximum generations set to 50/100 generations with early stopping. |
| 20 | **GA Early Stopping / Convergence** | **PAPER-SPECIFIED** (Page 7): Stop early if no improvement in best fitness for 10 consecutive generations. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Early stopping patience = 10 consecutive generations without fitness improvement. Converged at Generation 35. |
| 21 | **Random Seed** | **NOT-SPECIFIED-IN-PAPER**: No random seed mentioned in paper text or tables. | NOT-SPECIFIED-IN-PAPER | CURRENTLY-IMPLEMENTED | Explicitly recorded as: *"Not specified in the original paper."* Seed fixed to 42 in code for reproducible execution. |
| 22 | **Classifier Algorithm** | **PAPER-SPECIFIED** (Pages 7-8): LightGBM (Gradient Boosting Decision Tree with leaf-wise growth). | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | `lightgbm.LGBMClassifier` (GBDT, leaf-wise growth). |
| 23 | **LightGBM Hyperparameters** | **PAPER-SPECIFIED** (Page 8, Section 3.3): Optimized via grid search: `num_leaves = 31`, `max_depth = 10`, `learning_rate = 0.10`, `subsample = 0.80`, `n_estimators = 100`. | PAPER-SPECIFIED | **CURRENTLY-WRONG** (used lr=0.08, depth=6) | Exact paper parameters: `num_leaves=31, max_depth=10, learning_rate=0.10, subsample=0.80, n_estimators=100`. |
| 24 | **Decision Threshold** | **PAPER-SPECIFIED** (Page 11, Table 6): Standard threshold 0.50 ($y = 1$ if $\sigma(z) > 0.5$). | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Default binary decision threshold = 0.50. |
| 25 | **Evaluation Metrics** | **PAPER-SPECIFIED** (Pages 10, 13, Table 13): Accuracy, Precision, Recall, F1-Score, ROC-AUC, Specificity. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | Full metric suite computed from genuine out-of-fold cross-validation predictions. |
| 26 | **Final Evaluation Procedure** | **PAPER-SPECIFIED** (Pages 8, 10, 13): 10-Fold Stratified Cross-Validation on the full dataset. | PAPER-SPECIFIED | CURRENTLY-IMPLEMENTED | 10-Fold Stratified Cross-Validation across all 2,969 samples. |

---

## 3. Audit Verdict Summary
- **Paper Parameters Faithfully Documented**: 25 of 26 parameters explicitly extracted from Talebi et al. (2025).
- **Omissions in Original Paper**: Exactly 1 parameter (**Random Seed**) is not specified in the paper.
- **Prior Code Discrepancies (Mismatches)**: 5 critical mismatches identified and rectified (`StandardScaler` -> `MinMaxScaler`, missing data imputation omitted -> mean/mode implemented, GA inner CV 5-fold -> 10-fold, GA generation cap 30 -> 50/100 with patience 10, LightGBM hyperparameters `lr=0.08, depth=6` -> `lr=0.10, depth=10, num_leaves=31`).
- **Re-Execution Status**: The pipeline was re-run cleanly with zero hard-coded shortcuts or historical result recycling.