# Comprehensive Model Inventory Report: Official Step 32 Production Pipeline

**Project Root**: `E:\R & D anitgravity\final_employee_turnover_system\`  
**Models Directory**: `E:\R & D anitgravity\final_employee_turnover_system\models\`  
**Archive Directory**: `E:\R & D anitgravity\final_employee_turnover_system\archive\models\`  
**Date of Audit**: September 28, 2026  
**Auditor**: Antigravity Autonomous Coding Agent  
**Audited Baseline**: Official Frozen Step 32 Hybrid Production Champion (23 errors / 96.13% accuracy on 594-record holdout)

---

## 1. Executive Summary

This inventory audit conducts an exhaustive, object-level inspection of all 12 model and pipeline artifacts located in the production `models/` directory of the `final_employee_turnover_system`. 

### Key Findings:
1. **Actual Algorithm Verification**: Estimators and transformers were identified directly by inspecting serialized Python object types, classes, coefficients, and parameter dictionaries—**not** by inferring from file names.
2. **Complete Pipeline Participation**: All 12 artifacts genuinely participate in the verified multi-stage Step 32 champion inference pipeline. Tracing of prediction paths confirmed that Stage 1 preprocessing, Stage 2 anchor prior generation, Stage 3 dual base classifiers, Stage 4 meta-learner stacking, Stage 5 false-negative recovery, and Stage 6 boundary domain refinement each rely on dedicated artifacts.
3. **No Unused Bloat**: Zero orphan or obsolete model binaries exist in the production models directory. Every artifact actively influences either the base predictions, the stacking ensemble, the explainability engine, or the safety override rules.
4. **Historical Naming Obsoleted**: The active artifacts previously carried confusing research stage prefixes (`step27_`, `step30_`, `step37_`). These have been systematically mapped to descriptive, algorithm-specific filenames reflecting their verified roles and components.
5. **Byte-for-Byte Archival**: Prior to any file renaming, all 12 original artifacts were backed up to `final_employee_turnover_system/archive/models/` with 100% cryptographic SHA-256 match.

---

## 2. Comprehensive Artifact Inventory Table

| # | Original Filename | Actual Object Type | Verified Algorithm / Structure | Production Pipeline Role | Usage Pattern | Loading Location & Dependent Modules | Target Clean Filename | Final Status |
|:-:|:---|:---|:---|:---|:---|:---|:---|:---|
| **1** | `step27_anchor_catboost.joblib` | `catboost.core.CatBoostClassifier` | CatBoost GBDT (binary classification) | Stage 2 Anchor Model: Computes prior turnover probability $P_{s27}$ | Evaluated on every prediction | Loaded in `engine/pipeline.py` (L140); fed to `meta_clf` and `X_fn` builder | `anchor_catboost.joblib` | **Active Production** (Renamed) |
| **2** | `step27_pipeline.joblib` | `builtins.dict` | Dictionary containing `fe` (`HRFeatureEngineer`), `prep` (`HRPreprocessor`), `features` (28 features) | Preprocessing & feature selection pipeline for Step 27 anchor model | Executed on every prediction | Loaded in `engine/pipeline.py` (L141); transforms raw input into $df_{full27}$ | `anchor_pipeline.joblib` | **Active Production** (Renamed) |
| **3** | `step30_clf_fn.joblib` | `catboost.core.CatBoostClassifier` | CatBoost GBDT (binary classification) | Stage 5 Severe-FN Recovery Specialist: Recovers false negatives in boundary zone | Evaluated on every prediction; conditionally triggers override ($0 \to 1$) | Loaded in `engine/pipeline.py` (L149); evaluated when $P_{meta} < \tau$ and dual base models signal risk | `false_negative_recovery_catboost.joblib` | **Active Production** (Renamed) |
| **4** | `step30_clf_fp.joblib` | `catboost.core.CatBoostClassifier` | CatBoost GBDT (binary classification) | Stage 5 False-Positive Suppression Specialist: Suppresses false alarm turnover flags | Evaluated on every prediction; conditionally triggers override ($1 \to 0$) | Loaded in `engine/pipeline.py` (L150); evaluated when $P_{meta} \ge \tau$ and CatBoost probability is low | `false_positive_recovery_catboost.joblib` | **Active Production** (Renamed) |
| **5** | `step30_ref_pool.npz` | `numpy.lib.npyio.NpzFile` | Compressed NumPy archive: $P_{cb}, P_{lgb}, P_{s27}, P_{meta}$ (1,900 samples each) | Out-of-fold calibration reference distribution for computing distance and rank features | Accessed on every prediction | Loaded in `engine/pipeline.py` (L151); used by `build_specialized_fn_features()` | `reference_pool.npz` | **Active Production** (Renamed) |
| **6** | `step37_catboost_base.joblib` | `catboost.core.CatBoostClassifier` | CatBoost GBDT (30 selected features) | Stage 3 Level-1 Base Classifier: Computes primary $P_{cb}$ probability | Evaluated on every prediction | Loaded in `engine/pipeline.py` (L144); used by `explainer.py` for SHAP TreeExplainer | `base_catboost.joblib` | **Active Production** (Renamed) |
| **7** | `step37_catboost_specialist_refinement.joblib` | `catboost.core.CatBoostClassifier` | CatBoost GBDT (9 boundary & interaction features) | Stage 6 Step 32 Boundary Specialist: Overrides boundary zone stayers with severe dissatisfaction | Evaluated on every prediction; conditionally triggers override ($0 \to 1$) | Loaded in `engine/pipeline.py` (L154); evaluated on boundary mask $0.15 \le P_{meta} < \tau$ | `specialist_catboost.joblib` | **Active Production** (Renamed) |
| **8** | `step37_feature_engineer.joblib` | `src.feature_engineering.HRFeatureEngineer` | Domain Feature Transformer (ratios, satisfaction, tenure interactions) | Stage 1 Feature Engineering: Generates engineered domain features from raw inputs | Executed on every prediction | Loaded in `engine/pipeline.py` (L136); feeds `add_structural_features()` | `hr_feature_engineer.joblib` | **Active Production** (Renamed) |
| **9** | `step37_lightgbm_base.joblib` | `lightgbm.sklearn.LGBMClassifier` | LightGBM GBDT (30 selected features) | Stage 3 Level-1 Base Classifier: Computes primary $P_{lgb}$ probability | Evaluated on every prediction | Loaded in `engine/pipeline.py` (L145); used by `explainer.py` for SHAP TreeExplainer | `base_lightgbm.joblib` | **Active Production** (Renamed) |
| **10** | `step37_meta_logistic_regression.joblib` | `sklearn.linear_model.LogisticRegression` | L2-regularized Logistic Regression (6 meta-features, intercept = -3.77338) | Stage 4 Level-2 Stacking Meta-Learner: Calibrates base probabilities into ensemble probability $P_{meta}$ | Evaluated on every prediction | Loaded in `engine/pipeline.py` (L146); evaluated against decision threshold $\tau = 0.3610$ | `meta_logistic_regression.joblib` | **Active Production** (Renamed) |
| **11** | `step37_pipeline_schema.json` | `builtins.dict` (JSON) | Schema dictionary (`features_30`, `decision_threshold`, `refinement_rules`) | Pipeline Configuration: Specifies 30 authoritative schema features and threshold $\tau=0.3610$ | Loaded once at startup | Loaded in `engine/pipeline.py` (L130); consumed by prediction and explanation engines | `preprocessing_schema.json` | **Active Production** (Renamed) |
| **12** | `step37_preprocessor.joblib` | `src.feature_selection.HRPreprocessor` | Scikit-Learn based preprocessor (StandardScaler, MedianImputer, OneHotEncoder) | Stage 1 Feature Selection & Scaling: Imputes missing values, encodes categories, standardizes scales | Executed on every prediction | Loaded in `engine/pipeline.py` (L137); outputs scaled 30-feature vector | `preprocessor.joblib` | **Active Production** (Renamed) |

---

## 3. Detailed Component-by-Component Audit

### Component 1: `step27_anchor_catboost.joblib` & `step27_pipeline.joblib`
- **Actual Object Types**: `CatBoostClassifier` and `dict` containing fitted `HRFeatureEngineer` and `HRPreprocessor` instances.
- **Role in Hybrid Architecture**: Represents the foundational Step 27 Pareto multi-objective Anchor model. While later stages introduced LightGBM and stacking, Step 27 retains high recall sensitivity on critical attrition clusters.
- **Input Features**: 28 specific features extracted via `df_full27[s27_feats]`.
- **Inference Output**: $P_{s27} = P(\text{Attrition}=1 \mid X_{28})$.
- **Consumers**: Directly fed as feature index 2 of `X_meta` to `meta_clf`, and into the `X_fn` recovery feature matrix.
- **Conclusion**: Essential production component. Renamed to `anchor_catboost.joblib` and `anchor_pipeline.joblib`.

### Component 2: `step37_catboost_base.joblib` & `step37_lightgbm_base.joblib`
- **Actual Object Types**: `CatBoostClassifier` and `LGBMClassifier` (30 features in `n_features_in_`).
- **Role in Hybrid Architecture**: The primary Stage 3 base learners. Both independently score the authoritative 30-feature domain representation.
- **Inference Output**: $P_{cb}$ and $P_{lgb}$.
- **Consumers**: Fed to `meta_clf` as features 0 and 1, used to compute confidence spread $|P_{cb} - P_{lgb}|$ and certainty metrics $|P_{cb} - 0.5|$, and utilized by `SHAPExplainerEngine` for local TreeExplainer additive attribution decompositions.
- **Conclusion**: Essential production component. Renamed to `base_catboost.joblib` and `base_lightgbm.joblib`.

### Component 3: `step37_meta_logistic_regression.joblib`
- **Actual Object Type**: `LogisticRegression` from `sklearn.linear_model._logistic`.
- **Fitted Parameters**:
  - `coef_`: `[[2.36476, 0.71845, 3.73016, 1.10467, -0.30667, -0.54904]]`
  - `intercept_`: `[-3.77338]`
  - Number of input features: 6
- **Input Feature Vector**: $[P_{cb}, P_{lgb}, P_{s27}, |P_{cb} - 0.5|, |P_{lgb} - 0.5|, |P_{cb} - P_{lgb}|]$.
- **Inference Output**: Calibrated ensemble probability $P_{meta}$.
- **Decision Policy**: Compared against decision threshold $\tau = 0.3610$ to produce baseline classification $\text{pred}_{base}$.
- **Conclusion**: Essential production component. Renamed to `meta_logistic_regression.joblib`.

### Component 4: `step30_clf_fn.joblib`, `step30_clf_fp.joblib` & `step30_ref_pool.npz`
- **Actual Object Types**: Two specialized `CatBoostClassifier` models and a 4-array `.npz` archive.
- **Role in Hybrid Architecture**: Stage 5 recovery specialists trained on out-of-fold calibration errors.
- **Mechanics**: When $P_{meta} < \tau$ (model predicts stay), but both base learners exhibit elevated risk ($\ge 0.10$) and $P_{fn} \ge 0.35$, the recovery rule rescues the false negative. Empirical validation confirms that on the 594-record holdout, exactly 1 false negative is recovered by this rule.
- **Conclusion**: Essential production component. Renamed to `false_negative_recovery_catboost.joblib`, `false_positive_recovery_catboost.joblib`, and `reference_pool.npz`.

### Component 5: `step37_catboost_specialist_refinement.joblib`
- **Actual Object Type**: `CatBoostClassifier` trained on 9 domain features (`SevereDissatisfactionFlag`, `YearsWithCurrManager`, `JobHoppingRate`, `TotalExpToAgeRatio`, `PromotionPaceStrain`, `TotalWorkingYears`, $P_{cb}$, $P_{lgb}$, $P_{28}$).
- **Role in Hybrid Architecture**: Stage 6 Step 32 Boundary Domain Specialist. Recovers vulnerable employees situated in the boundary zone ($0.15 \le P_{meta} < 0.3610$) who exhibit severe dissatisfaction ($> 0.5$).
- **Impact**: Empirically confirms recovery of 2 boundary zone records on the authoritative holdout, bringing total misclassifications down to the verified benchmark of 23 errors.
- **Conclusion**: Essential production component. Renamed to `specialist_catboost.joblib`.

### Component 6: `step37_feature_engineer.joblib`, `step37_preprocessor.joblib` & `step37_pipeline_schema.json`
- **Actual Object Types**: Custom domain transformers and schema JSON.
- **Role in Hybrid Architecture**: Stage 1 transformation pipeline. Converts raw employee profile dictionaries into the verified 30-feature numerical matrix.
- **Conclusion**: Essential production components. Renamed to `hr_feature_engineer.joblib`, `preprocessor.joblib`, and `preprocessing_schema.json`.

---

## 4. Separation from Research Gap Extension Models

The models in `research_gap_extension/models/` are completely separate experimental artifacts:
- `research_gap_extension/models/anchor_catboost.joblib` (124,858 bytes)
- `research_gap_extension/models/boundary_specialist.joblib` (37,720 bytes)
- `research_gap_extension/models/catboost_base.joblib` (277,090 bytes)
- `research_gap_extension/models/hr_feature_engineer.joblib` (312 bytes)
- `research_gap_extension/models/lightgbm_base.joblib` (424,804 bytes)
- `research_gap_extension/models/meta_logistic_regression.joblib` (911 bytes)
- `research_gap_extension/models/pipeline_schema.json` (1,351 bytes)
- `research_gap_extension/models/scaler.joblib` (2,495 bytes)

These 8 experimental artifacts were trained as part of the Talebi et al. (2025) 5-fold nested cross-validation ablation study. They are self-contained inside `research_gap_extension/` and are **not** mixed with or promoted into the production `models/` directory.

---

## 5. Byte-Exact Cryptographic Hash Verification

| Artifact Role | Original Filename | Clean Filename | SHA-256 Digest |
|:---|:---|:---|:---|
| Stage 2 Anchor Model | `step27_anchor_catboost.joblib` | `anchor_catboost.joblib` | `80cbb01ff4c9950aef99c7b97cad3d9ed2b8598738bae74a6d084b66a1ce51a9` |
| Stage 2 Anchor Pipeline | `step27_pipeline.joblib` | `anchor_pipeline.joblib` | `d0c0120ed340c2552c61a4e91c322658e70a46eb5664d0e9a192f868f789d5c1` |
| Stage 5 FN Recovery Model | `step30_clf_fn.joblib` | `false_negative_recovery_catboost.joblib` | `39ff32975b8ff5e369a81a0a4a653a44a3038c1d58ac325710e68b5b24e0c4d5` |
| Stage 5 FP Suppression Model | `step30_clf_fp.joblib` | `false_positive_recovery_catboost.joblib` | `cce32be26924117eb278458259afa4a32a724ec8020597a690cea3fb63b38089` |
| Stage 5 Reference Pool | `step30_ref_pool.npz` | `reference_pool.npz` | `a1bfb14a72cbac22a5c7f4a0a02bd8b127e1aee2e0852f5e937278b2c69a1928` |
| Stage 3 Base CatBoost Model | `step37_catboost_base.joblib` | `base_catboost.joblib` | `409fcb1adef0f6f84340744ff88f2a03dade0a7ec078b3aeae530706c3f40315` |
| Stage 6 Boundary Specialist | `step37_catboost_specialist_refinement.joblib` | `specialist_catboost.joblib` | `c9fead0570607c6d4bd39639f3aa607985652009c7be180eac9385a85a277544` |
| Stage 1 Feature Engineer | `step37_feature_engineer.joblib` | `hr_feature_engineer.joblib` | `0772868eb01575415ad8ab2ee1a40d6a2cf6ea6775db03b11d4ddecae5b23fe9` |
| Stage 3 Base LightGBM Model | `step37_lightgbm_base.joblib` | `base_lightgbm.joblib` | `0ab6411a44c90d27a580fdac4314ff17a6257831addf74302fd2108e6ed5abc9` |
| Stage 4 Stacking Meta-Learner | `step37_meta_logistic_regression.joblib` | `meta_logistic_regression.joblib` | `f3f615b3c925633745c8cfee7f273e8974cde891cc44192ccd0e34461137084b` |
| Pipeline Schema & Config | `step37_pipeline_schema.json` | `preprocessing_schema.json` | `424302fa567bcc63142b3a61b2e560b2259bea336c107f12f806b1fe598b37dd` |
| Stage 1 Feature Preprocessor | `step37_preprocessor.joblib` | `preprocessor.joblib` | `25175dc3ff8b18fd02a101153a87e1676ec5bd7f2ecc89bcd72aeb44dc8ec34f` |
