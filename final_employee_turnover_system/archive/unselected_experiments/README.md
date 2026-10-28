# Unselected Experiments Archive

## Overview
This directory serves as the archive registry for intermediate experimental iterations evaluated during the 32-step R&D lifecycle of the Employee Turnover Prediction research initiative.

Across the development trajectory (Steps 1 through 32), dozens of modeling techniques, feature selection strategies, class-weight rebalancing schemes, and ensemble configurations were explored. While five specific models demonstrated pareto-optimal trade-offs on the retrospective cohort and were preserved as the official version collection (**FINAL** and **CHAMPIONS 1–4**), intermediate unselected experiments provide valuable context for ablation and negative result analysis.

## Key Historical Experiment Tracks (Parent Workspace Repository)
The detailed execution traces, run scripts, and raw outputs for these historical experiments are preserved in the parent workspace (`E:\R & D anitgravity`):

1. **Step 1–15: Baseline Exploration & Feature Space Prototyping**
   - Location: `E:\R & D anitgravity\outputs_v2` through `outputs_v12`
   - Tested single estimators (RandomForest, ExtraTrees, GradientBoosting, XGBoost, CatBoost, LightGBM) and threshold scanning.
   - Finding: Single tree models plateaued around 88–91% accuracy due to severe class imbalance (16.2% turnover).

2. **Step 16–27: Multi-Objective & Anchor-Specialist Architectures**
   - Location: `E:\R & D anitgravity\track1_step16_common_features`, `track1_champion_refinement_v2`
   - Introduced Anchor CatBoost with secondary gradient boosting specialist estimators.
   - Preserved Artifacts: Serialized models preserved in `archive/legacy/` (`step27_anchor_catboost.joblib`, `step27_pipeline.joblib`).

3. **Step 28: Dual Meta-Learner (Selected as CHAMPION 3)**
   - Preserved in: `CHAMPIONS/CHAMPION_3_STEP28/`
   - Retrospective performance: 95.62% Accuracy, 26 errors.

4. **Step 29: Genetic Algorithm / Pareto Search Exploration**
   - Location: `E:\R & D anitgravity\artifacts\track1_step28_pareto_ga`
   - Hyperparameter exploration over ensemble weighting vectors. Replaced by targeted error-reduction in Step 30.

5. **Step 30: Severe-FN Recovery (Selected as CHAMPION 2)**
   - Preserved in: `CHAMPIONS/CHAMPION_2_STEP30/`
   - Retrospective performance: 95.79% Accuracy, 25 errors.

6. **Step 31: Candidate v2 Aggressive Optimization (Selected as CHAMPION 1)**
   - Preserved in: `CHAMPIONS/CHAMPION_1_STEP31/`
   - Retrospective performance: 95.96% Accuracy, 24 errors.

7. **Step 32: Dual Meta-Learner with Overtime/Satisfaction Conditional Refinement (Selected as FINAL)**
   - Preserved in: `FINAL/`
   - Retrospective performance: 96.13% Accuracy, 23 errors, 94.5469% ROC-AUC.

8. **Alternative Post-Step-32 Explorations (e.g., Step 37 Prototyping)**
   - Preserved in: `archive/legacy/` (`step37_*.joblib`)
   - Experimental deep ensemble exploration; retained for architectural reference without active production deployment.
