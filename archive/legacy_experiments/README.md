# Historical Experiments Archive Registry

## Overview
This registry summarizes the experimental history and intermediate research trajectories conducted throughout the development of the Employee Turnover Prediction initiative.

## Development Trajectory (Steps 1–32)
1. **Steps 1–15: Baseline Exploration & Feature Space Prototyping**
   - Single-model estimators (RandomForest, ExtraTrees, GradientBoosting, CatBoost, LightGBM).
   - Finding: Tree models plateaued at 88–91% accuracy due to severe class imbalance (16.2% turnover).
2. **Steps 16–27: Anchor-Specialist Exploration & Multi-Objective Optimization**
   - Introduced dual-tier architecture with Anchor CatBoost and secondary gradient-boosting specialists.
3. **Step 28: Dual Meta-Learner (CHAMPION 3)**
   - Stacking classifier with logistic regression meta-learner: 95.62% accuracy, 26 errors on 594 holdout.
4. **Step 30: Severe-FN Recovery Specialist (CHAMPION 2)**
   - Targeted False Negative rescue logic: 95.79% accuracy, 25 errors on 594 holdout.
5. **Step 31: Candidate v2 Aggressive Optimization (CHAMPION 1)**
   - Transition-zone boundary scanning [0.15, 0.361]: 95.96% accuracy, 24 errors on 594 holdout.
6. **Step 32: Production Architecture (FINAL)**
   - Two-tier stacking with domain specialist & experience-density calibration: 96.13% accuracy, 23 errors, 94.5469% ROC-AUC.
   - Preserved in active production repository at: `final_employee_turnover_system/`

## Pre-Cleanup Full External Backup
All intermediate raw predictions, training logs, hyperparameter sweep files, and trial checkpoints from unselected experiments (Steps 1–36) have been cryptographically preserved in the external backup:
- **Location**: `E:\R & D anitgravity_backup_20261001_213158`
- **Total Files**: 2,576 files (1,070.80 MB)
- **Manifest**: `backup_manifest.json` (SHA-256 verified)
