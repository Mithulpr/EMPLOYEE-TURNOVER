# Champion 3 — Step 28 Dual Meta-Learner

## Overview
- **Champion Name:** Champion 3 — Step 28 Dual Meta-Learner
- **Step / Stage:** Step 28
- **Purpose:** Dual gradient boosting ensemble integration with Step 27 anchor via Logistic Regression stacking
- **Decision Policy Threshold (tau):** 0.3610

## Model Architecture
Level-1: CatBoost (55%) + LightGBM (45%) + Step 27 Anchor; Level-2: Logistic Regression Meta-Learner (tau=0.3610)

## Official Retrospective Holdout Results (N = 594)
- **Accuracy:** 95.62%
- **Precision:** 88.04%
- **Recall:** 84.38%
- **F1-Score:** 86.17%
- **Total Misclassification Errors:** 26 out of 594
- **Confusion Matrix:** `[TN=487, FP=11, FN=15, TP=81]`

## Directory Structure
- `models/`: Model weights, specialist classifiers, or reference pools belonging to this version.
- `data/`: Frozen holdout cohort reference dataset (N = 594 records).
- `results/`: Authoritative prediction output CSV files evaluated on the frozen holdout.
- `reports/`: Audit logs, benchmark verification reports, and selection documentation.
- `figures/`: Confusion matrices, ROC/PR curves, and calibration graphs.
- `scripts/`: Independent execution and verification script (`evaluate_champion3.py`).
- `configs/`: Frozen hyperparameters, threshold definitions, and pipeline configurations.

## How to Verify
Run the standalone evaluation script:
```bash
python scripts/evaluate_champion3.py
```
This script executes pure forward calculation on the preserved holdout predictions without retraining.
