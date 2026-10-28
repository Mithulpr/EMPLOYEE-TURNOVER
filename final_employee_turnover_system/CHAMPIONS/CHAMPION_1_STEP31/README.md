# Champion 1 — Step 31 Candidate v2

## Overview
- **Champion Name:** Champion 1 — Step 31 Candidate v2
- **Step / Stage:** Step 31
- **Purpose:** Precision False-Negative boundary recovery on Step 28 dual-meta backbone
- **Decision Policy Threshold (tau):** 0.3610

## Model Architecture
CatBoost + LightGBM + Step 27 Anchor combined via Logistic Regression Meta-Learner (tau=0.3610) with candidate v2 decision boundary refinement

## Official Retrospective Holdout Results (N = 594)
- **Accuracy:** 95.96%
- **Precision:** 87.50%
- **Recall:** 87.50%
- **F1-Score:** 87.50%
- **Total Misclassification Errors:** 24 out of 594
- **Confusion Matrix:** `[TN=486, FP=12, FN=12, TP=84]`

## Directory Structure
- `models/`: Model weights, specialist classifiers, or reference pools belonging to this version.
- `data/`: Frozen holdout cohort reference dataset (N = 594 records).
- `results/`: Authoritative prediction output CSV files evaluated on the frozen holdout.
- `reports/`: Audit logs, benchmark verification reports, and selection documentation.
- `figures/`: Confusion matrices, ROC/PR curves, and calibration graphs.
- `scripts/`: Independent execution and verification script (`evaluate_champion1.py`).
- `configs/`: Frozen hyperparameters, threshold definitions, and pipeline configurations.

## How to Verify
Run the standalone evaluation script:
```bash
python scripts/evaluate_champion1.py
```
This script executes pure forward calculation on the preserved holdout predictions without retraining.
