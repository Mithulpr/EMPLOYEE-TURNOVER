# Champion 2 — Step 30 Severe-FN Recovery

## Overview
- **Champion Name:** Champion 2 — Step 30 Severe-FN Recovery
- **Step / Stage:** Step 30
- **Purpose:** Severe false-negative recovery heuristic addressing high-dissatisfaction leaver omissions
- **Decision Policy Threshold (tau):** 0.3610

## Model Architecture
Step 28 Dual Meta-Learner + Severe False-Negative Recovery Rule (dual-model consensus veto)

## Official Retrospective Holdout Results (N = 594)
- **Accuracy:** 95.79%
- **Precision:** 88.17%
- **Recall:** 85.42%
- **F1-Score:** 86.77%
- **Total Misclassification Errors:** 25 out of 594
- **Confusion Matrix:** `[TN=487, FP=11, FN=14, TP=82]`

## Directory Structure
- `models/`: Model weights, specialist classifiers, or reference pools belonging to this version.
- `data/`: Frozen holdout cohort reference dataset (N = 594 records).
- `results/`: Authoritative prediction output CSV files evaluated on the frozen holdout.
- `reports/`: Audit logs, benchmark verification reports, and selection documentation.
- `figures/`: Confusion matrices, ROC/PR curves, and calibration graphs.
- `scripts/`: Independent execution and verification script (`evaluate_champion2.py`).
- `configs/`: Frozen hyperparameters, threshold definitions, and pipeline configurations.

## How to Verify
Run the standalone evaluation script:
```bash
python scripts/evaluate_champion2.py
```
This script executes pure forward calculation on the preserved holdout predictions without retraining.
