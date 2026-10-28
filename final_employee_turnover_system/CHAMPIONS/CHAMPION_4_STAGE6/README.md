# Champion 4 — Stage 6 / Review-2 Primary

## Overview
- **Champion Name:** Champion 4 — Stage 6 / Review-2 Primary
- **Step / Stage:** Stage 6
- **Purpose:** Initial high-performance single-model baseline establishing the benchmark holdout protocol
- **Decision Policy Threshold (tau):** 0.4850

## Model Architecture
Tuned Single-Model CatBoost (depth 6, balanced class weights, tau=0.4850)

## Official Retrospective Holdout Results (N = 594)
- **Accuracy:** 94.95%
- **Precision:** 87.50%
- **Recall:** 80.21%
- **F1-Score:** 83.70%
- **Total Misclassification Errors:** 30 out of 594
- **Confusion Matrix:** `[TN=487, FP=11, FN=19, TP=77]`

## Directory Structure
- `models/`: Model weights, specialist classifiers, or reference pools belonging to this version.
- `data/`: Frozen holdout cohort reference dataset (N = 594 records).
- `results/`: Authoritative prediction output CSV files evaluated on the frozen holdout.
- `reports/`: Audit logs, benchmark verification reports, and selection documentation.
- `figures/`: Confusion matrices, ROC/PR curves, and calibration graphs.
- `scripts/`: Independent execution and verification script (`evaluate_champion4.py`).
- `configs/`: Frozen hyperparameters, threshold definitions, and pipeline configurations.

## How to Verify
Run the standalone evaluation script:
```bash
python scripts/evaluate_champion4.py
```
This script executes pure forward calculation on the preserved holdout predictions without retraining.
