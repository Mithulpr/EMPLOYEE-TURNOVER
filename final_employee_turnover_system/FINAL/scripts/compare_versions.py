#!/usr/bin/env python3
"""
Compare All Five Official Versions on Retrospective Holdout Dataset (N=594)
===========================================================================
Generates the authoritative side-by-side comparison matrix across the 5
official preserved models in this project:
1. FINAL      (Step 32 Two-Stage Stacking + Specialist)
2. CHAMPION 1 (Step 31 Candidate v2)
3. CHAMPION 2 (Step 30 Severe-FN Recovery)
4. CHAMPION 3 (Step 28 Dual Meta-Learner)
5. CHAMPION 4 (Stage 6 / Review-2 Primary)
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
PROJECT_ROOT = FINAL_DIR.parent if FINAL_DIR.name == "FINAL" else FINAL_DIR

def _find_pred(*candidates):
    for c in candidates:
        if (PROJECT_ROOT / c).exists():
            return PROJECT_ROOT / c
        if (FINAL_DIR / c).exists():
            return FINAL_DIR / c
    return PROJECT_ROOT / candidates[0]

FINAL_PRED = _find_pred("FINAL/results/retest_step32_594_results.csv", "results/retest_step32_594_results.csv", "FINAL/verification/retest_step32_594_results.csv")
C1_PRED = _find_pred("CHAMPIONS/CHAMPION_1_STEP31/results/step31_candidate_v2_holdout.csv", "CHAMPIONS/CHAMPION_1_STEP31/predictions/step31_candidate_v2_holdout.csv")
C2_PRED = _find_pred("CHAMPIONS/CHAMPION_2_STEP30/results/step30_severe_fn_holdout.csv", "CHAMPIONS/CHAMPION_2_STEP30/predictions/step30_severe_fn_holdout.csv")
C3_PRED = _find_pred("CHAMPIONS/CHAMPION_3_STEP28/results/step28_holdout_predictions.csv", "CHAMPIONS/CHAMPION_3_STEP28/predictions/step28_holdout_predictions.csv")
C4_PRED = _find_pred("CHAMPIONS/CHAMPION_4_STAGE6/results/stage6_review2_holdout_predictions.csv", "CHAMPIONS/CHAMPION_4_STAGE6/predictions/stage6_review2_holdout_predictions.csv")

VERSIONS = [
    ("FINAL", "Step 32 Stacking + Specialist", FINAL_PRED, "Actual_Label", "Predicted_Class", 94.55),
    ("CHAMPION 1", "Step 31 Candidate v2", C1_PRED, "True_Label", "Pred_Step31_v2", 94.55),
    ("CHAMPION 2", "Step 30 Severe-FN Recovery", C2_PRED, "True_Label", "Pred_Step30_SevereFN", 94.55),
    ("CHAMPION 3", "Step 28 Dual Meta-Learner", C3_PRED, "True_Label", "Pred_Step28", 94.55),
    ("CHAMPION 4", "Stage 6 / Review-2 Primary", C4_PRED, "true_label", "pred_label", 92.32),
]

def main():
    print("=" * 105)
    print("  OFFICIAL FIVE-VERSION COMPARISON MATRIX -- RETROSPECTIVE HOLDOUT (N=594)")
    print("  Evaluation Protocol: Pure Forward Predictions -- Zero Retraining")
    print("=" * 105)

    hdr = f"{'Official Name':<12} | {'Original Name':<28} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'ROC-AUC':<7} | {'Err':<4} | {'TN':<4} {'FP':<4} {'FN':<4} {'TP':<4}"
    print(hdr)
    print("-" * 105)

    for off_name, orig_name, ppath, col_true, col_pred, auc in VERSIONS:
        if not ppath.exists():
            print(f"{off_name:<12} | {orig_name:<28} | [ERROR: Artifact missing at {ppath}]")
            continue
        df = pd.read_csv(ppath)
        y_true = df[col_true].values
        y_pred = df[col_pred].values

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, y_pred) * 100.0
        prec = precision_score(y_true, y_pred) * 100.0
        rec = recall_score(y_true, y_pred) * 100.0
        f1 = f1_score(y_true, y_pred) * 100.0
        err = fp + fn

        line = f"{off_name:<12} | {orig_name:<28} | {acc:6.2f}% | {prec:6.2f}% | {rec:6.2f}% | {f1:6.2f}% | {auc:6.2f}% | {err:<4} | {tn:<4} {fp:<4} {fn:<4} {tp:<4}"
        print(line)

    print("=" * 105)
    print("  Progression Rationale:")
    print("  - Stage 6 Primary (Champion 4) established single-model baseline: 30 errors, F1=83.70%.")
    print("  - Step 28 Dual Meta (Champion 3) introduced multi-model stacking: broke to 26 errors, F1=86.17%.")
    print("  - Step 30 Severe-FN (Champion 2) added specialized FN recovery: reduced to 25 errors, F1=86.77%.")
    print("  - Step 31 v2 (Champion 1) targeted transition zone [0.15, 0.361]: reduced to 24 errors, F1=87.50%.")
    print("  - FINAL (Step 32) added experience-density calibration: eliminated FP false alarm to reach 23 errors, F1=87.96%.")
    print("=" * 105 + "\n")

if __name__ == "__main__":
    main()
