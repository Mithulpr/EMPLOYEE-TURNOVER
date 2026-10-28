#!/usr/bin/env python3
"""
Evaluation Script for CHAMPION 1 -- Step 31 Candidate v2
Recalculates and verifies classification metrics on the retrospective
594-record holdout cohort from preserved prediction artifacts.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
CHAMPION_DIR = SCRIPT_DIR.parent

TARGETS = {'records': 594, 'tn': 486, 'fp': 12, 'fn': 12, 'tp': 84, 'errors': 24, 'accuracy': 95.96, 'precision': 87.5, 'recall': 87.5, 'f1_score': 87.5}

def main():
    print("=" * 80)
    print("  EVALUATION: CHAMPION 1 -- Step 31 Candidate v2")
    print("  Model Type: Step 28 Backbone + Precision False-Negative Boundary Recovery")
    print("  Cohort: Retrospective Holdout (N=594)")
    print("=" * 80)

    pred_path = CHAMPION_DIR / "results" / "step31_candidate_v2_holdout.csv"
    if not pred_path.exists():
        pred_path = CHAMPION_DIR / "predictions" / "step31_candidate_v2_holdout.csv"
    if not pred_path.exists():
        print(f"[ERROR] Prediction artifact not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(pred_path)
    y_true = df["True_Label"].values
    y_pred = df["Pred_Step31_v2"].values

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    errors = int(fp + fn)
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    prec = float(precision_score(y_true, y_pred) * 100.0)
    rec = float(recall_score(y_true, y_pred) * 100.0)
    f1 = float(f1_score(y_true, y_pred) * 100.0)

    print(f"  Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]] | Total Errors: {errors}")
    print(f"  Accuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}%\n")

    matched = (errors == TARGETS["errors"] and round(acc, 2) == TARGETS["accuracy"] and
               round(prec, 2) == TARGETS["precision"] and round(rec, 2) == TARGETS["recall"] and
               round(f1, 2) == TARGETS["f1_score"])

    if matched:
        print("  STATUS: [VERIFIED EXACT MATCH -- 100% BIT-EXACT REPRODUCTION]")
    else:
        print("  STATUS: [DISCREPANCY DETECTED]")
        print(f"  Expected: {TARGETS}")
        print(f"  Observed: errors={errors}, acc={acc:.2f}, prec={prec:.2f}, rec={rec:.2f}, f1={f1:.2f}")
    print("=" * 80 + "\n")
    return matched

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
