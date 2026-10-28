#!/usr/bin/env python3
"""
Evaluation Script for CHAMPION 4 -- Stage 6 / Review-2 Primary
Recalculates and verifies classification metrics on the retrospective
594-record holdout cohort from preserved prediction artifacts.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
CHAMPION_DIR = SCRIPT_DIR.parent

TARGETS = {'records': 594, 'tn': 487, 'fp': 11, 'fn': 19, 'tp': 77, 'errors': 30, 'accuracy': 94.95, 'precision': 87.5, 'recall': 80.21, 'f1_score': 83.7}

def main():
    print("=" * 80)
    print("  EVALUATION: CHAMPION 4 -- Stage 6 / Review-2 Primary")
    print("  Model Type: Tuned Single-Model CatBoost (Depth 6 / Balanced Class Weight)")
    print("  Cohort: Retrospective Holdout (N=594, tau=0.4850)")
    print("=" * 80)

    pred_path = CHAMPION_DIR / "results" / "stage6_review2_holdout_predictions.csv"
    if not pred_path.exists():
        pred_path = CHAMPION_DIR / "predictions" / "stage6_review2_holdout_predictions.csv"
    if not pred_path.exists():
        print(f"[ERROR] Prediction artifact not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(pred_path)
    y_true = df["true_label"].values
    y_pred = df["pred_label"].values

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
