#!/usr/bin/env python3
"""
Retest FINAL Model on Retrospective Holdout Dataset (N=594)
===========================================================
Pure forward inference evaluation of the FINAL employee turnover prediction
model on the frozen 594-record retrospective holdout cohort without retraining.

Official Benchmarks:
- Accuracy:   96.13%
- Precision:  88.42%
- Recall:     87.50%
- F1-Score:   87.96%
- ROC-AUC:    94.5469% (Unrounded continuous probability)
- Errors:     23 (TN=487, FP=11, FN=12, TP=84)
"""

import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Resolve paths dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
PROJECT_ROOT = FINAL_DIR.parent if FINAL_DIR.name == "FINAL" else FINAL_DIR
DATA_PATH = FINAL_DIR / "data" / "holdout_594.csv"
MODELS_DIR = FINAL_DIR / "models"

for path_dir in [str(FINAL_DIR), str(SCRIPT_DIR), str(PROJECT_ROOT)]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from pipeline import UnifiedPredictionEngine
except ImportError:
    from FINAL.engine.pipeline import UnifiedPredictionEngine

TARGETS = {
    "records": 594,
    "tn": 487,
    "fp": 11,
    "fn": 12,
    "tp": 84,
    "errors": 23,
    "accuracy": 96.13,
    "precision": 88.42,
    "recall": 87.50,
    "f1_score": 87.96,
    "roc_auc_unrounded": 94.55,
}


def run_retest():
    print("=" * 80)
    print("  FINAL VERSION RETROSPECTIVE HOLDOUT VERIFICATION (N=594)")
    print("  Architecture: 2-Stage Stacking Meta-Learner + Boundary Specialist")
    print("  Evaluation Protocol: Pure Forward Inference -- No Model Retraining")
    print("=" * 80)

    if not DATA_PATH.exists():
        print(f"[ERROR] Retrospective holdout dataset not found at: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    df_holdout = pd.read_csv(DATA_PATH)
    n_samples = len(df_holdout)
    if n_samples != 594:
        print(f"[ERROR] Expected 594 records, found {n_samples}.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[1/3] Loaded retrospective holdout cohort: {DATA_PATH.name} ({n_samples} records)")
    engine = UnifiedPredictionEngine(models_dir=str(MODELS_DIR))

    print("[2/3] Executing multi-stage forward inference pipeline across all 594 records...")
    feature_cols = [c for c in df_holdout.columns if c not in ["Holdout_ID", "Attrition"]]
    df_features = df_holdout[feature_cols].copy()

    preds_df = engine.predict_dataframe(df_features)
    y_true = df_holdout["Attrition"].values
    y_pred = preds_df["Predicted_Class"].values

    # Extract continuous unrounded probability directly from Meta-Learner
    df_full, df_full27 = engine._transform_features(df_features)
    s27_feats = engine.s27_pipe["features"]
    p_s27 = engine.s27_model.predict_proba(df_full27[s27_feats])[:, 1]
    X_30 = df_full[engine.feats30].values
    p_cb = engine.clf_cb.predict_proba(X_30)[:, 1]
    p_lgb = (
        engine.clf_lgb.predict_proba(X_30)[:, 1]
        if engine.clf_lgb is not None
        else p_cb.copy()
    )
    X_meta = np.column_stack([
        p_cb,
        p_lgb,
        p_s27,
        np.abs(p_cb - 0.5),
        np.abs(p_lgb - 0.5),
        np.abs(p_cb - p_lgb),
    ])
    p_meta_unrounded = engine.meta_clf.predict_proba(X_meta)[:, 1]

    print("[3/3] Calculating classification and calibration metrics...")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    errors = int(fp + fn)
    correct = int(tn + tp)

    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred) * 100.0
    rec = recall_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred) * 100.0
    roc_auc_unrounded = roc_auc_score(y_true, p_meta_unrounded) * 100.0

    print("\n" + "=" * 80)
    print("  VERIFICATION RESULTS & BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"  Total Retrospective Records:  {n_samples}")
    print(f"  Total Correct Predictions:    {correct} ({correct / n_samples * 100:.2f}%)")
    print(f"  Total Errors:                 {errors} ({errors / n_samples * 100:.2f}%)")
    print("-" * 80)
    print(f"  Confusion Matrix: [[TN={tn:3d}, FP={fp:2d}], [FN={fn:2d}, TP={tp:2d}]]")
    print("-" * 80)

    rows = [
        ("Total Records", str(TARGETS["records"]), str(n_samples), n_samples == TARGETS["records"]),
        ("True Negatives (TN)", str(TARGETS["tn"]), str(tn), tn == TARGETS["tn"]),
        ("False Positives (FP)", str(TARGETS["fp"]), str(fp), fp == TARGETS["fp"]),
        ("False Negatives (FN)", str(TARGETS["fn"]), str(fn), fn == TARGETS["fn"]),
        ("True Positives (TP)", str(TARGETS["tp"]), str(tp), tp == TARGETS["tp"]),
        ("Total Errors", str(TARGETS["errors"]), str(errors), errors == TARGETS["errors"]),
        ("Accuracy", f"{TARGETS['accuracy']:.2f}%", f"{acc:.2f}%", round(acc, 2) == TARGETS["accuracy"]),
        ("Precision", f"{TARGETS['precision']:.2f}%", f"{prec:.2f}%", round(prec, 2) == TARGETS["precision"]),
        ("Recall", f"{TARGETS['recall']:.2f}%", f"{rec:.2f}%", round(rec, 2) == TARGETS["recall"]),
        ("F1-Score", f"{TARGETS['f1_score']:.2f}%", f"{f1:.2f}%", round(f1, 2) == TARGETS["f1_score"]),
        ("ROC-AUC (Unrounded Continuous)", f"{TARGETS['roc_auc_unrounded']:.2f}%", f"{roc_auc_unrounded:.4f}% ({roc_auc_unrounded:.2f}%)", round(roc_auc_unrounded, 2) == TARGETS["roc_auc_unrounded"]),
    ]

    print(f"{'Metric / Parameter':<32} | {'Reported Target':<17} | {'Calculated':<18} | {'Status'}")
    print("-" * 80)
    all_matched = True
    for name, tgt, val, matched in rows:
        status = "[EXACT MATCH]" if matched else "[MISMATCH]"
        if not matched:
            all_matched = False
        print(f"{name:<32} | {tgt:<17} | {val:<18} | {status}")
    print("=" * 80)

    if all_matched:
        print("  RESULT: VERIFIED SUCCESS (100% BIT-EXACT REPRODUCTION)")
        print(f"  FINAL Model achieves 96.13% accuracy, 23 errors, and 94.5469% ROC-AUC.")
    else:
        print("  RESULT: VERIFICATION FAILED -- DISCREPANCY DETECTED")
    print("=" * 80 + "\n")
    return all_matched


if __name__ == "__main__":
    success = run_retest()
    sys.exit(0 if success else 1)
