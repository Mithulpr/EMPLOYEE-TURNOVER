#!/usr/bin/env python3
"""
Step 32 Official Champion Holdout Evaluation Script
===================================================
A reusable, standalone evaluation tool to calculate and verify classification
metrics for the official Step 32 Champion on the legacy holdout dataset.

This script evaluates saved discrete predictions without retraining or loading
machine learning models.

Usage (from project root or any directory):
    python scripts/evaluate_step32.py
"""

import sys
from pathlib import Path

# Resolve project root dynamically from script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Input prediction artifact path
REL_INPUT_PATH = Path("artifacts") / "track1_step32_aggressive" / "final_candidate" / "step32_candidate_holdout.csv"
INPUT_CSV_PATH = PROJECT_ROOT / REL_INPUT_PATH

# Required columns
COL_TRUE = "True_Label"
COL_PRED = "Pred_Step32_Candidate"

# Official Step 32 reported benchmark targets
BENCHMARK = {
    "Accuracy": 96.13,
    "Precision": 88.42,
    "Recall": 87.50,
    "F1_Score": 87.96,
    "TN": 487,
    "FP": 11,
    "FN": 12,
    "TP": 84,
    "Errors": 23,
    "Records": 594,
    "Reported_ROC_AUC": "94.55% (requires continuous probabilities)"
}


def validate_and_load_data(csv_path: Path):
    """
    Validates file existence, required schema, absence of nulls, and binary values.
    Exits with a descriptive error message on validation failure.
    """
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] Required dependency 'pandas' is not installed.", file=sys.stderr)
        print("Please install it using: pip install pandas", file=sys.stderr)
        sys.exit(1)

    if not csv_path.exists():
        print(f"[ERROR] Step 32 holdout prediction file not found at:\n  {csv_path}", file=sys.stderr)
        print(f"Expected relative location from project root:\n  {REL_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate row count
    if len(df) == 0:
        print(f"[ERROR] Prediction file '{csv_path.name}' is empty (0 records).", file=sys.stderr)
        sys.exit(1)

    # Validate required columns
    missing_cols = [col for col in (COL_TRUE, COL_PRED) if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required column(s) in prediction CSV: {missing_cols}", file=sys.stderr)
        print(f"Available columns: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    # Check for missing/null values
    null_true = df[COL_TRUE].isna().sum()
    null_pred = df[COL_PRED].isna().sum()
    if null_true > 0 or null_pred > 0:
        print(f"[ERROR] Found missing values in predictions: {COL_TRUE}={null_true}, {COL_PRED}={null_pred}", file=sys.stderr)
        sys.exit(1)

    # Validate binary values (0 or 1)
    unique_true = set(df[COL_TRUE].unique())
    unique_pred = set(df[COL_PRED].unique())
    valid_classes = {0, 1}

    if not unique_true.issubset(valid_classes):
        print(f"[ERROR] Invalid values in '{COL_TRUE}': {unique_true}. Must contain only 0 and 1.", file=sys.stderr)
        sys.exit(1)

    if not unique_pred.issubset(valid_classes):
        print(f"[ERROR] Invalid values in '{COL_PRED}': {unique_pred}. Must contain only 0 and 1.", file=sys.stderr)
        sys.exit(1)

    return df[COL_TRUE].values.astype(int), df[COL_PRED].values.astype(int)


def compute_metrics(y_true, y_pred):
    """
    Computes classification metrics using scikit-learn.
    """
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    except ImportError:
        print("[ERROR] Required dependency 'scikit-learn' is not installed.", file=sys.stderr)
        print("Please install it using: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    errors = int(fp + fn)
    records = len(y_true)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Errors": errors,
        "Records": records
    }


def print_evaluation_report(results, input_path: Path):
    """
    Displays a formatted evaluation report and compares with official Step 32 benchmark.
    """
    # Check parity with reported benchmark
    matches = (
        round(results["Accuracy"], 2) == BENCHMARK["Accuracy"] and
        round(results["Precision"], 2) == BENCHMARK["Precision"] and
        round(results["Recall"], 2) == BENCHMARK["Recall"] and
        round(results["F1_Score"], 2) == BENCHMARK["F1_Score"] and
        results["TN"] == BENCHMARK["TN"] and
        results["FP"] == BENCHMARK["FP"] and
        results["FN"] == BENCHMARK["FN"] and
        results["TP"] == BENCHMARK["TP"] and
        results["Errors"] == BENCHMARK["Errors"] and
        results["Records"] == BENCHMARK["Records"]
    )

    sep_double = "=" * 76
    sep_single = "-" * 76

    print(sep_double)
    print("STEP 32 OFFICIAL CHAMPION -- HOLDOUT EVALUATION REPORT")
    print(sep_double)
    print(f"Evaluation Mode    : Saved Prediction Verification (No retraining)")
    print(f"Artifact Evaluated : {input_path}")
    print(f"Records Evaluated  : {results['Records']}")
    print(sep_single)
    print(f"{'Metric':<24} | {'Calculated':<14} | {'Reported Baseline':<18} | {'Status':<10}")
    print(sep_single)

    def status_str(actual, expected, is_pct=True):
        act_val = round(actual, 2) if is_pct else actual
        return "MATCH" if act_val == expected else f"DIFF ({act_val - expected:+.2f})"

    print(f"{'Accuracy':<24} | {results['Accuracy']:>6.2f}%        | {BENCHMARK['Accuracy']:>6.2f}%           | {status_str(results['Accuracy'], BENCHMARK['Accuracy'])}")
    print(f"{'Precision':<24} | {results['Precision']:>6.2f}%        | {BENCHMARK['Precision']:>6.2f}%           | {status_str(results['Precision'], BENCHMARK['Precision'])}")
    print(f"{'Recall':<24} | {results['Recall']:>6.2f}%        | {BENCHMARK['Recall']:>6.2f}%           | {status_str(results['Recall'], BENCHMARK['Recall'])}")
    print(f"{'F1-score':<24} | {results['F1_Score']:>6.2f}%        | {BENCHMARK['F1_Score']:>6.2f}%           | {status_str(results['F1_Score'], BENCHMARK['F1_Score'])}")
    print(f"{'Total Errors':<24} | {results['Errors']:>6d}         | {BENCHMARK['Errors']:>6d}            | {status_str(results['Errors'], BENCHMARK['Errors'], is_pct=False)}")
    print(f"{'Evaluated Records':<24} | {results['Records']:>6d}         | {BENCHMARK['Records']:>6d}            | {status_str(results['Records'], BENCHMARK['Records'], is_pct=False)}")

    print(sep_single)
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"  True Negatives  (TN) : {results['TN']:>4d}  (Reported: {BENCHMARK['TN']}) -> {'MATCH' if results['TN'] == BENCHMARK['TN'] else 'MISMATCH'}")
    print(f"  False Positives (FP) : {results['FP']:>4d}  (Reported: {BENCHMARK['FP']}) -> {'MATCH' if results['FP'] == BENCHMARK['FP'] else 'MISMATCH'}")
    print(f"  False Negatives (FN) : {results['FN']:>4d}  (Reported: {BENCHMARK['FN']}) -> {'MATCH' if results['FN'] == BENCHMARK['FN'] else 'MISMATCH'}")
    print(f"  True Positives  (TP) : {results['TP']:>4d}  (Reported: {BENCHMARK['TP']}) -> {'MATCH' if results['TP'] == BENCHMARK['TP'] else 'MISMATCH'}")

    print(sep_single)
    print("EXPECTED FORMAT SUMMARY:")
    print(f"Accuracy: {results['Accuracy']:.2f}%")
    print(f"Precision: {results['Precision']:.2f}%")
    print(f"Recall: {results['Recall']:.2f}%")
    print(f"F1-score: {results['F1_Score']:.2f}%")
    print(f"TN={results['TN']}, FP={results['FP']}, FN={results['FN']}, TP={results['TP']}")
    print(f"Total errors: {results['Errors']}")
    print(f"Records: {results['Records']}")

    print(sep_single)
    print("NOTE ON ROC-AUC:")
    print("  The evaluation CSV contains discrete binary classifications (0 and 1)")
    print("  without continuous probability scores. In accordance with rigorous ML")
    print("  standards, ROC-AUC is NOT calculated from binary thresholded decisions,")
    print("  as ROC-AUC requires rank-ordered continuous probability estimates.")
    print("  The official Step 32 continuous ROC-AUC benchmark is: 94.55%.")

    print(sep_double)
    if matches:
        print("[VERDICT] EXACT PARITY CONFIRMED: Saved predictions 100% match the official Step 32 Champion benchmark.")
    else:
        print("[VERDICT] DISCREPANCY DETECTED: Computed metrics differ from reported Step 32 baseline.")
    print(sep_double)


def main():
    y_true, y_pred = validate_and_load_data(INPUT_CSV_PATH)
    results = compute_metrics(y_true, y_pred)
    print_evaluation_report(results, INPUT_CSV_PATH)


if __name__ == "__main__":
    main()
