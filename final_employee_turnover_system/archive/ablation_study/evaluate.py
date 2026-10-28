"""
Research Gap Extension: Standalone Retrospective Evaluation Module
==================================================================
Evaluates the frozen experimental pipeline from research_gap_extension/models/
on the 594-record retrospective holdout partition.

Mandatory Disclosures:
- Retrospective Holdout Evaluation: Examined in prior research, not a virgin external test set.
- Pure forward inference: Zero training, fitting, threshold tuning, or calibration during evaluation.
"""

import os
import sys
import json
import joblib
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.dirname(CURRENT_DIR)
BASE_DIR = os.path.dirname(SYSTEM_DIR)

for p in [CURRENT_DIR, SYSTEM_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_and_validate_raw_data, clean_and_harmonize_dataset
from sklearn.model_selection import train_test_split


def run_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("RESEARCH GAP EXTENSION — FROZEN EXPERIMENTAL PIPELINE EVALUATION")
    print("=" * 80)

    models_dir = os.path.join(CURRENT_DIR, "models")
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found at {models_dir}. Run train.py first.")

    # 1. Load pipeline artifacts
    fe = joblib.load(os.path.join(models_dir, "hr_feature_engineer.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    cb_model = joblib.load(os.path.join(models_dir, "catboost_base.joblib"))
    lgb_model = joblib.load(os.path.join(models_dir, "lightgbm_base.joblib"))
    s27_model = joblib.load(os.path.join(models_dir, "anchor_catboost.joblib"))
    meta_lr = joblib.load(os.path.join(models_dir, "meta_logistic_regression.joblib"))

    spec_path = os.path.join(models_dir, "boundary_specialist.joblib")
    spec_cb = joblib.load(spec_path) if os.path.exists(spec_path) else None

    with open(os.path.join(models_dir, "pipeline_schema.json"), "r", encoding="utf-8") as f:
        schema = json.load(f)
    feature_names = schema["features"]
    tau = schema["decision_threshold"]

    # 2. Prepare test dataset (594 records, seed 42)
    df_raw, _ = load_and_validate_raw_data()
    df_clean = clean_and_harmonize_dataset(df_raw)
    y_all = df_clean["Attrition"].values
    X_raw_all = df_clean.drop(columns=["Attrition"])

    _, X_te_raw, _, y_te = train_test_split(
        X_raw_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )

    t0 = time.time()
    # 3. Pure forward inference
    X_te_fe = fe.transform(X_te_raw)
    X_te_mat = scaler.transform(X_te_fe[feature_names].fillna(0.0))

    p_cb = cb_model.predict_proba(X_te_mat)[:, 1]
    p_lgb = lgb_model.predict_proba(X_te_mat)[:, 1]
    p_s27 = s27_model.predict_proba(X_te_mat)[:, 1]

    X_meta = np.column_stack([
        p_cb, p_lgb, p_s27,
        (p_cb + p_lgb) / 2.0,
        np.maximum(p_cb, p_lgb),
        np.minimum(p_cb, p_lgb)
    ])
    p_meta = meta_lr.predict_proba(X_meta)[:, 1]
    pred_final = (p_meta >= tau).astype(int)

    # Specialist refinement
    if spec_cb is not None:
        bnd_mask = (p_meta >= 0.15) & (p_meta <= tau)
        if bnd_mask.sum() > 0:
            p_spec = spec_cb.predict_proba(X_te_mat[bnd_mask])[:, 1]
            override_indices = np.where(bnd_mask)[0][p_spec >= 0.50]
            pred_final[override_indices] = 1

    latency_ms = (time.time() - t0) * 1000.0

    # 4. Metrics computation
    cm = confusion_matrix(y_te, pred_final)
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    acc = float(accuracy_score(y_te, pred_final))
    prec = float(precision_score(y_te, pred_final, zero_division=0))
    rec = float(recall_score(y_te, pred_final, zero_division=0))
    f1 = float(f1_score(y_te, pred_final, zero_division=0))
    roc_auc = float(roc_auc_score(y_te, p_meta))
    pr_auc = float(average_precision_score(y_te, p_meta))
    bal_acc = float(balanced_accuracy_score(y_te, pred_final))
    total_errors = int(fp + fn)

    cls_report = classification_report(y_te, pred_final, output_dict=True, target_names=["No Turnover", "Turnover"])

    results = {
        "architecture": "Research Gap Extension — Domain-Engineered Hybrid Pipeline",
        "evaluation_protocol": "retrospective holdout evaluation",
        "evaluated_records": len(y_te),
        "test_distribution": {"No_Turnover": int((y_te == 0).sum()), "Turnover": int((y_te == 1).sum())},
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(roc_auc * 100, 2),
        "pr_auc": round(pr_auc * 100, 2),
        "specificity": round(spec * 100, 2),
        "balanced_accuracy": round(bal_acc * 100, 2),
        "total_errors": total_errors,
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "classification_report": cls_report,
        "decision_threshold": tau,
        "inference_latency_ms": round(latency_ms, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print("\n" + "-" * 60)
    print("EVALUATION METRIC BREAKDOWN (N = 594 Test Records)")
    print("-" * 60)
    print(f"Accuracy:          {results['accuracy']}%")
    print(f"Precision:         {results['precision']}%")
    print(f"Recall:            {results['recall']}%")
    print(f"F1-Score:          {results['f1_score']}%")
    print(f"ROC-AUC:           {results['roc_auc']}%")
    print(f"PR-AUC:            {results['pr_auc']}%")
    print(f"Specificity:       {results['specificity']}%")
    print(f"Balanced Accuracy: {results['balanced_accuracy']}%")
    print(f"Total Errors:      {results['total_errors']} (TN={tn}, FP={fp}, FN={fn}, TP={tp})")
    print(f"Inference Latency: {results['inference_latency_ms']} ms")
    print("-" * 60)

    # Save outputs
    eval_json_path = os.path.join(CURRENT_DIR, "evaluation_results.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation report saved to {eval_json_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
