"""
Retrospective Legacy Holdout Evaluation Module
==============================================
Evaluates the multi-stage hybrid pipeline on the 594-record cohort.
Strictly calculates performance directly from predictions:
Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix, and Errors.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from typing import Dict, Any, Tuple, Optional

from src.pipeline import MultiStageHybridPipeline
from src.data_loader import load_holdout_dataset, load_config
from src.risk import compute_risk_distribution


def evaluate_holdout(file_path: Optional[str] = None, tau: float = 0.361) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Runs complete holdout evaluation and computes all verification metrics.
    """
    df_holdout = load_holdout_dataset(file_path)
    if "Attrition" not in df_holdout.columns:
        raise ValueError("Holdout dataset must contain 'Attrition' ground-truth column for evaluation.")

    pipeline = MultiStageHybridPipeline(threshold=tau)
    df_preds = pipeline.predict(df_holdout)

    y_true = df_holdout["Attrition"].values
    y_pred = df_preds["Predicted_Class"].values
    y_scores = df_preds["Turnover_Score"].values

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    total_errors = int(fp + fn)
    total_correct = int(tp + tn)

    metrics = {
        "evaluation_cohort": "Retrospective Legacy Holdout (N = 594)",
        "operating_threshold": tau,
        "total_records": len(y_true),
        "accuracy_pct": round(acc * 100.0, 2),
        "precision_pct": round(prec * 100.0, 2),
        "recall_pct": round(rec * 100.0, 2),
        "f1_score_pct": round(f1 * 100.0, 2),
        "roc_auc_pct": round(roc_auc * 100.0, 2),
        "pr_auc_pct": round(pr_auc * 100.0, 2),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "total_correct": total_correct,
        "total_errors": total_errors
    }

    return metrics, df_preds
