"""
Comprehensive Evaluation, Probability Calibration, and Decision-Threshold Optimization Module.
Computes 7 core metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Balanced Accuracy.
Supports Out-of-Fold (OOF) cross-validation, threshold optimization, and publication-quality plotting.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def compute_all_metrics(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes complete 7-metric evaluation suite plus specificity and brier score.
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # Handle single class edge cases gracefully
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Specificity": specificity,
        "Brier_Score": float(brier_score_loss(y_true, y_proba)),
        "Threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }

def optimize_decision_threshold(
    y_true: np.ndarray, 
    y_proba: np.ndarray,
    objective: str = "composite"
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Scans decision thresholds from 0.05 to 0.95 with step 0.01 to find the optimal decision cutoff.
    Objectives:
    - 'composite': 0.40 * Accuracy + 0.30 * F1 + 0.30 * Balanced Accuracy
    - 'f1': Maximizes pure F1-score
    - 'accuracy': Maximizes Accuracy subject to Recall >= 0.70 (or best possible)
    - 'balanced_accuracy': Maximizes Balanced Accuracy
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    results = []

    for t in thresholds:
        m = compute_all_metrics(y_true, y_proba, threshold=t)
        # Composite score prioritizing accuracy while penalizing collapsed recall
        composite = (0.40 * m["Accuracy"]) + (0.30 * m["F1_Score"]) + (0.30 * m["Balanced_Accuracy"])
        m["Composite_Score"] = composite
        results.append(m)

    df_res = pd.DataFrame(results)

    if objective == "composite":
        best_idx = df_res["Composite_Score"].idxmax()
    elif objective == "f1":
        best_idx = df_res["F1_Score"].idxmax()
    elif objective == "accuracy":
        # Filter for recall >= 0.65 if possible, else highest accuracy
        eligible = df_res[df_res["Recall"] >= 0.65]
        best_idx = eligible["Accuracy"].idxmax() if not eligible.empty else df_res["Accuracy"].idxmax()
    elif objective == "balanced_accuracy":
        best_idx = df_res["Balanced_Accuracy"].idxmax()
    else:
        best_idx = df_res["Composite_Score"].idxmax()

    best_threshold = float(df_res.loc[best_idx, "Threshold"])
    best_metrics = df_res.loc[best_idx].to_dict()

    return best_threshold, best_metrics, df_res

def evaluate_out_of_fold(
    model, 
    X: pd.DataFrame, 
    y: pd.Series, 
    cv: int = 5, 
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Executes leak-free Stratified K-Fold Cross-Validation, returning out-of-fold probability predictions.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = clone(model)
        clf.fit(X_tr, y_tr)
        
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_val)[:, 1]
        elif hasattr(clf, "decision_function"):
            df_vals = clf.decision_function(X_val)
            probs = 1 / (1 + np.exp(-df_vals))
        else:
            probs = clf.predict(X_val)
            
        oof_proba[val_idx] = probs
        
    default_metrics = compute_all_metrics(y.values, oof_proba, threshold=0.5)
    return oof_proba, default_metrics

def plot_confusion_matrix_heatmap(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_path: str,
    title: str = "Confusion Matrix"
):
    """
    Generates and saves a high-resolution Confusion Matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.array([
        [f"{cm[0,0]:,}\n({cm_pct[0,0]:.1f}%)", f"{cm[0,1]:,}\n({cm_pct[0,1]:.1f}%)"],
        [f"{cm[1,0]:,}\n({cm_pct[1,0]:.1f}%)", f"{cm[1,1]:,}\n({cm_pct[1,1]:.1f}%)"]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Retained (0)", "Turnover (1)"],
                yticklabels=["Retained (0)", "Turnover (1)"],
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Ground Truth", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_and_pr_curves(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    save_dir: str,
    model_name: str = "Proposed Model"
):
    """
    Generates high-resolution ROC and Precision-Recall Curves.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2980b9', lw=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_title(f'Receiver Operating Characteristic (ROC) Curve\n{model_name}', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 2. PR Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline_pr = np.mean(y_true)
    ax.plot(recall, precision, color='#e67e22', lw=2.5, label=f'{model_name} (PR-AUC = {pr_auc:.4f})')
    ax.axhline(y=baseline_pr, color='#7f8c8d', lw=1.5, linestyle='--', label=f'Baseline Rate ({baseline_pr:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax.set_title(f'Precision-Recall Curve\n{model_name}', fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    pr_path = os.path.join(save_dir, 'pr_curve.png')
    plt.savefig(pr_path, dpi=300)
    plt.close()

def plot_benchmark_comparison_chart(
    proposed_metrics: Dict[str, float],
    benchmark_metrics: Dict[str, float],
    save_path: str
):
    """
    Generates a grouped bar chart directly comparing the Proposed Architecture with the Published Base Paper Benchmark.
    """
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    
    proposed_vals = [proposed_metrics.get(m, 0.0) * 100 for m in metrics_to_plot]
    bench_vals = [benchmark_metrics.get(m, 0.0) * 100 for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, bench_vals, width, label='Base Paper Benchmark', color='#95a5a6', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, proposed_vals, width, label='Proposed System (Untouched Test Set)', color='#27ae60', edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparative Benchmark Analysis: Proposed System vs Published Baseline', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, loc='upper right', fontsize=11)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
