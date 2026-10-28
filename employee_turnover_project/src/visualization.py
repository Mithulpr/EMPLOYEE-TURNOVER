"""
Publication-Quality Graphics & Visualization Engine
===================================================
Renders 300 DPI publication-grade figures:
- Fig 01: Multi-Metric Performance Overview
- Fig 02: High-Resolution Confusion Matrix
- Fig 03: Dual ROC and Precision-Recall Curves
- Fig 04: Operational 4-Tier Risk Distribution
- Fig 05: Historical Developmental Champion Progression
- Fig 06: Nested Cross-Validation Feature Ablation
- Fig 07: Global SHAP Feature Importance Ranking (15 Features)
- Fig 08: Individual Employee SHAP Decomposition (EMP_HD_001)
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

PALETTE = {
    "primary": "#1E40AF",       # Royal Blue
    "primary_dark": "#0F172A",  # Slate Dark
    "accent": "#0D9488",       # Teal
    "danger": "#DC2626",       # Crimson Red
    "warning": "#D97706",      # Amber
    "success": "#059669",      # Green
    "slate_med": "#64748B",    # Slate Muted
    "grid": "#E2E8F0"          # Light Slate
}


def plot_all_figures(output_dir: Optional[str] = None) -> Dict[str, str]:
    """Generates all 8 research figures and saves to figures directory."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    paths["fig01"] = plot_performance_metrics(os.path.join(output_dir, "fig01_performance_metrics.png"))
    paths["fig02"] = plot_confusion_matrix(os.path.join(output_dir, "fig02_confusion_matrix.png"))
    paths["fig03"] = plot_roc_and_pr_curves(os.path.join(output_dir, "fig03_roc_and_pr_curves.png"))
    paths["fig04"] = plot_risk_tier_distribution(os.path.join(output_dir, "fig04_risk_tier_distribution.png"))
    paths["fig05"] = plot_champion_progression(os.path.join(output_dir, "fig05_champion_progression.png"))
    paths["fig06"] = plot_feature_ablation(os.path.join(output_dir, "fig06_feature_ablation.png"))
    paths["fig07"] = plot_global_shap(os.path.join(output_dir, "fig07_global_shap.png"))
    paths["fig08"] = plot_local_shap_emp001(os.path.join(output_dir, "fig08_local_shap_EMP_HD_001.png"))
    return paths


def plot_performance_metrics(save_path: str) -> str:
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    values = [96.13, 88.42, 87.50, 87.96, 94.55, 86.89]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(metrics, values, color=PALETTE["primary"], width=0.55, edgecolor=PALETTE["primary_dark"], linewidth=1.0)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Metric Score (%)", fontsize=10.5, fontweight="bold", color=PALETTE["primary_dark"])
    ax.set_title("Retrospective Performance Benchmark (Holdout N = 594, tau = 0.361)", fontsize=12, fontweight="bold", pad=14)

    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.8, f"{val:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.grid(axis="y", linestyle="--", alpha=0.5, color=PALETTE["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion_matrix(save_path: str) -> str:
    cm = np.array([[487, 11], [12, 84]])
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
    fig.patch.set_facecolor("white")

    cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Stayer", "Predicted Leaver"], fontsize=10, fontweight="bold")
    ax.set_yticklabels(["Actual Stayer", "Actual Leaver"], fontsize=10, fontweight="bold")
    ax.set_title("Holdout Confusion Matrix (N = 594, Total Errors = 23)", fontsize=11.5, fontweight="bold", pad=12)

    labels = [["TN = 487\n(82.0%)", "FP = 11\n(1.9%)"], ["FN = 12\n(2.0%)", "TP = 84\n(14.1%)"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > 200 else PALETTE["primary_dark"]
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_roc_and_pr_curves(save_path: str) -> str:
    from sklearn.metrics import roc_curve, precision_recall_curve
    from src.data_loader import load_cached_evaluation
    cache = load_cached_evaluation()
    records = cache.get("records", [])

    y_true = np.array([r["actual_attrition"] for r in records])
    y_scores = np.array([r["risk_score"] for r in records])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)
    fig.patch.set_facecolor("white")

    # ROC
    ax1.plot(fpr, tpr, color=PALETTE["primary"], lw=2.2, label="Hybrid Ensemble (AUC = 0.9455)")
    ax1.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["slate_med"], lw=1.2)
    ax1.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
    ax1.set_title("ROC Curve (N = 594)", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])

    # PR
    ax2.plot(rec, prec, color=PALETTE["accent"], lw=2.2, label="Hybrid Ensemble (PR-AUC = 0.8689)")
    baseline = np.sum(y_true) / len(y_true)
    ax2.axhline(baseline, linestyle="--", color=PALETTE["danger"], lw=1.2, label=f"No-Skill Base ({baseline*100:.1f}%)")
    ax2.set_xlabel("Recall", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax2.set_title("Precision-Recall Curve (N = 594)", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_risk_tier_distribution(save_path: str) -> str:
    tiers = ["Low", "Moderate", "High", "Critical"]
    counts = [491, 11, 14, 78]
    pcts = [82.66, 1.85, 2.36, 13.13]
    colors = [PALETTE["success"], PALETTE["warning"], PALETTE["danger"], "#7F1D1D"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(tiers, counts, color=colors, width=0.52, edgecolor=PALETTE["primary_dark"], linewidth=1.0)
    ax.set_ylabel("Employee Count", fontsize=10.5, fontweight="bold")
    ax.set_title("Operational Risk-Tier Distribution (Holdout N = 594)", fontsize=12, fontweight="bold", pad=14)
    ax.set_ylim(0, 540)

    for bar, c, p in zip(bars, counts, pcts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 8, f"{c}\n({p:.2f}%)", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.grid(axis="y", linestyle="--", alpha=0.5, color=PALETTE["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_champion_progression(save_path: str) -> str:
    milestones = ["Champion 4", "Step 28", "Step 30", "Step 31", "Step 32 Final"]
    accs = [94.95, 95.62, 95.79, 95.96, 96.13]
    errors = [30, 26, 25, 24, 23]

    fig, ax1 = plt.subplots(figsize=(9.2, 5.0), dpi=300)
    fig.patch.set_facecolor("white")

    ax1.plot(milestones, accs, marker="o", color=PALETTE["primary"], lw=2.4, markersize=8, label="Accuracy (%)")
    ax1.set_ylabel("Holdout Accuracy (%)", color=PALETTE["primary"], fontsize=10.5, fontweight="bold")
    ax1.set_ylim(94.0, 97.0)

    for i, (m, a) in enumerate(zip(milestones, accs)):
        ax1.text(i, a + 0.15, f"{a:.2f}%", ha="center", va="bottom", fontweight="bold", color=PALETTE["primary"], fontsize=9.2)

    ax2 = ax1.twinx()
    ax2.bar(milestones, errors, alpha=0.25, color=PALETTE["danger"], width=0.38, label="Errors")
    ax2.set_ylabel("Total Misclassifications", color=PALETTE["danger"], fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 45)

    for i, e in enumerate(errors):
        ax2.text(i, e + 1.2, f"{e} err", ha="center", va="bottom", fontweight="bold", color=PALETTE["danger"], fontsize=8.8)

    ax1.set_title("Developmental Champion Progression (Retrospective Development History)", fontsize=11.5, fontweight="bold", pad=14)
    ax1.grid(axis="x", linestyle="--", alpha=0.5, color=PALETTE["grid"])
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_feature_ablation(save_path: str) -> str:
    from src.ablation import load_ablation_results
    df = load_ablation_results()

    arms = ["Arm A\n(30 Features)", "Arm B\n(40 Features)", "Arm C\n(35 Features)"]
    f1_means = df["F1_Mean"].values
    acc_means = df["Accuracy_Mean"].values

    x = np.arange(len(arms))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    fig.patch.set_facecolor("white")

    rects1 = ax.bar(x - width/2, acc_means, width, label="Accuracy (%)", color=PALETTE["primary"], edgecolor=PALETTE["primary_dark"])
    rects2 = ax.bar(x + width/2, f1_means, width, label="F1-Score (%)", color=PALETTE["accent"], edgecolor=PALETTE["primary_dark"])

    ax.set_ylabel("CV Metric Score (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("Nested Cross-Validation Feature Ablation Experiment", fontsize=12, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)

    for r in rects1:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.5, f"{r.get_height():.1f}%", ha="center", fontsize=8.8, fontweight="bold")
    for r in rects2:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.5, f"{r.get_height():.1f}%", ha="center", fontsize=8.8, fontweight="bold")

    ax.grid(axis="y", linestyle="--", alpha=0.5, color=PALETTE["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_global_shap(save_path: str) -> str:
    from src.shap_analysis import get_global_shap_ranking
    df_shap = get_global_shap_ranking()
    df_rev = df_shap.iloc[::-1].reset_index(drop=True)

    labels = df_rev["Feature_Name"].values
    vals = df_rev["Mean_Absolute_SHAP"].values

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(labels, vals, color=PALETTE["primary"], height=0.62, edgecolor=PALETTE["primary_dark"], linewidth=0.8)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Base Ensemble Log-Odds)", fontsize=10, fontweight="bold")
    ax.set_title("Global Feature Importance Ranking (TreeExplainer on Base Ensemble, N = 594)", fontsize=11.5, fontweight="bold", pad=12)

    for bar, v in zip(bars, vals):
        ax.text(v + 0.012, bar.get_y() + bar.get_height()/2, f"{v:.4f}", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color=PALETTE["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_local_shap_emp001(save_path: str) -> str:
    features = [
        "MaritalStatus_Single",
        "PromotionPaceStrain",
        "TotalWorkingYears",
        "WorkLifeDeficit",
        "PromotionLagScore",
        "CompanyLoyaltyRatio",
        "TrainingTimesLastYear",
        "Age"
    ]
    shap_vals = [
        +1.0640,
        +0.8365,
        +0.3969,
        +0.3959,
        +0.3105,
        -0.2547,
        -0.8153,
        -0.8278
    ]

    colors = [PALETTE["danger"] if v > 0 else PALETTE["primary"] for v in shap_vals]

    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, shap_vals, color=colors, height=0.62, edgecolor="#0F172A", linewidth=0.7)
    ax.axvline(0, color=PALETTE["primary_dark"], linewidth=1.0)

    for bar, val in zip(bars, shap_vals):
        offset = 0.03 if val > 0 else -0.03
        ha = "left" if val > 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:+.4f}",
                va="center", ha=ha, fontsize=9, fontweight="bold", color=PALETTE["primary_dark"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("SHAP Attribution (Base Gradient-Boosted Ensemble Margin Impact)", fontsize=10, fontweight="bold")
    ax.set_title("Individual Employee SHAP Explanation: EMP_HD_001\n(Actual: Leaver | Predicted: Leaver | Probability = 85.32% | Risk: Critical)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(-1.25, 1.45)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color=PALETTE["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["danger"], edgecolor="#0F172A", label="Push Factors (Increases Estimated Turnover Risk)"),
        Patch(facecolor=PALETTE["primary"], edgecolor="#0F172A", label="Pull Factors (Retention Anchors)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=8.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path
