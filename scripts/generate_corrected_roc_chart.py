#!/usr/bin/env python3
"""
Corrected Publication Figure Generator: ROC & PR Curves (Unrounded 94.55% ROC-AUC)
==================================================================================
Non-destructive standalone script to generate publication-grade ROC & PR curves
using the unrounded continuous meta-probabilities (P_Step28_Meta) from the official
Step 32 holdout prediction artifact.

Outputs:
    final_employee_turnover_system/reports/figures/fig03_roc_and_pr_curves_corrected_9455.png
Does NOT modify or overwrite any existing figure or project file.
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = PROJECT_ROOT / "artifacts" / "track1_step32_aggressive" / "baseline_step31_champion" / "step30_baseline_predictions" / "final_holdout_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "final_employee_turnover_system" / "reports" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "fig03_roc_and_pr_curves_corrected_9455.png"

PALETTE = {
    "primary": "#2563EB",       # Royal Blue
    "primary_dark": "#1D4ED8",  # Deep Navy Blue
    "secondary": "#4F46E5",     # Indigo
    "danger": "#DC2626",        # Crimson Red
    "slate_dark": "#0F172A",    # Dark Slate Navy
    "slate_med": "#475569",     # Medium Slate
    "grid": "#E2E8F0"           # Light Grid / Border
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = PALETTE["grid"]
plt.rcParams["axes.linewidth"] = 0.8


def main():
    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found at: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    y_true = df["True_Label"].values.astype(int)
    y_scores = df["P_Step28_Meta"].values.astype(float)

    # Compute unrounded exact metrics
    exact_auc = roc_auc_score(y_true, y_scores)
    exact_pr_auc = average_precision_score(y_true, y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.4), dpi=300)
    fig.patch.set_facecolor("white")

    # 1. ROC Curve
    ax1.plot(fpr, tpr, color=PALETTE["primary"], lw=2.5,
             label=f"ROC Curve (AUC = {exact_auc*100:.2f}%)")
    ax1.plot([0, 1], [0, 1], color=PALETTE["slate_med"], lw=1.2, linestyle="--",
             label="Chance Baseline (AUC = 50.00%)")

    # Operating point (Step 32: TN=487, FP=11, FN=12, TP=84 at tau=0.3610)
    op_fpr = 11 / (487 + 11)
    op_tpr = 84 / (84 + 12)
    ax1.scatter([op_fpr], [op_tpr], color=PALETTE["danger"], s=85, zorder=5,
                label=f"Operating Point (τ = 0.3610)\nFPR = {op_fpr*100:.2f}%, Recall = {op_tpr*100:.2f}%")

    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5, fontweight="bold", color=PALETTE["slate_dark"])
    ax1.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10.5, fontweight="bold", color=PALETTE["slate_dark"])
    ax1.set_title("(a) Receiver Operating Characteristic (ROC)", fontsize=11.5, fontweight="bold", color=PALETTE["slate_dark"], pad=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=9.0)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Precision-Recall Curve
    no_skill = np.sum(y_true) / len(y_true)
    ax2.plot(rec, prec, color=PALETTE["secondary"], lw=2.5,
             label=f"PR Curve (PR-AUC = {exact_pr_auc*100:.2f}%)")
    ax2.plot([0, 1], [no_skill, no_skill], color=PALETTE["slate_med"], lw=1.2, linestyle="--",
             label=f"Imbalance Baseline ({no_skill*100:.2f}%)")

    op_prec = 84 / (84 + 11)
    op_rec = 84 / (84 + 12)
    ax2.scatter([op_rec], [op_prec], color=PALETTE["danger"], s=85, zorder=5,
                label=f"Operating Point (τ = 0.3610)\nRecall = {op_rec*100:.2f}%, Precision = {op_prec*100:.2f}%")

    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10.5, fontweight="bold", color=PALETTE["slate_dark"])
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=10.5, fontweight="bold", color=PALETTE["slate_dark"])
    ax2.set_title("(b) Precision-Recall Curve (PR)", fontsize=11.5, fontweight="bold", color=PALETTE["slate_dark"], pad=10)
    ax2.legend(loc="lower left", frameon=True, fontsize=9.0)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Step 32 Official Champion — Discriminative Performance (N = 594, Leavers = 96)",
                 fontsize=13, fontweight="bold", y=0.98, color=PALETTE["slate_dark"])

    plt.subplots_adjust(top=0.88, bottom=0.14, wspace=0.25)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"[SUCCESS] Corrected ROC/PR chart generated successfully:")
    print(f"  Output path: {OUTPUT_PNG}")
    print(f"  Exact ROC-AUC: {exact_auc*100:.4f}% -> Displayed: {exact_auc*100:.2f}%")
    print(f"  Exact PR-AUC:  {exact_pr_auc*100:.4f}% -> Displayed: {exact_pr_auc*100:.2f}%")


if __name__ == "__main__":
    main()
