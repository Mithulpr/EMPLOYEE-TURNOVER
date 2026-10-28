#!/usr/bin/env python3
"""
Publication Figure Generator: Chapter 5 Experiment-Wise Comparison
==================================================================
Generates an uncrowded, publication-grade grouped bar chart comparing:
- Experiment 1: 30-Feature Configuration (5-Fold CV, N = 2,969)
- Experiment 2: 35-Feature Configuration (5-Fold CV, N = 2,969)
- Experiment 3: 40-Feature Configuration (5-Fold CV, N = 2,969)
- Experiment 4: GA–LightGBM Baseline Reproduction (10-Fold CV, N = 2,969)
- Final Hybrid Model: Multi-Stage Stacking Architecture (Holdout Cohort, N = 594)
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mtick
import numpy as np

PROJECT_ROOT = Path("e:/R&D final")
OUTPUT_DIRS = [
    PROJECT_ROOT / "research_documents" / "reports" / "figures",
    PROJECT_ROOT / "final_employee_turnover_system" / "FINAL" / "figures" / "performance",
    Path("C:/Users/mithu/.gemini/antigravity-ide/brain/81207504-f1b2-4481-b24e-8f9b3c0dbb79")
]

for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "acc_cv": "#1E40AF",          # Deep Blue (Cross-Validation Accuracy)
    "f1_cv": "#D97706",           # Amber Orange (Cross-Validation F1-score)
    "acc_final": "#1D4ED8",       # Deep Royal Blue (Final Model Accuracy)
    "f1_final": "#B45309",        # Deep Amber (Final Model F1-score)
    "text_dark": "#0F172A",       # Slate 900
    "text_muted": "#475569",      # Slate 600
    "border_light": "#CBD5E1",    # Slate 300
    "grid": "#E2E8F0",            # Grid Line
    "divider": "#64748B"          # Protocol divider
}

def generate_figure():
    experiments = [
        {
            "id": "Experiment 1",
            "desc": "30 Features",
            "protocol": "5-Fold CV",
            "accuracy": 89.32,
            "f1": 67.31,
            "is_holdout": False
        },
        {
            "id": "Experiment 2",
            "desc": "35 Features",
            "protocol": "5-Fold CV",
            "accuracy": 89.26,
            "f1": 64.24,
            "is_holdout": False
        },
        {
            "id": "Experiment 3",
            "desc": "40 Features",
            "protocol": "5-Fold CV",
            "accuracy": 87.98,
            "f1": 61.33,
            "is_holdout": False
        },
        {
            "id": "Experiment 4",
            "desc": "GA–LightGBM",
            "protocol": "10-Fold CV",
            "accuracy": 90.97,
            "f1": 65.82,
            "is_holdout": False
        },
        {
            "id": "Final Hybrid Model",
            "desc": "Multi-Stage Stacking",
            "protocol": "Holdout Cohort (N=594)*",
            "accuracy": 96.13,
            "f1": 87.96,
            "is_holdout": True
        }
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    n_groups = len(experiments)
    indices = np.arange(n_groups)
    bar_width = 0.32
    offset = bar_width / 2.0

    # Subtle shaded background for holdout protocol region
    ax.axvspan(3.5, 4.6, facecolor="#F8FAFC", edgecolor="none", zorder=0)

    # Plot grouped bars
    for i, exp in enumerate(experiments):
        x_acc = indices[i] - offset
        x_f1 = indices[i] + offset
        
        acc_val = exp["accuracy"]
        f1_val = exp["f1"]

        if exp["is_holdout"]:
            ax.bar(x_acc, acc_val, bar_width, color=PALETTE["acc_final"],
                   edgecolor=PALETTE["text_dark"], linewidth=1.2, hatch="//", zorder=3)
            ax.bar(x_f1, f1_val, bar_width, color=PALETTE["f1_final"],
                   edgecolor=PALETTE["text_dark"], linewidth=1.2, hatch="\\\\", zorder=3)
        else:
            ax.bar(x_acc, acc_val, bar_width, color=PALETTE["acc_cv"],
                   edgecolor="#0F172A", linewidth=1.0, zorder=3)
            ax.bar(x_f1, f1_val, bar_width, color=PALETTE["f1_cv"],
                   edgecolor="#0F172A", linewidth=1.0, zorder=3)

        # Exact percentage text labels on top of bars
        ax.text(x_acc, acc_val + 1.2, f"{acc_val:.2f}%", ha="center", va="bottom",
                fontsize=9.0, fontweight="bold", color=PALETTE["text_dark"], zorder=4)
        ax.text(x_f1, f1_val + 1.2, f"{f1_val:.2f}%", ha="center", va="bottom",
                fontsize=9.0, fontweight="bold", color=PALETTE["text_dark"], zorder=4)

    # Protocol demarcation line
    ax.axvline(x=3.5, color=PALETTE["divider"], linestyle="--", linewidth=1.5, zorder=2)

    # Protocol Zone Banners across the top of the plot
    # Zone 1: Feature Representation Experiments
    ax.plot([-0.25, 2.25], [103.5, 103.5], color=PALETTE["text_muted"], lw=1.2, zorder=3)
    ax.plot([-0.25, -0.25], [102.3, 103.5], color=PALETTE["text_muted"], lw=1.2, zorder=3)
    ax.plot([2.25, 2.25], [102.3, 103.5], color=PALETTE["text_muted"], lw=1.2, zorder=3)
    ax.text(1.0, 105.0, "Feature-Representation Ablation\n[5-Fold CV, N = 2,969]",
            ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=PALETTE["text_muted"])

    # Zone 2: Baseline Reproduction
    ax.plot([2.75, 3.25], [103.5, 103.5], color=PALETTE["text_muted"], lw=1.2, zorder=3)
    ax.plot([2.75, 2.75], [102.3, 103.5], color=PALETTE["text_muted"], lw=1.2, zorder=3)
    ax.plot([3.25, 3.25], [102.3, 103.5], color=PALETTE["text_muted"], lw=1.2, zorder=3)
    ax.text(3.0, 105.0, "Literature Baseline\n[10-Fold CV, N = 2,969]",
            ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=PALETTE["text_muted"])

    # Zone 3: Proposed Final Model
    ax.plot([3.75, 4.25], [103.5, 103.5], color="#B45309", lw=1.4, zorder=3)
    ax.plot([3.75, 3.75], [102.3, 103.5], color="#B45309", lw=1.4, zorder=3)
    ax.plot([4.25, 4.25], [102.3, 103.5], color="#B45309", lw=1.4, zorder=3)
    ax.text(4.0, 105.0, "Proposed Final Model\n[Holdout Cohort, N = 594]*",
            ha="center", va="bottom", fontsize=8.6, fontweight="bold", color="#B45309")

    # X-Axis Labels: Clean, multi-line formatting
    x_tick_labels = [
        f"{exp['id']}\n{exp['desc']}\n({exp['protocol']})"
        for exp in experiments
    ]
    ax.set_xticks(indices)
    ax.set_xticklabels(x_tick_labels, fontsize=9.2, fontweight="bold", color=PALETTE["text_dark"], y=-0.01)

    # Y-Axis Formatting
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylabel("Evaluation Metric Score (%)", fontsize=11.0, fontweight="bold",
                  color=PALETTE["text_dark"], labelpad=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_locator(mtick.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(5))
    ax.grid(axis="y", linestyle=":", color=PALETTE["grid"], linewidth=0.8, alpha=0.9, zorder=1)

    # Clean axes spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(PALETTE["border_light"])
    ax.spines["bottom"].set_color(PALETTE["border_light"])

    # External Legend at top center, completely clear of data and annotations
    h_acc_cv = patches.Patch(facecolor=PALETTE["acc_cv"], edgecolor="#0F172A", label="Accuracy (Cross-Validation)")
    h_f1_cv = patches.Patch(facecolor=PALETTE["f1_cv"], edgecolor="#0F172A", label="F1-Score (Cross-Validation)")
    h_acc_final = patches.Patch(facecolor=PALETTE["acc_final"], edgecolor=PALETTE["text_dark"], hatch="//",
                                label="Accuracy (Holdout Cohort*)")
    h_f1_final = patches.Patch(facecolor=PALETTE["f1_final"], edgecolor=PALETTE["text_dark"], hatch="\\\\",
                               label="F1-Score (Holdout Cohort*)")

    leg = ax.legend(handles=[h_acc_cv, h_f1_cv, h_acc_final, h_f1_final],
                    loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=True, fontsize=8.5,
                    facecolor="white", edgecolor=PALETTE["border_light"], framealpha=0.98)
    leg.get_frame().set_boxstyle("round,pad=0.35")

    # Title above the legend
    fig.suptitle("Experiment-Wise Comparison: Performance Across Development Iterations and Final Model",
                 fontsize=12.5, fontweight="bold", color=PALETTE["text_dark"], y=0.98, ha="center")

    # Methodological footnote at bottom
    fig.text(0.08, 0.02,
             "*Methodological Note: Evaluation protocols differ across stages. Experiments 1–3 use 5-fold cross-validation and Experiment 4 uses 10-fold cross-validation\n"
             "across the full 2,969-profile dataset. The Final Hybrid Model is evaluated on a locked 594-record development holdout cohort consulted during iterative\n"
             "development. Direct cross-protocol equivalence is not assumed; values are displayed together to document empirical development progression.",
             fontsize=7.8, color=PALETTE["text_muted"], style="italic")

    plt.subplots_adjust(top=0.82, bottom=0.17, left=0.08, right=0.96)

    saved_paths = []
    for out_dir in OUTPUT_DIRS:
        p = out_dir / "chapter5_experiment_comparison.png"
        fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
        saved_paths.append(p)
        print(f"[SUCCESS] Saved figure to: {p}")

    plt.close(fig)
    return saved_paths

if __name__ == "__main__":
    generate_figure()
