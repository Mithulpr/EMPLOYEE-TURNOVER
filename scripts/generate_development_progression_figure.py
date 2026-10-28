#!/usr/bin/env python3
"""
Publication Figure & Data Generator: Model Development Progression (Table 5.2 Audit)
====================================================================================
Generates:
1. Coordinated two-panel academic figure (300 DPI):
   - Panel (a): Performance Metrics Evolution (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
   - Panel (b): Classification Error Reduction (Zero-based honest axis, 30 -> 23 errors)
2. Authoritative CSV dataset with verified metrics, confusion matrices, and progression deltas.
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("e:/R&D final")
OUTPUT_DIRS = [
    PROJECT_ROOT / "research_documents" / "reports" / "figures",
    PROJECT_ROOT / "final_employee_turnover_system" / "FINAL" / "figures" / "performance",
    Path("C:/Users/mithu/.gemini/antigravity-ide/brain/81207504-f1b2-4481-b24e-8f9b3c0dbb79")
]
CSV_OUTPUT_DIRS = [
    PROJECT_ROOT / "research_documents" / "reports",
    PROJECT_ROOT / "final_employee_turnover_system" / "FINAL" / "results",
    Path("C:/Users/mithu/.gemini/antigravity-ide/brain/81207504-f1b2-4481-b24e-8f9b3c0dbb79")
]

for d in OUTPUT_DIRS + CSV_OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# Curated academic palette
PALETTE = {
    "accuracy": "#1E40AF",     # Royal Navy Blue
    "precision": "#0D9488",    # Teal
    "recall": "#D97706",       # Warm Amber
    "f1": "#7C3AED",           # Purple
    "roc_auc": "#DC2626",      # Crimson Red
    "errors_prev": "#64748B",  # Slate 500 for precursor models
    "errors_final": "#059669", # Emerald Green for final hybrid model
    "text_dark": "#0F172A",    # Dark Slate 900
    "text_muted": "#475569",   # Medium Slate 600
    "border_light": "#CBD5E1", # Slate 300
    "grid": "#E2E8F0"          # Grid Line
}

DATA = [
    {
        "Version": "Experiment 1",
        "Precursor_ID": "CHAMPION 4 (Stage 6)",
        "Architecture": "Tuned Single CatBoost",
        "Accuracy": 94.95,
        "Precision": 87.50,
        "Recall": 80.21,
        "F1-score": 83.70,
        "ROC-AUC": 92.32,
        "Errors": 30,
        "TN": 487,
        "FP": 11,
        "FN": 19,
        "TP": 77
    },
    {
        "Version": "Experiment 2",
        "Precursor_ID": "CHAMPION 3 (Step 28)",
        "Architecture": "Dual Meta-Learner",
        "Accuracy": 95.62,
        "Precision": 88.04,
        "Recall": 84.38,
        "F1-score": 86.17,
        "ROC-AUC": 94.55,
        "Errors": 26,
        "TN": 487,
        "FP": 11,
        "FN": 15,
        "TP": 81
    },
    {
        "Version": "Experiment 3",
        "Precursor_ID": "CHAMPION 2 (Step 30)",
        "Architecture": "Severe-FN Recovery",
        "Accuracy": 95.79,
        "Precision": 88.17,
        "Recall": 85.42,
        "F1-score": 86.77,
        "ROC-AUC": 94.55,
        "Errors": 25,
        "TN": 487,
        "FP": 11,
        "FN": 14,
        "TP": 82
    },
    {
        "Version": "Experiment 4",
        "Precursor_ID": "CHAMPION 1 (Step 31)",
        "Architecture": "Transition Boundary Recovery",
        "Accuracy": 95.96,
        "Precision": 87.50,
        "Recall": 87.50,
        "F1-score": 87.50,
        "ROC-AUC": 94.55,
        "Errors": 24,
        "TN": 486,
        "FP": 12,
        "FN": 12,
        "TP": 84
    },
    {
        "Version": "Final Hybrid Model",
        "Precursor_ID": "FINAL (Step 32)",
        "Architecture": "Two-Stage Stacking + Specialist",
        "Accuracy": 96.13,
        "Precision": 88.42,
        "Recall": 87.50,
        "F1-score": 87.96,
        "ROC-AUC": 94.55,
        "Errors": 23,
        "TN": 487,
        "FP": 11,
        "FN": 12,
        "TP": 84
    }
]

def save_csv():
    df = pd.DataFrame(DATA)
    # Add step deltas
    df["d_Accuracy_pp"] = [0.0] + [round(df.loc[i, "Accuracy"] - df.loc[i-1, "Accuracy"], 2) for i in range(1, len(df))]
    df["d_F1_pp"] = [0.0] + [round(df.loc[i, "F1-score"] - df.loc[i-1, "F1-score"], 2) for i in range(1, len(df))]
    df["d_Errors"] = [0] + [int(df.loc[i, "Errors"] - df.loc[i-1, "Errors"]) for i in range(1, len(df))]
    
    for d in CSV_OUTPUT_DIRS:
        out_csv = d / "chapter5_model_development_progression.csv"
        df.to_csv(out_csv, index=False)
        print(f"[SUCCESS] Saved CSV to: {out_csv}")
    return df

def generate_figure(df):
    tick_labels = [
        "Experiment 1",
        "Experiment 2",
        "Experiment 3",
        "Experiment 4",
        "Final Hybrid\nModel"
    ]
    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    # -------------------------------------------------------------
    # PANEL (a): Multi-Metric Trajectories Across Progression
    # -------------------------------------------------------------
    # Custom label positions to ensure ZERO overlap
    # Format: (name, vals, color, marker, ls, [(dx, dy, ha, va), ... for each x])
    metrics_info = [
        ("Accuracy", df["Accuracy"].values, PALETTE["accuracy"], "o", "-", [
            (0, 0.55, "center", "bottom"),
            (0, 0.55, "center", "bottom"),
            (0, 0.55, "center", "bottom"),
            (0, 0.55, "center", "bottom"),
            (0, 0.55, "center", "bottom")
        ]),
        ("Precision", df["Precision"].values, PALETTE["precision"], "s", "-", [
            (0, 0.55, "center", "bottom"),
            (0, 0.55, "center", "bottom"),
            (0, 0.55, "center", "bottom"),
            (0.12, 0.45, "left", "bottom"),     # Exp 4: nudge right
            (0, 0.55, "center", "bottom")       # Final: above
        ]),
        ("Recall", df["Recall"].values, PALETTE["recall"], "^", "--", [
            (0, -0.65, "center", "top"),
            (0, -0.65, "center", "top"),
            (0, -0.65, "center", "top"),
            (0, -0.65, "center", "top"),        # Exp 4: below
            (0, -0.65, "center", "top")         # Final: below
        ]),
        ("F1-score", df["F1-score"].values, PALETTE["f1"], "D", "-", [
            (0, 0.55, "center", "bottom"),
            (0, 0.55, "center", "bottom"),
            (0, 0.55, "center", "bottom"),
            (-0.12, 0.0, "right", "center"),     # Exp 4: nudge left
            (0.12, 0.0, "left", "center")        # Final: nudge right
        ]),
        ("ROC-AUC", df["ROC-AUC"].values, PALETTE["roc_auc"], "v", "-.", [
            (0, -0.65, "center", "top"),
            (0, -0.65, "center", "top"),
            (0, -0.65, "center", "top"),
            (0, -0.65, "center", "top"),
            (0, -0.65, "center", "top")
        ])
    ]

    for name, vals, color, marker, ls, offsets in metrics_info:
        ax1.plot(x, vals, label=name, color=color, marker=marker,
                 markersize=7.5, linewidth=2.0, linestyle=ls, zorder=4)
        for i, (val, (dx, dy, ha, va)) in enumerate(zip(vals, offsets)):
            ax1.text(x[i] + dx, val + dy, f"{val:.2f}%", ha=ha, va=va,
                     fontsize=8.0, fontweight="bold", color=color, zorder=5,
                     bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))

    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels, fontsize=9.2, fontweight="bold", color=PALETTE["text_dark"])
    ax1.set_ylabel("Evaluation Metric Score (%)", fontsize=10.5, fontweight="bold",
                   color=PALETTE["text_dark"], labelpad=8)
    ax1.set_ylim(77, 98.8)
    ax1.set_xlim(-0.35, 4.35)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.yaxis.set_major_locator(mtick.MultipleLocator(5))
    ax1.yaxis.set_minor_locator(mtick.MultipleLocator(1))
    ax1.grid(True, linestyle=":", color=PALETTE["grid"], linewidth=0.8, alpha=0.9, zorder=1)

    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.spines["left"].set_color(PALETTE["border_light"])
    ax1.spines["bottom"].set_color(PALETTE["border_light"])

    ax1.legend(loc="lower right", frameon=True, fontsize=8.8, ncol=2,
               facecolor="white", edgecolor=PALETTE["border_light"], framealpha=0.98)
    ax1.set_title("(a) Performance Metric Evolution Across Iterations",
                  fontsize=11.5, fontweight="bold", color=PALETTE["text_dark"], pad=12)

    # -------------------------------------------------------------
    # PANEL (b): Classification Error Reduction (Zero-based honest axis)
    # -------------------------------------------------------------
    errors = df["Errors"].values
    bar_colors = [PALETTE["errors_prev"]] * 4 + [PALETTE["errors_final"]]

    bars = ax2.bar(x, errors, color=bar_colors, width=0.48,
                   edgecolor=PALETTE["text_dark"], linewidth=1.0, zorder=3)

    ax2.set_ylim(0, 36)
    ax2.set_xlim(-0.5, 4.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels, fontsize=9.2, fontweight="bold", color=PALETTE["text_dark"])
    ax2.set_ylabel("Classification Errors (N = 594 Cohort)", fontsize=10.5,
                   fontweight="bold", color=PALETTE["text_dark"], labelpad=8)
    ax2.yaxis.set_major_locator(mtick.MultipleLocator(5))
    ax2.yaxis.set_minor_locator(mtick.MultipleLocator(1))
    ax2.grid(axis="y", linestyle=":", color=PALETTE["grid"], linewidth=0.8, alpha=0.9, zorder=1)

    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.spines["left"].set_color(PALETTE["border_light"])
    ax2.spines["bottom"].set_color(PALETTE["border_light"])

    for bar, err in zip(bars, errors):
        ax2.text(bar.get_x() + bar.get_width() / 2, err + 0.8, f"{int(err)} errors",
                 ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=PALETTE["text_dark"])

    # Error reduction callout badge positioned neatly with clear spacing
    ax2.annotate("-23.3% Error Reduction\n(30 \u2192 23 errors)",
                 xy=(4.0, 24.5), xytext=(2.6, 31.0),
                 arrowprops=dict(facecolor=PALETTE["errors_final"], edgecolor=PALETTE["errors_final"],
                                 arrowstyle="->", lw=1.5),
                 fontsize=8.5, fontweight="bold", color=PALETTE["errors_final"],
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECFDF5", edgecolor="#A7F3D0"),
                 ha="center")

    ax2.set_title("(b) Absolute Classification Error Reduction (Zero-Based Axis)",
                  fontsize=11.5, fontweight="bold", color=PALETTE["text_dark"], pad=12)

    # Main Title
    fig.suptitle("Model Development Progression and Error Forensics Across Experimental Iterations",
                 fontsize=12.5, fontweight="bold", color=PALETTE["text_dark"], y=0.98)

    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.96, wspace=0.22)

    saved_paths = []
    for out_dir in OUTPUT_DIRS:
        p = out_dir / "chapter5_model_development_progression.png"
        fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
        saved_paths.append(p)
        print(f"[SUCCESS] Saved figure to: {p}")

    plt.close(fig)
    return saved_paths

if __name__ == "__main__":
    df_data = save_csv()
    generate_figure(df_data)
