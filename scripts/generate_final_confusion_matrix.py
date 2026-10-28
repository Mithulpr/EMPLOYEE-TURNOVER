#!/usr/bin/env python3
"""
Publication Figure Generator: Final Evaluation Confusion Matrix
===============================================================
Generates a clean, professional, publication-quality Confusion Matrix
for the Final Hybrid Model on the Final Evaluation Cohort (N = 594).

Aesthetics & Design:
- High-resolution 300 DPI vector-crisp output.
- Clean academic card-style layout with soft pastel fills and matching borders.
- Zero clutter: removes "retrospective", uses "Final Evaluation Cohort".
- Generous padding and title headroom to prevent clipping when inserted into Word/LaTeX.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("e:/R&D final")
OUTPUT_PATHS = [
    PROJECT_ROOT / "research_documents" / "reports" / "figures" / "chapter5_confusion_matrix.png",
    PROJECT_ROOT / "research_documents" / "reports" / "figures" / "fig02_confusion_matrix.png",
    PROJECT_ROOT / "final_employee_turnover_system" / "FINAL" / "figures" / "performance" / "fig02_confusion_matrix.png",
    Path("C:/Users/mithu/.gemini/antigravity-ide/brain/81207504-f1b2-4481-b24e-8f9b3c0dbb79/chapter5_confusion_matrix.png"),
    Path("C:/Users/mithu/.gemini/antigravity-ide/brain/81207504-f1b2-4481-b24e-8f9b3c0dbb79/fig02_confusion_matrix.png")
]

for p in OUTPUT_PATHS:
    p.parent.mkdir(parents=True, exist_ok=True)

# Curated palette
PALETTE = {
    "slate_dark": "#0F172A",
    "slate_med": "#475569",
    "green_bg": "#F0FDF4",
    "green_border": "#10B981",
    "green_badge": "#059669",
    "red_bg": "#FEF2F2",
    "red_border": "#EF4444",
    "red_badge": "#DC2626",
    "blue_bg": "#EFF6FF",
    "blue_border": "#3B82F6",
    "blue_badge": "#2563EB",
}

def generate_confusion_matrix():
    tn = 487
    fp = 11
    fn = 12
    tp = 84
    total = tn + fp + fn + tp

    matrix = np.array([[tn, fp], [fn, tp]])
    matrix_pct = matrix / total * 100

    fig, ax = plt.subplots(figsize=(8.8, 6.6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cell_styles = [
        [{"bg": PALETTE["green_bg"], "border": PALETTE["green_border"], "badge_col": PALETTE["green_badge"]},
         {"bg": PALETTE["red_bg"], "border": PALETTE["red_border"], "badge_col": PALETTE["red_badge"]}],
        [{"bg": PALETTE["red_bg"], "border": PALETTE["red_border"], "badge_col": PALETTE["red_badge"]},
         {"bg": PALETTE["blue_bg"], "border": PALETTE["blue_border"], "badge_col": PALETTE["blue_badge"]}]
    ]

    labels_type = [["True Negative (TN)", "False Positive (FP)"],
                   ["False Negative (FN)", "True Positive (TP)"]]
    role_descs = [["Correctly Identified Stayers", "False Alarms (Flagged Leavers)"],
                  ["Missed Leavers (Departed)", "Correctly Caught Leavers"]]

    # Draw cells with clean border lines and balanced typography
    for i in range(2):
        for j in range(2):
            cnt = matrix[i, j]
            pct = matrix_pct[i, j]
            style = cell_styles[i][j]

            rect = plt.Rectangle((j, 1 - i), 1, 1, facecolor=style["bg"],
                                 edgecolor=style["border"], linewidth=2.2)
            ax.add_patch(rect)

            # 1. Category Badge
            ax.text(j + 0.5, 1 - i + 0.76, labels_type[i][j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=style["badge_col"])
            # 2. Main Count
            ax.text(j + 0.5, 1 - i + 0.50, f"{cnt:,}", ha="center", va="center",
                    fontsize=30, fontweight="bold", color=PALETTE["slate_dark"])
            # 3. Proportion of cohort
            ax.text(j + 0.5, 1 - i + 0.30, f"{pct:.2f}% of cohort", ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color=PALETTE["slate_med"])
            # 4. Role description
            ax.text(j + 0.5, 1 - i + 0.16, role_descs[i][j], ha="center", va="center",
                    fontsize=10, color=PALETTE["slate_med"])

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted: Stayed (0)", "Predicted: Left (1)"],
                       fontsize=11.5, fontweight="bold", color=PALETTE["slate_dark"])
    ax.set_yticklabels(["Actually: Left (1)", "Actually: Stayed (0)"],
                       fontsize=11.5, fontweight="bold", color=PALETTE["slate_dark"])
    
    # Hide protruding tick lines and outer spines for clean card presentation
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Model Decision Output (Threshold \u03c4 = 0.3610)",
                  fontsize=12, fontweight="bold", labelpad=14, color=PALETTE["slate_dark"])
    ax.set_ylabel("Ground Truth Outcome (Final Evaluation Records)",
                  fontsize=12, fontweight="bold", labelpad=14, color=PALETTE["slate_dark"])

    # Clean title with generous top padding to ensure zero clipping
    ax.set_title("Confusion Matrix Analysis \u2014 Final Evaluation Cohort (N = 594)\n"
                 "Accuracy: 96.13% (571 / 594)   \u2022   Total Misclassifications: 23 (3.87%)",
                 fontsize=13, fontweight="bold", pad=24, color=PALETTE["slate_dark"], linespacing=1.4)

    plt.subplots_adjust(top=0.82, bottom=0.08, left=0.15, right=0.95)

    for p in OUTPUT_PATHS:
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"[SUCCESS] Saved figure to: {p}")

    plt.close(fig)

if __name__ == "__main__":
    generate_confusion_matrix()
