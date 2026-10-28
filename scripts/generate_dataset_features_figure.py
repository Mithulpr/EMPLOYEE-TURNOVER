#!/usr/bin/env python3
"""
Publication Figure Generator: Dataset Architecture & 30 Selected Features Overview
===================================================================================
Generates an executive, publication-grade diagram image showcasing:
1. Complete Dataset Feature Pipeline: Raw 45 attributes -> Data hygiene & pruning -> 30 Curated Model Features.
2. Comprehensive, neatly grouped listing of all 30 features used in the Step 32 Final Hybrid Model.
3. Dataset split context (2,969 total = 2,375 Training + 594 Evaluation Cohort).
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Directories
PROJECT_ROOT = Path("e:/R&D final")
BRAIN_DIR = Path("C:/Users/mithu/.gemini/antigravity-ide/brain/a599bef2-0828-49a1-ab52-0ccc8b9ffb48")
OUTPUT_DIRS = [
    PROJECT_ROOT / "research_documents" / "reports" / "figures",
    PROJECT_ROOT / "final_employee_turnover_system" / "FINAL" / "figures" / "performance",
    BRAIN_DIR
]

for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# Curated palette
COLORS = {
    "bg": "#F8FAFC",                # Slate 50
    "card_bg": "#FFFFFF",           # Pure White
    "header_dark": "#0F172A",       # Slate 900
    "sub_text": "#475569",          # Slate 600
    "border": "#CBD5E1",            # Slate 300
    
    # Feature Category Accents
    "num_accent": "#1E40AF",        # Royal Blue
    "num_light": "#EFF6FF",         # Blue 50
    "num_border": "#93C5FD",        # Blue 300
    
    "eng_accent": "#047857",        # Emerald Green
    "eng_light": "#ECFDF5",         # Emerald 50
    "eng_border": "#6EE7B7",        # Emerald 300
    
    "cat_accent": "#7C3AED",        # Purple
    "cat_light": "#F5F3FF",         # Purple 50
    "cat_border": "#C4B5FD",        # Purple 300
    
    # Summary Card
    "summary_bg": "#0F172A",        # Dark Navy
    "summary_gold": "#F59E0B",      # Amber/Gold
    "summary_teal": "#06B6D4",      # Cyan
}

def draw_badge(ax, x, y, text, color, bg_color, fontsize=10, pad_x=0.03, pad_y=0.015):
    """Draw a clean pill badge."""
    t = ax.text(x, y, text, color=color, fontsize=fontsize, fontweight="bold",
                va="center", ha="center", zorder=5)
    return t

def generate_figure():
    fig = plt.figure(figsize=(18, 12), facecolor=COLORS["bg"], dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # =========================================================================
    # 1. HEADER SECTION
    # =========================================================================
    ax.text(0.8, 11.45, "EMPLOYEE TURNOVER PREDICTION SYSTEM", 
            fontsize=12, fontweight="heavy", color=COLORS["num_accent"])
    ax.text(0.8, 10.95, "Dataset Feature Architecture & 30 Model Input Predictors", 
            fontsize=24, fontweight="bold", color=COLORS["header_dark"])
    ax.text(0.8, 10.60, "Complete breakdown of dataset dimensional transformation from 45 raw features to the locked 30-feature vector in the Step 32 Final Hybrid Model", 
            fontsize=12, color=COLORS["sub_text"])

    # =========================================================================
    # 2. TOP METRIC CARDS (PIPELINE TRANSFORMATION FLOW)
    # =========================================================================
    cards_data = [
        {
            "rect": (0.8, 9.35, 3.8, 1.05),
            "label": "RAW DATASET COLUMNS",
            "val": "45 Features",
            "desc": "Initial columns (2,969 records)",
            "color": "#334155",
            "accent": "#64748B"
        },
        {
            "rect": (5.0, 9.35, 3.8, 1.05),
            "label": "PRUNED & RECTIFIED",
            "val": "-14 Removed",
            "desc": "IDs, constants & redundant labels",
            "color": "#DC2626",
            "accent": "#EF4444"
        },
        {
            "rect": (9.2, 9.35, 3.8, 1.05),
            "label": "DOMAIN ENGINEERED",
            "val": "+12 Ratios",
            "desc": "Tenure, burnout & hazard metrics",
            "color": COLORS["eng_accent"],
            "accent": "#10B981"
        },
        {
            "rect": (13.4, 9.35, 3.8, 1.05),
            "label": "FINAL INPUT PREDICTORS",
            "val": "30 Features",
            "desc": "Strict input vector for all models",
            "color": COLORS["num_accent"],
            "accent": "#3B82F6"
        }
    ]

    for c in cards_data:
        x, y, w, h = c["rect"]
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                     facecolor=COLORS["card_bg"], edgecolor=COLORS["border"], linewidth=1.2, zorder=2)
        ax.add_patch(box)
        # Left accent stripe
        stripe = patches.FancyBboxPatch((x, y), 0.12, h, boxstyle="round,pad=0,rounding_size=0.06",
                                        facecolor=c["accent"], edgecolor="none", zorder=3)
        ax.add_patch(stripe)
        ax.text(x + 0.35, y + 0.76, c["label"], fontsize=9, fontweight="bold", color=COLORS["sub_text"])
        ax.text(x + 0.35, y + 0.42, c["val"], fontsize=17, fontweight="bold", color=c["color"])
        ax.text(x + 0.35, y + 0.18, c["desc"], fontsize=9.5, color=COLORS["sub_text"])

    # Connectors between cards
    for arrow_x in [4.65, 8.85, 13.05]:
        ax.annotate("", xy=(arrow_x + 0.3, 9.87), xytext=(arrow_x - 0.05, 9.87),
                    arrowprops=dict(arrowstyle="->", color=COLORS["border"], lw=2.2))

    # =========================================================================
    # 3. THREE FEATURE COLUMNS (THE 30 FEATURES LIST)
    # =========================================================================
    
    # --- Column 1: Baseline Demographics & Tenure (8 features) ---
    col1_x, col_y, col_w, col_h = 0.8, 1.6, 5.0, 7.35
    box1 = patches.FancyBboxPatch((col1_x, col_y), col_w, col_h, boxstyle="round,pad=0,rounding_size=0.2",
                                  facecolor=COLORS["card_bg"], edgecolor=COLORS["num_border"], linewidth=1.5, zorder=2)
    ax.add_patch(box1)
    
    # Header bar
    hbar1 = patches.FancyBboxPatch((col1_x, col_y + col_h - 0.85), col_w, 0.85, 
                                   boxstyle="round,pad=0,rounding_size=0.2",
                                   facecolor=COLORS["num_light"], edgecolor=COLORS["num_border"], linewidth=1.2, zorder=3)
    ax.add_patch(hbar1)
    ax.text(col1_x + 0.4, col_y + col_h - 0.42, "1. Core Demographics & Tenure", fontsize=13, fontweight="bold", color=COLORS["num_accent"], va="center")
    ax.text(col1_x + col_w - 0.4, col_y + col_h - 0.42, "8 Features", fontsize=10.5, fontweight="bold", color=COLORS["num_accent"], ha="right", va="center")

    num_features = [
        ("1", "Age", "Continuous", "Employee age in years"),
        ("2", "TotalWorkingYears", "Continuous", "Total career working history"),
        ("3", "YearsWithCurrManager", "Continuous", "Tenure working under current manager"),
        ("4", "YearsSinceLastPromotion", "Continuous", "Years elapsed since previous promotion"),
        ("5", "TrainingTimesLastYear", "Discrete", "Professional development sessions attended"),
        ("6", "JobSatisfaction", "Ordinal (1-4)", "Standardized employee survey rating"),
        ("7", "EnvironmentSatisfaction", "Ordinal (1-4)", "Workplace psychological climate rating"),
        ("8", "WorkLifeBalance", "Ordinal (1-4)", "Equilibrium rating between life and job")
    ]

    item_y = col_y + col_h - 1.35
    for idx, fname, ftype, fdesc in num_features:
        # Mini bullet pill
        pill = patches.FancyBboxPatch((col1_x + 0.35, item_y - 0.15), 0.45, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                     facecolor=COLORS["num_light"], edgecolor=COLORS["num_accent"], linewidth=1, zorder=3)
        ax.add_patch(pill)
        ax.text(col1_x + 0.575, item_y + 0.06, idx, fontsize=9.5, fontweight="bold", color=COLORS["num_accent"], ha="center", va="center", zorder=4)
        
        ax.text(col1_x + 0.95, item_y + 0.15, fname, fontsize=11, fontweight="bold", color=COLORS["header_dark"])
        ax.text(col1_x + col_w - 0.35, item_y + 0.15, ftype, fontsize=9, fontweight="semibold", color=COLORS["num_accent"], ha="right")
        ax.text(col1_x + 0.95, item_y - 0.12, fdesc, fontsize=9, color=COLORS["sub_text"])
        
        # subtle divider
        ax.plot([col1_x + 0.35, col1_x + col_w - 0.35], [item_y - 0.32, item_y - 0.32], color="#F1F5F9", lw=1)
        item_y -= 0.73

    # --- Column 2: Domain Engineered Ratios (12 features) ---
    col2_x = 6.2
    box2 = patches.FancyBboxPatch((col2_x, col_y), col_w, col_h, boxstyle="round,pad=0,rounding_size=0.2",
                                  facecolor=COLORS["card_bg"], edgecolor=COLORS["eng_border"], linewidth=1.5, zorder=2)
    ax.add_patch(box2)
    
    hbar2 = patches.FancyBboxPatch((col2_x, col_y + col_h - 0.85), col_w, 0.85, 
                                   boxstyle="round,pad=0,rounding_size=0.2",
                                   facecolor=COLORS["eng_light"], edgecolor=COLORS["eng_border"], linewidth=1.2, zorder=3)
    ax.add_patch(hbar2)
    ax.text(col2_x + 0.4, col_y + col_h - 0.42, "2. Domain Engineered Signals", fontsize=13, fontweight="bold", color=COLORS["eng_accent"], va="center")
    ax.text(col2_x + col_w - 0.4, col_y + col_h - 0.42, "12 Features", fontsize=10.5, fontweight="bold", color=COLORS["eng_accent"], ha="right", va="center")

    eng_features = [
        ("9", "TotalExpToAgeRatio", "Ratio", "Total career years divided by age"),
        ("10", "CompanyLoyaltyRatio", "Ratio", "Years at company / Total working years"),
        ("11", "PriorCompanyExperience", "Tenure", "Experience accumulated prior to joining"),
        ("12", "JobHoppingRate", "Rate", "Frequency of organizational transitions"),
        ("13", "PromotionStagnationRatio", "Hazard", "Years since promotion / Company tenure"),
        ("14", "PromotionLagScore", "Latency", "Career velocity delay indicator"),
        ("15", "ManagerTenureMismatch", "Delta", "Mismatch between company and manager tenure"),
        ("16", "WorkLifeDeficit", "Deficit", "Gap between expected and actual balance"),
        ("17", "SevereDissatisfactionFlag", "Binary", "Active trigger when multiple surveys <= 1"),
        ("18", "EarlyCareerFlightRisk", "Risk Index", "Compound vulnerability during tenure < 2 yrs"),
        ("19", "PromotionVelocity", "Velocity", "Speed of promotion per career phase"),
        ("20", "PromotionPaceStrain", "Interaction", "Strain of rapid pace vs organizational support")
    ]

    item_y = col_y + col_h - 1.25
    for idx, fname, ftype, fdesc in eng_features:
        pill = patches.FancyBboxPatch((col2_x + 0.35, item_y - 0.12), 0.45, 0.36, boxstyle="round,pad=0,rounding_size=0.08",
                                     facecolor=COLORS["eng_light"], edgecolor=COLORS["eng_accent"], linewidth=1, zorder=3)
        ax.add_patch(pill)
        ax.text(col2_x + 0.575, item_y + 0.06, idx, fontsize=9, fontweight="bold", color=COLORS["eng_accent"], ha="center", va="center", zorder=4)
        
        ax.text(col2_x + 0.95, item_y + 0.13, fname, fontsize=10.5, fontweight="bold", color=COLORS["header_dark"])
        ax.text(col2_x + col_w - 0.35, item_y + 0.13, ftype, fontsize=8.5, fontweight="semibold", color=COLORS["eng_accent"], ha="right")
        ax.text(col2_x + 0.95, item_y - 0.11, fdesc, fontsize=8.2, color=COLORS["sub_text"])
        
        ax.plot([col2_x + 0.35, col2_x + col_w - 0.35], [item_y - 0.23, item_y - 0.23], color="#F1F5F9", lw=0.8)
        item_y -= 0.50

    # --- Column 3: One-Hot Categorical Features (10 features) ---
    col3_x = 11.6
    box3 = patches.FancyBboxPatch((col3_x, col_y), col_w, col_h, boxstyle="round,pad=0,rounding_size=0.2",
                                  facecolor=COLORS["card_bg"], edgecolor=COLORS["cat_border"], linewidth=1.5, zorder=2)
    ax.add_patch(box3)
    
    hbar3 = patches.FancyBboxPatch((col3_x, col_y + col_h - 0.85), col_w, 0.85, 
                                   boxstyle="round,pad=0,rounding_size=0.2",
                                   facecolor=COLORS["cat_light"], edgecolor=COLORS["cat_border"], linewidth=1.2, zorder=3)
    ax.add_patch(hbar3)
    ax.text(col3_x + 0.35, col_y + col_h - 0.42, "3. Categorical Attributes", fontsize=12.5, fontweight="bold", color=COLORS["cat_accent"], va="center")
    ax.text(col3_x + col_w - 0.35, col_y + col_h - 0.42, "10 Features (One-Hot)", fontsize=10, fontweight="bold", color=COLORS["cat_accent"], ha="right", va="center")

    cat_features = [
        ("21", "BusinessTravel_Non-Travel", "Binary {0,1}", "Employee travels zero percent for work"),
        ("22", "BusinessTravel_Travel_Frequently", "Binary {0,1}", "High travel frequency workload"),
        ("23", "BusinessTravel_Travel_Rarely", "Binary {0,1}", "Occasional travel requirement"),
        ("24", "EducationField_Life Sciences", "Binary {0,1}", "Academic background: Life Sciences"),
        ("25", "EducationField_Marketing", "Binary {0,1}", "Academic background: Marketing"),
        ("26", "EducationField_Other", "Binary {0,1}", "Academic background: Other Disciplines"),
        ("27", "Gender_Female", "Binary {0,1}", "Demographic gender designation: Female"),
        ("28", "Gender_Male", "Binary {0,1}", "Demographic gender designation: Male"),
        ("29", "JobRole_Human Resources", "Binary {0,1}", "Department role: HR specialist"),
        ("30", "MaritalStatus_Single", "Binary {0,1}", "Marital designation: Unmarried single")
    ]

    item_y = col_y + col_h - 1.30
    for idx, fname, ftype, fdesc in cat_features:
        pill = patches.FancyBboxPatch((col3_x + 0.35, item_y - 0.12), 0.45, 0.38, boxstyle="round,pad=0,rounding_size=0.08",
                                     facecolor=COLORS["cat_light"], edgecolor=COLORS["cat_accent"], linewidth=1, zorder=3)
        ax.add_patch(pill)
        ax.text(col3_x + 0.575, item_y + 0.07, idx, fontsize=9, fontweight="bold", color=COLORS["cat_accent"], ha="center", va="center", zorder=4)
        
        ax.text(col3_x + 0.95, item_y + 0.14, fname, fontsize=10.5, fontweight="bold", color=COLORS["header_dark"])
        ax.text(col3_x + col_w - 0.35, item_y + 0.14, ftype, fontsize=8.5, fontweight="semibold", color=COLORS["cat_accent"], ha="right")
        ax.text(col3_x + 0.95, item_y - 0.11, fdesc, fontsize=8.5, color=COLORS["sub_text"])
        
        ax.plot([col3_x + 0.35, col3_x + col_w - 0.35], [item_y - 0.26, item_y - 0.26], color="#F1F5F9", lw=0.8)
        item_y -= 0.58

    # =========================================================================
    # 4. BOTTOM FOOTER BANNER (DATASET RECAP)
    # =========================================================================
    footer_box = patches.FancyBboxPatch((0.8, 0.4), 15.8, 0.95, boxstyle="round,pad=0,rounding_size=0.15",
                                        facecolor=COLORS["summary_bg"], edgecolor="none", zorder=2)
    ax.add_patch(footer_box)

    ax.text(1.2, 0.95, "DATASET RECAP & BENCHMARK INTEGRITY", fontsize=9.5, fontweight="bold", color=COLORS["summary_gold"])
    ax.text(1.2, 0.65, "Total Profiles: 2,969   •   Training: 2,375 (80%)   •   Evaluation Cohort: 594 (20%)",
            fontsize=11, fontweight="bold", color="#F8FAFC")
    
    ax.text(16.2, 0.95, "Target: Attrition (0 = Stayed, 1 = Left)", fontsize=9.5, fontweight="semibold", color="#E2E8F0", ha="right")
    ax.text(16.2, 0.65, "Champion: Step 32 Final Hybrid (Accuracy: 96.13%  •  Errors: 23 / 594)", 
            fontsize=10, fontweight="bold", color=COLORS["summary_teal"], ha="right")

    # Save to all target paths
    for out_dir in OUTPUT_DIRS:
        target_path = out_dir / "dataset_feature_architecture.png"
        fig.savefig(target_path, dpi=300, facecolor=COLORS["bg"], bbox_inches="tight")
        print(f"Saved: {target_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_figure()
