"""
Historical Champion Progression Module
======================================
Documents developmental iteration milestones across legacy holdout cohort:
- Champion 4: 94.95% accuracy, 30 errors
- Step 28: 95.62% accuracy, 26 errors
- Step 30: 95.79% accuracy, 25 errors
- Step 31: 95.96% accuracy, 24 errors
- Step 32 Final Hybrid: 96.13% accuracy, 23 errors

Scientific Note:
These figures represent retrospective development history, not independent test progress.
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def load_champion_progression() -> pd.DataFrame:
    """Returns verified historical milestone progression data."""
    data = [
        {"Milestone": "Champion 4", "Model_Type": "Single CatBoost Base", "Accuracy_Pct": 94.95, "Total_Errors": 30, "FN_Errors": 17, "FP_Errors": 13},
        {"Milestone": "Step 28", "Model_Type": "Two-Stage Stacking Backbone", "Accuracy_Pct": 95.62, "Total_Errors": 26, "FN_Errors": 15, "FP_Errors": 11},
        {"Milestone": "Step 30", "Model_Type": "Severe-FN Recovery", "Accuracy_Pct": 95.79, "Total_Errors": 25, "FN_Errors": 14, "FP_Errors": 11},
        {"Milestone": "Step 31", "Model_Type": "Candidate v2 (Boundary Recovery)", "Accuracy_Pct": 95.96, "Total_Errors": 24, "FN_Errors": 12, "FP_Errors": 12},
        {"Milestone": "Step 32 Final Hybrid", "Model_Type": "Full Hybrid + Specialist Refinement", "Accuracy_Pct": 96.13, "Total_Errors": 23, "FN_Errors": 12, "FP_Errors": 11}
    ]
    df = pd.DataFrame(data)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUTS_DIR, "champion_progression.csv"), index=False)
    return df
