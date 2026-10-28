"""
Error Analysis and Misclassification Diagnostic Module.
Profiles False Positives and False Negatives across demographic, organizational, and tenure cohorts.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, Tuple

def perform_error_analysis(
    X_test_orig: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    figures_dir: str,
    tables_dir: str
) -> pd.DataFrame:
    """
    Performs granular cohort-based error analysis on test predictions.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    print("Performing granular misclassification error analysis...")

    df_err = X_test_orig.copy()
    df_err['Actual_Attrition'] = y_test.values
    df_err['Predicted_Attrition'] = y_pred
    df_err['Predicted_Probability'] = y_proba

    # Error classification
    def categorize_outcome(row):
        act = row['Actual_Attrition']
        pred = row['Predicted_Attrition']
        if act == 1 and pred == 1:
            return "True Positive (TP)"
        elif act == 0 and pred == 0:
            return "True Negative (TN)"
        elif act == 0 and pred == 1:
            return "False Positive (FP - False Alarm)"
        else:
            return "False Negative (FN - Missed Turnover)"

    df_err['Error_Category'] = df_err.apply(categorize_outcome, axis=1)

    # Breakdown by Department
    dept_breakdown = pd.crosstab(df_err['Department'], df_err['Error_Category'], normalize='index') * 100
    
    # Breakdown by Age Group
    df_err['AgeCohort'] = pd.cut(df_err['Age'], bins=[0, 29, 40, 50, 100], labels=['<30', '30-40', '41-50', '50+'])
    age_breakdown = pd.crosstab(df_err['AgeCohort'], df_err['Error_Category'], normalize='index') * 100

    # Breakdown by Tenure
    df_err['TenureCohort'] = pd.cut(df_err['YearsAtCompany'], bins=[-1, 2, 5, 10, 100], labels=['0-2 yrs', '3-5 yrs', '6-10 yrs', '>10 yrs'])
    tenure_breakdown = pd.crosstab(df_err['TenureCohort'], df_err['Error_Category'], normalize='index') * 100

    # Plot Error Diagnostics across cohorts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Department
    dept_breakdown[['False Positive (FP - False Alarm)', 'False Negative (FN - Missed Turnover)']].plot(
        kind='bar', stacked=False, color=['#e67e22', '#e74c3c'], ax=axes[0], edgecolor='black'
    )
    axes[0].set_title('Misclassification Rate by Department', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Cohort Percentage (%)', fontsize=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha='right')
    axes[0].legend(fontsize=9)

    # Age
    age_breakdown[['False Positive (FP - False Alarm)', 'False Negative (FN - Missed Turnover)']].plot(
        kind='bar', stacked=False, color=['#e67e22', '#e74c3c'], ax=axes[1], edgecolor='black'
    )
    axes[1].set_title('Misclassification Rate by Age Cohort', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Cohort Percentage (%)', fontsize=10)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(fontsize=9)

    # Tenure
    tenure_breakdown[['False Positive (FP - False Alarm)', 'False Negative (FN - Missed Turnover)']].plot(
        kind='bar', stacked=False, color=['#e67e22', '#e74c3c'], ax=axes[2], edgecolor='black'
    )
    axes[2].set_title('Misclassification Rate by Tenure Cohort', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Cohort Percentage (%)', fontsize=10)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    error_plot_path = os.path.join(figures_dir, 'error_distribution_by_strata.png')
    plt.savefig(error_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {error_plot_path}")

    # Save detailed error log CSV
    error_csv_path = os.path.join(tables_dir, 'test_error_analysis_details.csv')
    df_err.to_csv(error_csv_path, index=False)
    print(f"Saved error details to: {error_csv_path}")

    return df_err
