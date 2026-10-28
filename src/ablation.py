"""
Systematic Component Ablation Study Module.
Quantifies the marginal performance contribution of each methodological step:
1. Raw Baseline
2. Data Hygiene & Normalization
3. HR Domain Feature Engineering
4. Multi-Method Feature Selection
5. Imbalance Handling
6. Optuna Hyperparameter Optimization
7. Optimal Decision-Threshold Tuning
8. Advanced Stacking Ensemble
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from src.evaluation import compute_all_metrics, optimize_decision_threshold

def run_systematic_ablation(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_train_hygiene: pd.DataFrame,
    X_train_engineered: pd.DataFrame,
    X_train_selected: pd.DataFrame,
    tuned_model,
    best_threshold: float,
    stacking_ensemble = None,
    cv: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, str]:
    """
    Executes a step-by-step ablation study isolating the impact of each innovation.
    """
    print("Running Systematic Ablation Study across 8 distinct stages...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    stages = []

    # Helper for evaluating CV
    def eval_stage_oof(clf, X_data, threshold=0.5):
        oof_probs = np.zeros(len(y_train))
        for tr_idx, val_idx in skf.split(X_data, y_train):
            X_tr, y_tr = X_data.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_data.iloc[val_idx], y_train.iloc[val_idx]
            
            c = clone(clf)
            c.fit(X_tr, y_tr)
            if hasattr(c, "predict_proba"):
                oof_probs[val_idx] = c.predict_proba(X_val)[:, 1]
            elif hasattr(c, "decision_function"):
                df_v = c.decision_function(X_val)
                oof_probs[val_idx] = 1 / (1 + np.exp(-df_v))
            else:
                oof_probs[val_idx] = c.predict(X_val)
        return compute_all_metrics(y_train.values, oof_probs, threshold=threshold)

    # Stage 1: Raw Baseline (No hygiene, standard RF)
    base_rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    m1 = eval_stage_oof(base_rf, X_train_raw, threshold=0.5)
    stages.append({
        "Stage_Index": 1,
        "Stage_Name": "1. Raw Baseline (Uncleaned)",
        "Description": "Default Random Forest on raw unharmonized features",
        **m1
    })

    # Stage 2: + Data Hygiene & Normalization
    m2 = eval_stage_oof(base_rf, X_train_hygiene, threshold=0.5)
    stages.append({
        "Stage_Index": 2,
        "Stage_Name": "2. + Data Hygiene & Normalization",
        "Description": "Fixed 10x salary scale, string typos, pruned IDs",
        **m2
    })

    # Stage 3: + HR Domain Feature Engineering
    m3 = eval_stage_oof(base_rf, X_train_engineered, threshold=0.5)
    stages.append({
        "Stage_Index": 3,
        "Stage_Name": "3. + HR Feature Engineering",
        "Description": "Added tenure stability, stagnation, and burnout features",
        **m3
    })

    # Stage 4: + Multi-Method Feature Selection
    m4 = eval_stage_oof(base_rf, X_train_selected, threshold=0.5)
    stages.append({
        "Stage_Index": 4,
        "Stage_Name": "4. + Feature Selection",
        "Description": "Filtered features using consensus selection",
        **m4
    })

    # Stage 5: + Imbalance Handling
    balanced_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state)
    m5 = eval_stage_oof(balanced_rf, X_train_selected, threshold=0.5)
    stages.append({
        "Stage_Index": 5,
        "Stage_Name": "5. + Imbalance Handling",
        "Description": "Applied balanced class weights inside CV",
        **m5
    })

    # Stage 6: + Optuna Hyperparameter Optimization
    m6 = eval_stage_oof(tuned_model, X_train_selected, threshold=0.5)
    stages.append({
        "Stage_Index": 6,
        "Stage_Name": "6. + Optuna Optimization",
        "Description": "Bayesian tuned hyperparameters on best algorithm",
        **m6
    })

    # Stage 7: + Optimal Decision-Threshold Tuning
    m7 = eval_stage_oof(tuned_model, X_train_selected, threshold=best_threshold)
    stages.append({
        "Stage_Index": 7,
        "Stage_Name": "7. + Threshold Optimization",
        "Description": f"Optimal decision threshold calibrated (tau={best_threshold:.2f})",
        **m7
    })

    # Stage 8: + Advanced Stacking Ensemble (if present)
    if stacking_ensemble is not None:
        m8 = eval_stage_oof(stacking_ensemble, X_train_selected, threshold=best_threshold)
        stages.append({
            "Stage_Index": 8,
            "Stage_Name": "8. + Stacking Ensemble",
            "Description": "Multi-family meta-learning ensemble with optimal threshold",
            **m8
        })

    df_ablation = pd.DataFrame(stages)
    return df_ablation

def plot_ablation_progression(df_ablation: pd.DataFrame, save_path: str):
    """
    Generates high-resolution progression line chart of Accuracy, F1, Recall, and ROC-AUC across ablation stages.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stages = [f"S{row['Stage_Index']}" for _, row in df_ablation.iterrows()]
    stage_names = [f"S{row['Stage_Index']}: {row['Stage_Name'].split(' ')[1]}" for _, row in df_ablation.iterrows()]
    
    ax.plot(stages, df_ablation['Accuracy'] * 100, marker='o', lw=2.2, color='#2980b9', label='Accuracy (%)')
    ax.plot(stages, df_ablation['F1_Score'] * 100, marker='s', lw=2.2, color='#e67e22', label='F1-Score (%)')
    ax.plot(stages, df_ablation['Recall'] * 100, marker='^', lw=2.2, color='#27ae60', label='Recall (%)')
    ax.plot(stages, df_ablation['ROC_AUC'] * 100, marker='d', lw=2.2, color='#8e44ad', label='ROC-AUC (%)')

    for i, row in df_ablation.iterrows():
        ax.annotate(f"{row['Accuracy']*100:.1f}%", (stages[i], row['Accuracy']*100), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9, fontweight='bold', color='#2980b9')
        ax.annotate(f"{row['F1_Score']*100:.1f}%", (stages[i], row['F1_Score']*100), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=9, fontweight='bold', color='#e67e22')

    ax.set_title('Systematic Step-by-Step Ablation Progression Across R&D Components', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Ablation Stage', fontsize=11, fontweight='bold')
    ax.set_ylabel('Validation Performance Metric (%)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
