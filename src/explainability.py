"""
Explainability and Interpretability Module (SHAP and Glassbox EBM).
Generates publication-quality global and local interpretability artifacts:
- SHAP Beeswarm Summary Plots
- SHAP Mean Feature Importance Bar Plots
- Local Waterfall Explanations for High-Risk and Low-Risk Cases
- Explainable Boosting Machine (EBM) Global Term and Shape Function Visualizations
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List
from interpret.glassbox import ExplainableBoostingClassifier

def generate_shap_explanations(
    model, 
    X_sample: pd.DataFrame, 
    feature_names: List[str], 
    figures_dir: str
):
    """
    Computes SHAP values and outputs publication-grade visualization figures.
    """
    os.makedirs(figures_dir, exist_ok=True)
    print("Computing SHAP values and generating interpretability plots...")

    # Choose appropriate explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
    except Exception:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

    # Handle multi-dimensional SHAP outputs
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        # Binary classification: extract class 1 (Attrition=Yes)
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    # 1. SHAP Beeswarm Summary Plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values_class1, max_display=15, show=False)
    plt.title("SHAP Global Feature Impact (Beeswarm Distribution for Attrition=1)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    beeswarm_path = os.path.join(figures_dir, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {beeswarm_path}")

    # 2. SHAP Bar Feature Importance Plot
    plt.figure(figsize=(10, 7))
    shap.plots.bar(shap_values_class1, max_display=15, show=False)
    plt.title("SHAP Mean Absolute Feature Importance", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    bar_path = os.path.join(figures_dir, "shap_importance_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bar_path}")

    # 3. High-Risk Individual Local Waterfall Explanation
    # Find sample with highest predicted risk
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_sample)[:, 1]
    else:
        probs = model.predict(X_sample)
        
    high_risk_idx = int(np.argmax(probs))
    low_risk_idx = int(np.argmin(probs))

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values_class1[high_risk_idx], max_display=10, show=False)
    plt.title(f"Local Decision Attribution: High-Risk Profile (Predicted P={probs[high_risk_idx]:.2f})", fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    waterfall_high_path = os.path.join(figures_dir, "shap_waterfall_high_risk.png")
    plt.savefig(waterfall_high_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {waterfall_high_path}")

    # 4. Low-Risk Individual Local Waterfall Explanation
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values_class1[low_risk_idx], max_display=10, show=False)
    plt.title(f"Local Decision Attribution: Low-Risk Profile (Predicted P={probs[low_risk_idx]:.2f})", fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    waterfall_low_path = os.path.join(figures_dir, "shap_waterfall_low_risk.png")
    plt.savefig(waterfall_low_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {waterfall_low_path}")

def generate_ebm_explanations(
    ebm_model: ExplainableBoostingClassifier,
    figures_dir: str
):
    """
    Extracts and plots Explainable Boosting Machine (EBM) global feature explanations.
    """
    os.makedirs(figures_dir, exist_ok=True)
    print("Generating EBM glassbox global interpretability plots...")
    
    try:
        ebm_global = ebm_model.explain_global()
        feature_names = ebm_global.data()['names']
        scores = ebm_global.data()['scores']

        df_ebm = pd.DataFrame({
            'Feature': feature_names,
            'Importance': scores
        }).sort_values(by='Importance', ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(df_ebm['Feature'], df_ebm['Importance'], color='#16a085', edgecolor='black', height=0.6)
        ax.set_title("EBM (Glassbox) Global Term Importance", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Mean Absolute Score (Log-Odds Impact)", fontsize=11, fontweight='bold')
        plt.tight_layout()
        ebm_path = os.path.join(figures_dir, "ebm_global_importance.png")
        plt.savefig(ebm_path, dpi=300)
        plt.close()
        print(f"Saved: {ebm_path}")
    except Exception as e:
        print(f"Could not generate EBM plot: {e}")
