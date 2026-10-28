"""
Research Gap Extension: Complete SHAP Explainability Engine
===========================================================
Addresses Task 2 requirements:
1. Component-wise TreeExplainer on base tree learners in margin/log-odds space.
2. Additivity verification with residual tolerance < 1e-4.
3. Complete hybrid pipeline explanation using model-agnostic explainer in probability space [0, 1].
4. Global bar importance, beeswarm distribution, HR interaction dependence plots.
5. Individual employee waterfall explanations for high-risk and low-risk cases.
6. Formal non-causal disclaimers separating statistical attribution from organizational causation.
"""

import os
import sys
import json
import joblib
import warnings
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.dirname(CURRENT_DIR)
BASE_DIR = os.path.dirname(SYSTEM_DIR)

for p in [CURRENT_DIR, SYSTEM_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


def generate_shap_visualizations() -> Dict[str, Any]:
    print("=" * 80)
    print("GENERATING MODEL-SPECIFIC & COMPLETE-PIPELINE SHAP EXPLANATIONS")
    print("=" * 80)

    models_dir = os.path.join(CURRENT_DIR, "models")
    shap_dir = os.path.join(CURRENT_DIR, "shap")
    os.makedirs(shap_dir, exist_ok=True)

    # 1. Load saved model artifacts
    fe = joblib.load(os.path.join(models_dir, "hr_feature_engineer.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    cb_model = joblib.load(os.path.join(models_dir, "catboost_base.joblib"))
    lgb_model = joblib.load(os.path.join(models_dir, "lightgbm_base.joblib"))
    s27_model = joblib.load(os.path.join(models_dir, "anchor_catboost.joblib"))
    meta_lr = joblib.load(os.path.join(models_dir, "meta_logistic_regression.joblib"))

    with open(os.path.join(models_dir, "pipeline_schema.json"), "r", encoding="utf-8") as f:
        schema = json.load(f)
    feature_names = schema["features"]
    tau = schema["decision_threshold"]

    # 2. Load dataset and prepare test holdout
    from src.data_loader import load_and_validate_raw_data, clean_and_harmonize_dataset
    from sklearn.model_selection import train_test_split
    df_raw, _ = load_and_validate_raw_data()
    df_clean = clean_and_harmonize_dataset(df_raw)
    y_all = df_clean["Attrition"].values
    X_raw_all = df_clean.drop(columns=["Attrition"])

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )

    X_tr_fe = fe.transform(X_tr_raw)
    X_te_fe = fe.transform(X_te_raw)

    X_tr_scaled = scaler.transform(X_tr_fe[feature_names].fillna(0.0))
    X_te_scaled = scaler.transform(X_te_fe[feature_names].fillna(0.0))

    # 3. Component-Wise TreeExplainer on CatBoost
    print("\n[STEP 1] Running Component-Wise TreeExplainer on Base CatBoost...")
    explainer_cb = shap.TreeExplainer(cb_model)
    shap_vals_cb = explainer_cb.shap_values(X_te_scaled)
    ev_cb = explainer_cb.expected_value

    if isinstance(ev_cb, (list, np.ndarray)):
        ev_cb = float(ev_cb[-1]) if len(ev_cb) > 1 else float(ev_cb[0])
    else:
        ev_cb = float(ev_cb)

    # Additivity Verification on Sample Records
    raw_preds = cb_model.predict(X_te_scaled, prediction_type="RawFormulaVal")
    residuals = []
    for i in range(min(50, len(X_te_scaled))):
        sum_shap = np.sum(shap_vals_cb[i]) + ev_cb
        diff = abs(sum_shap - raw_preds[i])
        residuals.append(diff)
    max_residual = max(residuals)
    print(f"  [VERIFICATION] TreeExplainer Additivity Max Residual: {max_residual:.4e} (Tolerance < 1e-4: {'PASSED' if max_residual < 1e-4 else 'FAILED'})")

    # Plot 1: Global Feature Importance Bar Plot
    print("\n[STEP 2] Generating Global Feature Importance Bar Plot...")
    mean_abs_shap = np.mean(np.abs(shap_vals_cb), axis=0)
    top_indices = np.argsort(-mean_abs_shap)[:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_indices))
    ax.barh(y_pos, mean_abs_shap[top_indices][::-1], color="#2563eb", align="center", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_indices[::-1]], fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Model Log-Odds)", fontsize=11, fontweight="bold")
    ax.set_title("Global Feature Importance (Research Extension — CatBoost Ensemble)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    bar_plot_path = os.path.join(shap_dir, "shap_global_bar.png")
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    print(f"  Saved: {bar_plot_path}")

    # Plot 2: Global Beeswarm Plot
    print("\n[STEP 3] Generating SHAP Beeswarm Plot...")
    plt.figure(figsize=(11, 7))
    explanation = shap.Explanation(
        values=shap_vals_cb,
        base_values=np.full(len(shap_vals_cb), ev_cb),
        data=X_te_scaled,
        feature_names=feature_names
    )
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("SHAP Beeswarm Distribution (Feature Impact on Turnover Log-Odds)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    beeswarm_path = os.path.join(shap_dir, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    print(f"  Saved: {beeswarm_path}")

    # Plot 3: Dependence Plot for OverTime & JobSatisfaction Interaction
    print("\n[STEP 4] Generating Feature Dependence Plots for HR Domain Features...")
    if "Overtime_JobSatisfaction_Interaction" in feature_names:
        feat_idx = feature_names.index("Overtime_JobSatisfaction_Interaction")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(X_te_fe["Overtime_JobSatisfaction_Interaction"], shap_vals_cb[:, feat_idx], alpha=0.6, c="#dc2626", edgecolors="none")
        ax.set_xlabel("Overtime_JobSatisfaction_Interaction (Raw Value)", fontsize=10, fontweight="bold")
        ax.set_ylabel("SHAP Attribution (Log-Odds Impact)", fontsize=10, fontweight="bold")
        ax.set_title("SHAP Dependence: Overtime × Job Dissatisfaction Interaction", fontsize=11, fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        dep_path1 = os.path.join(shap_dir, "shap_dependence_overtime_satisfaction.png")
        plt.savefig(dep_path1, dpi=300)
        plt.close()
        print(f"  Saved: {dep_path1}")

    # Plot 4: Dependence Plot for ManagerTenureRatio
    if "ManagerTenureRatio" in feature_names:
        feat_idx = feature_names.index("ManagerTenureRatio")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(X_te_fe["ManagerTenureRatio"], shap_vals_cb[:, feat_idx], alpha=0.6, c="#059669", edgecolors="none")
        ax.set_xlabel("ManagerTenureRatio (Years with Manager / Years at Company)", fontsize=10, fontweight="bold")
        ax.set_ylabel("SHAP Attribution (Log-Odds Impact)", fontsize=10, fontweight="bold")
        ax.set_title("SHAP Dependence: Supervisory Continuity vs Turnover Risk", fontsize=11, fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        dep_path2 = os.path.join(shap_dir, "shap_dependence_manager_tenure.png")
        plt.savefig(dep_path2, dpi=300)
        plt.close()
        print(f"  Saved: {dep_path2}")

    # Plot 5 & 6: Local Waterfall Explanations (High Risk vs Low Risk)
    print("\n[STEP 5] Generating Individual Local Waterfall Explanations...")
    # Find a clear High-Risk and Low-Risk case
    probs_cb = cb_model.predict_proba(X_te_scaled)[:, 1]
    high_idx = int(np.where((y_te == 1) & (probs_cb > 0.70))[0][0])
    low_idx = int(np.where((y_te == 0) & (probs_cb < 0.10))[0][0])

    # Waterfall High Risk
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_high = shap.Explanation(
        values=shap_vals_cb[high_idx],
        base_values=ev_cb,
        data=np.round(X_te_scaled[high_idx], 2),
        feature_names=feature_names
    )
    shap.plots.waterfall(exp_high, max_display=10, show=False)
    plt.title(f"Local Attribution: High-Risk Employee (Prob = {probs_cb[high_idx]:.2f}, Actual = Resigned)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    wf_high_path = os.path.join(shap_dir, "shap_waterfall_high_risk.png")
    plt.savefig(wf_high_path, dpi=300)
    plt.close()
    print(f"  Saved: {wf_high_path}")

    # Waterfall Low Risk
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_low = shap.Explanation(
        values=shap_vals_cb[low_idx],
        base_values=ev_cb,
        data=np.round(X_te_scaled[low_idx], 2),
        feature_names=feature_names
    )
    shap.plots.waterfall(exp_low, max_display=10, show=False)
    plt.title(f"Local Attribution: Low-Risk Employee (Prob = {probs_cb[low_idx]:.2f}, Actual = Retained)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    wf_low_path = os.path.join(shap_dir, "shap_waterfall_low_risk.png")
    plt.savefig(wf_low_path, dpi=300)
    plt.close()
    print(f"  Saved: {wf_low_path}")

    # Plot 7: Complete Hybrid Pipeline Model-Agnostic Explainer
    print("\n[STEP 6] Running Complete Hybrid Pipeline Explainer (Model-Agnostic in Probability Space)...")
    def full_pipeline_predict(X_mat: np.ndarray) -> np.ndarray:
        p_cb = cb_model.predict_proba(X_mat)[:, 1]
        p_lgb = lgb_model.predict_proba(X_mat)[:, 1]
        p_s27 = s27_model.predict_proba(X_mat)[:, 1]
        meta_feats = np.column_stack([
            p_cb, p_lgb, p_s27,
            (p_cb + p_lgb) / 2.0,
            np.maximum(p_cb, p_lgb),
            np.minimum(p_cb, p_lgb)
        ])
        return meta_lr.predict_proba(meta_feats)[:, 1]

    # Background reference using k-means/random sample from train
    bg_sample = shap.sample(X_tr_scaled, 50, random_state=42)
    kernel_explainer = shap.KernelExplainer(full_pipeline_predict, bg_sample)

    # Explain high risk case under complete pipeline
    case_mat = X_te_scaled[high_idx:high_idx + 1]
    kernel_shap_vals = kernel_explainer.shap_values(case_mat, nsamples=100)
    expected_prob = float(kernel_explainer.expected_value)
    actual_prob = float(full_pipeline_predict(case_mat)[0])

    exp_pipeline = shap.Explanation(
        values=kernel_shap_vals[0],
        base_values=expected_prob,
        data=np.round(case_mat[0], 2),
        feature_names=feature_names
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp_pipeline, max_display=10, show=False)
    plt.title(f"Complete Multi-Stage Pipeline Attribution (Output Space: Probability [0, 1], P = {actual_prob:.3f})", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    wf_pipe_path = os.path.join(shap_dir, "shap_complete_pipeline_waterfall.png")
    plt.savefig(wf_pipe_path, dpi=300)
    plt.close()
    print(f"  Saved: {wf_pipe_path}")

    # Summary report
    results = {
        "max_tree_additivity_residual": max_residual,
        "tree_additivity_status": "PASSED" if max_residual < 1e-4 else "FAILED",
        "expected_value_log_odds": ev_cb,
        "complete_pipeline_expected_prob": expected_prob,
        "high_risk_sample_actual_prob": actual_prob,
        "top_5_global_features": [feature_names[i] for i in top_indices[:5]],
        "plots_generated": [
            "shap_global_bar.png",
            "shap_beeswarm.png",
            "shap_dependence_overtime_satisfaction.png",
            "shap_dependence_manager_tenure.png",
            "shap_waterfall_high_risk.png",
            "shap_waterfall_low_risk.png",
            "shap_complete_pipeline_waterfall.png"
        ],
        "causal_disclaimer": "SHAP feature attributions describe mathematical predictive importance within the model architecture and do NOT establish real-world causal mechanisms for employee attrition."
    }

    report_path = os.path.join(shap_dir, "shap_explainability_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    generate_shap_visualizations()
