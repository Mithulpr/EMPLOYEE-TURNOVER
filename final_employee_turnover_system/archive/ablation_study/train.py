"""
Research Gap Extension: Leakage-Safe Training & Cross-Validation Engine
========================================================================
Implements:
1. 5-Fold Stratified Nested Cross-Validation across all 2,969 dataset records.
2. Leakage-safe inner fold OOF generation for stacking meta-learner.
3. Inner threshold calibration without access to outer validation labels.
4. Three-arm controlled ablation comparison:
   - Arm A: Baseline Step 32 representation (30 features)
   - Arm B: Enriched representation (+ all 10 candidate HR domain features)
   - Arm C: Selected representation (+ top screened HR domain features)
5. Final model artifact serialization into research_gap_extension/models/
"""

import os
import sys
import time
import json
import joblib
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Workspace path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.dirname(CURRENT_DIR)
BASE_DIR = os.path.dirname(SYSTEM_DIR)

for p in [CURRENT_DIR, SYSTEM_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_and_validate_raw_data, clean_and_harmonize_dataset
try:
    from research_gap_extension.features import (
        HRFeatureEngineer,
        STEP32_BASELINE_FEATURES,
        CANDIDATE_HR_FEATURES
    )
except ImportError:
    from features import (
        HRFeatureEngineer,
        STEP32_BASELINE_FEATURES,
        CANDIDATE_HR_FEATURES
    )

RANDOM_SEED = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "specificity": float(spec),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "total_errors": int(fp + fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def fit_hybrid_fold(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_val_df: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: List[str]
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, Any]:
    """
    Fits multi-stage hybrid architecture strictly on X_train_df and evaluates on X_val_df.
    Zero data leakage: scaler, base models, inner OOF, meta-learner, and threshold are fitted on X_train_df.
    """
    # 1. Feature scaling
    scaler = StandardScaler()
    X_tr_mat = scaler.fit_transform(X_train_df[feature_cols].fillna(0.0))
    X_val_mat = scaler.transform(X_val_df[feature_cols].fillna(0.0))

    # 2. Inner K-Fold for leakage-safe meta-features
    inner_skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_cb = np.zeros(len(X_tr_mat))
    oof_lgb = np.zeros(len(X_tr_mat))
    oof_s27 = np.zeros(len(X_tr_mat))

    for in_tr_idx, in_val_idx in inner_skf.split(X_tr_mat, y_train):
        X_in_tr, y_in_tr = X_tr_mat[in_tr_idx], y_train[in_tr_idx]
        X_in_val = X_tr_mat[in_val_idx]

        # Inner models
        m_cb = CatBoostClassifier(iterations=350, learning_rate=0.04, depth=5, verbose=0, random_seed=RANDOM_SEED)
        m_cb.fit(X_in_tr, y_in_tr)
        oof_cb[in_val_idx] = m_cb.predict_proba(X_in_val)[:, 1]

        m_lgb = LGBMClassifier(n_estimators=150, learning_rate=0.04, max_depth=5, verbose=-1, random_state=RANDOM_SEED)
        m_lgb.fit(X_in_tr, y_in_tr)
        oof_lgb[in_val_idx] = m_lgb.predict_proba(X_in_val)[:, 1]

        m_s27 = CatBoostClassifier(iterations=250, learning_rate=0.05, depth=4, verbose=0, random_seed=RANDOM_SEED + 10)
        m_s27.fit(X_in_tr, y_in_tr)
        oof_s27[in_val_idx] = m_s27.predict_proba(X_in_val)[:, 1]

    # Meta-features for training meta-learner
    X_meta_tr = np.column_stack([
        oof_cb, oof_lgb, oof_s27,
        (oof_cb + oof_lgb) / 2.0,
        np.maximum(oof_cb, oof_lgb),
        np.minimum(oof_cb, oof_lgb)
    ])

    meta_lr = LogisticRegression(C=0.85, max_iter=500, random_state=RANDOM_SEED)
    meta_lr.fit(X_meta_tr, y_train)
    p_meta_oof = meta_lr.predict_proba(X_meta_tr)[:, 1]

    # Inner threshold tuning (maximize F1 without seeing outer validation labels)
    best_tau = 0.361
    best_f1 = -1.0
    for cand_tau in np.linspace(0.25, 0.50, 26):
        cand_pred = (p_meta_oof >= cand_tau).astype(int)
        score = f1_score(y_train, cand_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_tau = float(cand_tau)

    # 3. Fit base models on full outer training fold
    full_cb = CatBoostClassifier(iterations=450, learning_rate=0.04, depth=5, verbose=0, random_seed=RANDOM_SEED)
    full_cb.fit(X_tr_mat, y_train)

    full_lgb = LGBMClassifier(n_estimators=200, learning_rate=0.04, max_depth=5, verbose=-1, random_state=RANDOM_SEED)
    full_lgb.fit(X_tr_mat, y_train)

    full_s27 = CatBoostClassifier(iterations=350, learning_rate=0.05, depth=4, verbose=0, random_seed=RANDOM_SEED + 10)
    full_s27.fit(X_tr_mat, y_train)

    # 4. Out-of-fold inference on outer validation set
    val_cb = full_cb.predict_proba(X_val_mat)[:, 1]
    val_lgb = full_lgb.predict_proba(X_val_mat)[:, 1]
    val_s27 = full_s27.predict_proba(X_val_mat)[:, 1]

    X_meta_val = np.column_stack([
        val_cb, val_lgb, val_s27,
        (val_cb + val_lgb) / 2.0,
        np.maximum(val_cb, val_lgb),
        np.minimum(val_cb, val_lgb)
    ])
    p_meta_val = meta_lr.predict_proba(X_meta_val)[:, 1]
    pred_val = (p_meta_val >= best_tau).astype(int)

    # 5. Fit recovery specialist on training boundary subset
    bnd_mask = (p_meta_oof >= 0.15) & (p_meta_oof <= 0.45)
    if bnd_mask.sum() > 20:
        spec_cb = CatBoostClassifier(iterations=150, learning_rate=0.03, depth=3, verbose=0, random_seed=RANDOM_SEED + 20)
        spec_cb.fit(X_tr_mat[bnd_mask], y_train[bnd_mask])

        val_bnd_mask = (p_meta_val >= 0.15) & (p_meta_val <= best_tau)
        if val_bnd_mask.sum() > 0:
            p_spec_val = spec_cb.predict_proba(X_val_mat[val_bnd_mask])[:, 1]
            override_idx = np.where(val_bnd_mask)[0][p_spec_val >= 0.50]
            pred_val[override_idx] = 1
    else:
        spec_cb = None

    metrics = compute_metrics(y_val, pred_val, p_meta_val)
    metrics["best_tau"] = best_tau

    model_bundle = {
        "scaler": scaler,
        "full_cb": full_cb,
        "full_lgb": full_lgb,
        "full_s27": full_s27,
        "meta_lr": meta_lr,
        "spec_cb": spec_cb,
        "tau": best_tau,
        "feature_cols": feature_cols
    }

    return metrics, pred_val, p_meta_val, model_bundle


def run_cross_validation_experiment() -> Dict[str, Any]:
    print("=" * 80)
    print("STARTING REPRODUCIBLE LEAKAGE-SAFE CROSS-VALIDATION EXPERIMENT")
    print("Addressing Research Gaps in Talebi et al. (2025)")
    print("=" * 80)

    start_time = time.time()
    df_raw, diag = load_and_validate_raw_data()
    df_clean = clean_and_harmonize_dataset(df_raw)

    y_all = df_clean["Attrition"].values
    X_raw_all = df_clean.drop(columns=["Attrition"])

    skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Storage for results across folds
    results_arm_a = []
    results_arm_b = []
    results_arm_c = []

    engineer_b = HRFeatureEngineer(feature_set="baseline")
    engineer_e = HRFeatureEngineer(feature_set="enriched")
    engineer_s = HRFeatureEngineer(feature_set="selected")

    fold_idx = 1
    for tr_idx, val_idx in skf.split(X_raw_all, y_all):
        print(f"\n--- Running Outer Fold {fold_idx} / {N_OUTER_FOLDS} ---")
        X_tr_raw = X_raw_all.iloc[tr_idx].copy()
        y_tr = y_all[tr_idx]
        X_val_raw = X_raw_all.iloc[val_idx].copy()
        y_val = y_all[val_idx]

        # 1. Transform features strictly fitting on X_tr_raw
        fe_base = HRFeatureEngineer(feature_set="baseline").fit(X_tr_raw)
        X_tr_base = fe_base.transform(X_tr_raw)
        X_val_base = fe_base.transform(X_val_raw)
        cols_a = fe_base.get_feature_names()

        fe_enr = HRFeatureEngineer(feature_set="enriched").fit(X_tr_raw)
        X_tr_enr = fe_enr.transform(X_tr_raw)
        X_val_enr = fe_enr.transform(X_val_raw)
        cols_b = fe_enr.get_feature_names()

        fe_sel = HRFeatureEngineer(feature_set="selected").fit(X_tr_raw)
        X_tr_sel = fe_sel.transform(X_tr_raw)
        X_val_sel = fe_sel.transform(X_val_raw)
        cols_c = fe_sel.get_feature_names()

        # Arm A: Step 32 Baseline
        m_a, _, _, _ = fit_hybrid_fold(X_tr_base, y_tr, X_val_base, y_val, cols_a)
        results_arm_a.append(m_a)
        print(f"  [Arm A - Step 32 Baseline]    Acc: {m_a['accuracy']*100:.2f}%, F1: {m_a['f1']*100:.2f}%, ROC-AUC: {m_a['roc_auc']*100:.2f}%, Errors: {m_a['total_errors']}")

        # Arm B: Enriched All HR Features
        m_b, _, _, _ = fit_hybrid_fold(X_tr_enr, y_tr, X_val_enr, y_val, cols_b)
        results_arm_b.append(m_b)
        print(f"  [Arm B - Enriched All HR]     Acc: {m_b['accuracy']*100:.2f}%, F1: {m_b['f1']*100:.2f}%, ROC-AUC: {m_b['roc_auc']*100:.2f}%, Errors: {m_b['total_errors']}")

        # Arm C: Selected HR Features
        m_c, _, _, _ = fit_hybrid_fold(X_tr_sel, y_tr, X_val_sel, y_val, cols_c)
        results_arm_c.append(m_c)
        print(f"  [Arm C - Selected Screened]   Acc: {m_c['accuracy']*100:.2f}%, F1: {m_c['f1']*100:.2f}%, ROC-AUC: {m_c['roc_auc']*100:.2f}%, Errors: {m_c['total_errors']}")

        fold_idx += 1

    total_cv_time = time.time() - start_time
    print(f"\n[SUCCESS] 5-Fold Cross-Validation completed in {total_cv_time:.2f} seconds.")

    # Aggregate summaries
    def aggregate_arm(res_list):
        agg = {}
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity", "balanced_accuracy", "total_errors"]:
            vals = [r[k] for r in res_list]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        agg["total_errors_sum"] = int(np.sum([r["total_errors"] for r in res_list]))
        return agg

    agg_a = aggregate_arm(results_arm_a)
    agg_b = aggregate_arm(results_arm_b)
    agg_c = aggregate_arm(results_arm_c)

    print("\n" + "=" * 80)
    print("5-FOLD NESTED CROSS-VALIDATION SUMMARY RESULTS (N = 2,969 Records)")
    print("=" * 80)
    print(f"Arm A (Step 32 Baseline):     Acc = {agg_a['accuracy_mean']*100:.2f}% ± {agg_a['accuracy_std']*100:.2f}%, F1 = {agg_a['f1_mean']*100:.2f}% ± {agg_a['f1_std']*100:.2f}%, Total Errors = {agg_a['total_errors_sum']}")
    print(f"Arm B (Enriched All HR):      Acc = {agg_b['accuracy_mean']*100:.2f}% ± {agg_b['accuracy_std']*100:.2f}%, F1 = {agg_b['f1_mean']*100:.2f}% ± {agg_b['f1_std']*100:.2f}%, Total Errors = {agg_b['total_errors_sum']}")
    print(f"Arm C (Selected HR Features): Acc = {agg_c['accuracy_mean']*100:.2f}% ± {agg_c['accuracy_std']*100:.2f}%, F1 = {agg_c['f1_mean']*100:.2f}% ± {agg_c['f1_std']*100:.2f}%, Total Errors = {agg_c['total_errors_sum']}")

    # Build Ablation DataFrame
    df_ablation = pd.DataFrame([
        {
            "Arm": "A_Step32_Baseline",
            "Features_Count": len(engineer_b.get_feature_names()),
            "Accuracy_Mean": round(agg_a["accuracy_mean"] * 100, 2),
            "Accuracy_Std": round(agg_a["accuracy_std"] * 100, 2),
            "F1_Mean": round(agg_a["f1_mean"] * 100, 2),
            "F1_Std": round(agg_a["f1_std"] * 100, 2),
            "ROC_AUC_Mean": round(agg_a["roc_auc_mean"] * 100, 2),
            "PR_AUC_Mean": round(agg_a["pr_auc_mean"] * 100, 2),
            "Precision_Mean": round(agg_a["precision_mean"] * 100, 2),
            "Recall_Mean": round(agg_a["recall_mean"] * 100, 2),
            "Total_Errors_Across_Folds": agg_a["total_errors_sum"]
        },
        {
            "Arm": "B_Enriched_All_HR",
            "Features_Count": len(engineer_e.get_feature_names()),
            "Accuracy_Mean": round(agg_b["accuracy_mean"] * 100, 2),
            "Accuracy_Std": round(agg_b["accuracy_std"] * 100, 2),
            "F1_Mean": round(agg_b["f1_mean"] * 100, 2),
            "F1_Std": round(agg_b["f1_std"] * 100, 2),
            "ROC_AUC_Mean": round(agg_b["roc_auc_mean"] * 100, 2),
            "PR_AUC_Mean": round(agg_b["pr_auc_mean"] * 100, 2),
            "Precision_Mean": round(agg_b["precision_mean"] * 100, 2),
            "Recall_Mean": round(agg_b["recall_mean"] * 100, 2),
            "Total_Errors_Across_Folds": agg_b["total_errors_sum"]
        },
        {
            "Arm": "C_Selected_HR_Features",
            "Features_Count": len(engineer_s.get_feature_names()),
            "Accuracy_Mean": round(agg_c["accuracy_mean"] * 100, 2),
            "Accuracy_Std": round(agg_c["accuracy_std"] * 100, 2),
            "F1_Mean": round(agg_c["f1_mean"] * 100, 2),
            "F1_Std": round(agg_c["f1_std"] * 100, 2),
            "ROC_AUC_Mean": round(agg_c["roc_auc_mean"] * 100, 2),
            "PR_AUC_Mean": round(agg_c["pr_auc_mean"] * 100, 2),
            "Precision_Mean": round(agg_c["precision_mean"] * 100, 2),
            "Recall_Mean": round(agg_c["recall_mean"] * 100, 2),
            "Total_Errors_Across_Folds": agg_c["total_errors_sum"]
        }
    ])

    ablation_csv_path = os.path.join(CURRENT_DIR, "ablation_results.csv")
    df_ablation.to_csv(ablation_csv_path, index=False)
    print(f"\nSaved ablation table to {ablation_csv_path}")

    # Now train the final experimental candidate on the standard training partition (2,375 records)
    # and save model artifacts into research_gap_extension/models/
    print("\n--- Training Final Candidate Models on Development Split (2,375 records) ---")
    from sklearn.model_selection import train_test_split
    X_tr_dev, X_te_holdout, y_tr_dev, y_te_holdout = train_test_split(
        X_raw_all, y_all, test_size=0.20, random_state=RANDOM_SEED, stratify=y_all
    )

    final_engineer = HRFeatureEngineer(feature_set="selected").fit(X_tr_dev)
    X_tr_trans = final_engineer.transform(X_tr_dev)
    X_te_trans = final_engineer.transform(X_te_holdout)
    final_cols = final_engineer.get_feature_names()

    m_final, p_final_te, prob_final_te, final_bundle = fit_hybrid_fold(
        X_tr_trans, y_tr_dev, X_te_trans, y_te_holdout, final_cols
    )
    print(f"Final Candidate Retrospective Holdout (594 Records):")
    print(f"  Accuracy: {m_final['accuracy']*100:.2f}%, F1: {m_final['f1']*100:.2f}%, Total Errors: {m_final['total_errors']}")

    # Save artifacts into research_gap_extension/models/
    models_dir = os.path.join(CURRENT_DIR, "models")
    joblib.dump(final_engineer, os.path.join(models_dir, "hr_feature_engineer.joblib"))
    joblib.dump(final_bundle["scaler"], os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(final_bundle["full_cb"], os.path.join(models_dir, "catboost_base.joblib"))
    joblib.dump(final_bundle["full_lgb"], os.path.join(models_dir, "lightgbm_base.joblib"))
    joblib.dump(final_bundle["full_s27"], os.path.join(models_dir, "anchor_catboost.joblib"))
    joblib.dump(final_bundle["meta_lr"], os.path.join(models_dir, "meta_logistic_regression.joblib"))
    if final_bundle["spec_cb"] is not None:
        joblib.dump(final_bundle["spec_cb"], os.path.join(models_dir, "boundary_specialist.joblib"))

    schema = {
        "architecture": "Research Gap Extension: Domain-Engineered Hybrid Pipeline",
        "feature_set": "selected",
        "feature_count": len(final_cols),
        "features": final_cols,
        "decision_threshold": final_bundle["tau"],
        "random_seed": RANDOM_SEED,
        "base_paper": "Talebi et al. (2025)",
        "hash_manifest_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(models_dir, "pipeline_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Save metrics JSON and CSV
    metrics_path_json = os.path.join(CURRENT_DIR, "metrics.json")
    metrics_path_csv = os.path.join(CURRENT_DIR, "metrics.csv")

    complete_metrics = {
        "nested_cross_validation_5fold": {
            "Arm_A_Step32_Baseline": agg_a,
            "Arm_B_Enriched_All_HR": agg_b,
            "Arm_C_Selected_HR_Features": agg_c
        },
        "retrospective_holdout_evaluation_594": m_final,
        "timing_seconds": {
            "cv_duration_seconds": round(total_cv_time, 2)
        }
    }
    with open(metrics_path_json, "w", encoding="utf-8") as f:
        json.dump(complete_metrics, f, indent=2)

    df_metrics = pd.DataFrame([
        {"Metric": "Accuracy (%)", "Nested_CV_Arm_A": f"{agg_a['accuracy_mean']*100:.2f} ± {agg_a['accuracy_std']*100:.2f}", "Nested_CV_Arm_C": f"{agg_c['accuracy_mean']*100:.2f} ± {agg_c['accuracy_std']*100:.2f}", "Retrospective_Holdout": f"{m_final['accuracy']*100:.2f}"},
        {"Metric": "F1-Score (%)", "Nested_CV_Arm_A": f"{agg_a['f1_mean']*100:.2f} ± {agg_a['f1_std']*100:.2f}", "Nested_CV_Arm_C": f"{agg_c['f1_mean']*100:.2f} ± {agg_c['f1_std']*100:.2f}", "Retrospective_Holdout": f"{m_final['f1']*100:.2f}"},
        {"Metric": "Precision (%)", "Nested_CV_Arm_A": f"{agg_a['precision_mean']*100:.2f} ± {agg_a['precision_std']*100:.2f}", "Nested_CV_Arm_C": f"{agg_c['precision_mean']*100:.2f} ± {agg_c['precision_std']*100:.2f}", "Retrospective_Holdout": f"{m_final['precision']*100:.2f}"},
        {"Metric": "Recall (%)", "Nested_CV_Arm_A": f"{agg_a['recall_mean']*100:.2f} ± {agg_a['recall_std']*100:.2f}", "Nested_CV_Arm_C": f"{agg_c['recall_mean']*100:.2f} ± {agg_c['recall_std']*100:.2f}", "Retrospective_Holdout": f"{m_final['recall']*100:.2f}"},
        {"Metric": "ROC-AUC (%)", "Nested_CV_Arm_A": f"{agg_a['roc_auc_mean']*100:.2f} ± {agg_a['roc_auc_std']*100:.2f}", "Nested_CV_Arm_C": f"{agg_c['roc_auc_mean']*100:.2f} ± {agg_c['roc_auc_std']*100:.2f}", "Retrospective_Holdout": f"{m_final['roc_auc']*100:.2f}"},
        {"Metric": "Total Errors", "Nested_CV_Arm_A": str(agg_a["total_errors_sum"]), "Nested_CV_Arm_C": str(agg_c["total_errors_sum"]), "Retrospective_Holdout": str(m_final["total_errors"])}
    ])
    df_metrics.to_csv(metrics_path_csv, index=False)

    return complete_metrics


if __name__ == "__main__":
    run_cross_validation_experiment()
