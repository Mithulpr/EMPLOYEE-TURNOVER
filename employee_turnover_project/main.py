"""
Employee Turnover Prediction Research CLI
=========================================
Project Title: A Multi-stage Hybrid Approach for Employee Turnover Prediction
               using Conditional Error Correction

Usage:
  python main.py --mode validate
  python main.py --mode predict --input data/holdout_594.csv --output outputs/employee_predictions.csv
  python main.py --mode evaluate --input data/holdout_594.csv
  python main.py --mode risk-distribution --input data/holdout_594.csv
  python main.py --mode shap-global
  python main.py --mode shap-local --employee-id EMP_HD_001
  python main.py --mode ablation
  python main.py --mode champion-progression
  python main.py --mode run-all
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.validation import validate_system_integrity
from src.predict import predict_bulk_file, predict_single_employee
from src.evaluate import evaluate_holdout
from src.risk import compute_risk_distribution
from src.shap_analysis import get_global_shap_ranking, get_local_shap_explanation
from src.ablation import load_ablation_results
from src.champion_progression import load_champion_progression
from src.visualization import plot_all_figures, plot_performance_metrics, plot_confusion_matrix, plot_risk_tier_distribution


def print_banner():
    print("=" * 76)
    print(" A Multi-stage Hybrid Approach for Employee Turnover Prediction")
    print("           using Conditional Error Correction (Step 32)")
    print("=" * 76)


def run_validate():
    print_banner()
    print("[1/1] Validating System Artifacts and Schema Integrity...")
    status, issues = validate_system_integrity()
    if status:
        print("[PASS] All 11 model artifacts, schemas, and configurations verified.")
    else:
        print("[FAIL] Integrity verification failed:")
        for iss in issues:
            print(f"  - {iss}")
        sys.exit(1)


def run_predict(input_path: str, output_path: str, tau: float):
    print_banner()
    print(f"[*] Mode: Bulk Inference | Input: {input_path} | Operating Threshold: {tau}")
    df_pred = predict_bulk_file(input_path, output_path, tau=tau)
    print(f"[OK] Generated {len(df_pred)} employee turnover predictions.")
    print(df_pred[["Employee_ID", "Turnover_Score", "Binary_Decision", "Risk_Tier"]].head(10))
    if output_path:
        print(f"[OK] Predictions saved to: {output_path}")


def run_evaluate(input_path: str, tau: float):
    print_banner()
    print(f"[*] Mode: Retrospective Holdout Evaluation | Cohort: {input_path}")
    print(f"[*] Operating Decision Threshold: TAU = {tau:.4f}")
    print("=" * 76)
    
    metrics, df_preds = evaluate_holdout(input_path, tau=tau)
    
    cm = metrics["confusion_matrix"]
    print("\n--- RETROSPECTIVE PERFORMANCE METRICS (N = 594) ---")
    print(f"Accuracy         : {metrics['accuracy_pct']}%  (Target: 96.13%)")
    print(f"Precision        : {metrics['precision_pct']}%  (Target: 88.42%)")
    print(f"Recall           : {metrics['recall_pct']}%  (Target: 87.50%)")
    print(f"F1-Score         : {metrics['f1_score_pct']}%  (Target: 87.96%)")
    print(f"ROC-AUC          : {metrics['roc_auc_pct']}%  (Target: 94.55%)")
    print(f"PR-AUC           : {metrics['pr_auc_pct']}%  (Target: 86.89%)")
    print(f"Total Correct    : {metrics['total_correct']} / 594  (Target: 571)")
    print(f"Total Errors     : {metrics['total_errors']} / 594  (Target: 23)")
    print("\n--- CONFUSION MATRIX ---")
    print(f"True Negatives  (TN) : {cm['TN']}  (Stayers correctly identified)")
    print(f"False Positives (FP) : {cm['FP']}  (False alarms)")
    print(f"False Negatives (FN) : {cm['FN']}  (Missed leavers)")
    print(f"True Positives  (TP) : {cm['TP']}  (Leavers correctly identified)")
    print("=" * 76)

    # Save outputs
    out_preds = os.path.join(CURRENT_DIR, "outputs", "employee_predictions.csv")
    df_preds.to_csv(out_preds, index=False)
    print(f"[OK] Saved row-level predictions: {out_preds}")

    out_metrics = os.path.join(CURRENT_DIR, "outputs", "metrics_summary.json")
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved metrics summary: {out_metrics}")


def run_risk(input_path: str, tau: float):
    print_banner()
    print(f"[*] Mode: Operational Risk-Tier Distribution | Threshold: {tau}")
    df_holdout = pd.read_csv(input_path)
    from src.pipeline import MultiStageHybridPipeline
    pipe = MultiStageHybridPipeline(threshold=tau)
    df_preds = pipe.predict(df_holdout)

    df_risk = compute_risk_distribution(df_preds["Turnover_Score"].values, tau=tau)
    print("\n--- OPERATIONAL RISK-TIER DISTRIBUTION ---")
    for _, row in df_risk.iterrows():
        print(f"  {row['Risk_Tier']:<10}: {row['Count']:>3} employees ({row['Percentage']:>5.2f}%)")
    print("-" * 40)
    print(f"  Total     : {df_risk['Count'].sum():>3} employees ({df_risk['Percentage'].sum():>5.2f}%)")

    out_risk = os.path.join(CURRENT_DIR, "outputs", "risk_distribution.csv")
    df_risk.to_csv(out_risk, index=False)
    print(f"[OK] Saved risk distribution: {out_risk}")


def run_shap_global():
    print_banner()
    print("[*] Mode: Global TreeSHAP Feature Importance (Base Ensemble, N = 594)")
    df_shap = get_global_shap_ranking()
    print("\n--- TOP 15 GLOBAL SHAP FEATURE RANKING ---")
    for _, row in df_shap.iterrows():
        print(f"  {row['Rank']:>2}. {row['Feature_Name']:<34}: {row['Mean_Absolute_SHAP']:.4f}")

    out_csv = os.path.join(CURRENT_DIR, "outputs", "shap", "global_shap_ranking_15.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_shap.to_csv(out_csv, index=False)
    print(f"[OK] Saved ranking to: {out_csv}")


def run_shap_local(emp_id: str):
    print_banner()
    print(f"[*] Mode: Local SHAP Decomposition | Employee: {emp_id}")
    rec = get_local_shap_explanation(emp_id)
    print(f"\nEmployee Reference   : {rec.get('employee_reference', emp_id)}")
    print(f"Actual Outcome       : {rec.get('actual_outcome', 'N/A')}")
    print(f"Predicted Outcome    : {rec.get('predicted_outcome', 'N/A')}")
    print(f"Model Risk Score     : {rec.get('risk_score', 0.0) * 100:.2f}%")
    print(f"Risk Tier            : {rec.get('risk_level', 'N/A')}")
    print(f"Evaluation Category  : {rec.get('evaluation_result', 'N/A')}")

    print("\nTop Positive Attributions (Push Factors):")
    for feat in rec.get("shap_top_positive", []):
        print(f"  (+) {feat['feature_name']:<35} : {feat['shap_value']:+.4f}")

    print("\nTop Negative Attributions (Retention Anchors):")
    for feat in rec.get("shap_top_negative", []):
        print(f"  (-) {feat['feature_name']:<35} : {feat['shap_value']:+.4f}")


def run_ablation():
    print_banner()
    print("[*] Mode: Nested Cross-Validation Feature Ablation Experiment")
    df_abl = load_ablation_results()
    print(df_abl.to_string(index=False))


def run_champion():
    print_banner()
    print("[*] Mode: Historical Development Champion Progression")
    df_champ = load_champion_progression()
    print(df_champ.to_string(index=False))


def run_all():
    run_validate()
    h_path = os.path.join(CURRENT_DIR, "data", "holdout_594.csv")
    run_evaluate(h_path, tau=0.361)
    run_risk(h_path, tau=0.361)
    run_shap_global()
    run_shap_local("EMP_HD_001")
    run_ablation()
    run_champion()
    print("\n[*] Generating all 8 publication-grade figures...")
    paths = plot_all_figures()
    print("[OK] Generated all figures successfully:")
    for k, p in paths.items():
        print(f"  - {k}: {p}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Stage Hybrid Employee Turnover Prediction System CLI")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["validate", "predict", "evaluate", "risk-distribution", "shap-global", "shap-local", "ablation", "champion-progression", "run-all"],
                        help="Operating mode")
    parser.add_argument("--input", type=str, default=os.path.join(CURRENT_DIR, "data", "holdout_594.csv"), help="Input CSV path")
    parser.add_argument("--output", type=str, default=os.path.join(CURRENT_DIR, "outputs", "employee_predictions.csv"), help="Output CSV path")
    parser.add_argument("--employee-id", type=str, default="EMP_HD_001", help="Employee reference ID for local SHAP")
    parser.add_argument("--threshold", type=float, default=0.361, help="Operating turnover decision threshold (default: 0.361)")

    args = parser.parse_args()

    if args.mode == "validate":
        run_validate()
    elif args.mode == "predict":
        run_predict(args.input, args.output, args.threshold)
    elif args.mode == "evaluate":
        run_evaluate(args.input, args.threshold)
    elif args.mode == "risk-distribution":
        run_risk(args.input, args.threshold)
    elif args.mode == "shap-global":
        run_shap_global()
    elif args.mode == "shap-local":
        run_shap_local(args.employee_id)
    elif args.mode == "ablation":
        run_ablation()
    elif args.mode == "champion-progression":
        run_champion()
    elif args.mode == "run-all":
        run_all()


if __name__ == "__main__":
    main()
