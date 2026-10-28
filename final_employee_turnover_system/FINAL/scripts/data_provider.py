"""
Data Provider for the Terminal-Driven Reporting Engine.
Safely extracts and aggregates all verified analytical data, experimental results,
and model metadata without modifying or retraining any model.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

# Path definitions relative to project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(CURRENT_DIR)
if os.path.exists(os.path.join(_parent, "data")):
    PROJECT_ROOT = _parent
elif os.path.basename(_parent) == "FINAL":
    PROJECT_ROOT = os.path.dirname(_parent)
else:
    PROJECT_ROOT = _parent

DATA_DIR = os.path.join(_parent, "data") if os.path.exists(os.path.join(_parent, "data")) else os.path.join(PROJECT_ROOT, "data")
CACHE_FILE = os.path.join(DATA_DIR, "evaluation_cache_594.json")
HOLDOUT_CSV = os.path.join(DATA_DIR, "holdout_594.csv")

p_ablation_res = os.path.join(PROJECT_ROOT, "research", "ablation", "ablation_results.csv")
p_ablation_leg = os.path.join(PROJECT_ROOT, "research_gap_extension", "ablation_results.csv")
ABLATION_CSV = p_ablation_res if os.path.exists(p_ablation_res) else p_ablation_leg


def load_cached_evaluation() -> Dict[str, Any]:
    """Loads verified 594-record holdout evaluation cache."""
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Evaluation cache file not found at: {CACHE_FILE}")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_holdout_ground_truth_and_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_scores, y_pred) arrays for the 594 holdout cases.
    Extracted directly from the verified evaluation cache.
    """
    cache = load_cached_evaluation()
    records = cache.get("records", [])
    if not records:
        raise ValueError("No records found in evaluation cache.")

    y_true = np.array([r["actual_attrition"] for r in records], dtype=int)
    y_scores = np.array([r["risk_score"] for r in records], dtype=float)
    y_pred = np.array([r["predicted_class"] for r in records], dtype=int)
    return y_true, y_scores, y_pred


def get_summary_metrics() -> Dict[str, Any]:
    """Returns official Step 32 summary metrics."""
    cache = load_cached_evaluation()
    return cache.get("summary_metrics", {})


def get_confusion_matrix_data() -> Dict[str, Any]:
    """Returns confusion matrix counts and percentages."""
    cache = load_cached_evaluation()
    return cache.get("confusion_matrix", {})


def get_risk_distribution_data() -> Dict[str, Any]:
    """Returns turnover risk distribution tiers."""
    cache = load_cached_evaluation()
    return cache.get("risk_distribution", {})


def get_global_shap_data() -> Dict[str, Any]:
    """Returns top 10 global SHAP feature importance attributions."""
    cache = load_cached_evaluation()
    return cache.get("global_shap", {})


def get_top_five_champions_data() -> List[Dict[str, Any]]:
    """Returns top 5 historical champions progression data."""
    # From step32_final.py CHAMPION_COMPARISON_DATA
    try:
        from step32_final import CHAMPION_COMPARISON_DATA
        return CHAMPION_COMPARISON_DATA.get("top_five_champions", [])
    except ImportError:
        # Fallback to verified frozen constants
        return [
            {
                "rank": 1,
                "version": "Step 32 Final Champion",
                "model_type": "Two-Stage Stacking Meta-Learner + Domain Specialist",
                "accuracy": 96.13,
                "precision": 88.42,
                "recall": 87.50,
                "f1_score": 87.96,
                "roc_auc": 94.56,
                "pr_auc": 86.89,
                "errors": 23,
                "status": "Official Production Champion",
                "is_final_champion": True,
                "dataset": "Locked Legacy Holdout (N=594)",
                "confusion_matrix": {"TN": 487, "FP": 11, "FN": 12, "TP": 84}
            },
            {
                "rank": 2,
                "version": "Step 31 Candidate v2",
                "model_type": "Step 28 Backbone + Precision False-Negative Boundary Recovery",
                "accuracy": 95.96,
                "precision": 87.50,
                "recall": 87.50,
                "f1_score": 87.50,
                "roc_auc": 94.55,
                "pr_auc": 86.10,
                "errors": 24,
                "status": "Verified Precursor Champion",
                "is_final_champion": False,
                "dataset": "Locked Legacy Holdout (N=594)",
                "confusion_matrix": {"TN": 486, "FP": 12, "FN": 12, "TP": 84}
            },
            {
                "rank": 3,
                "version": "Step 30 Severe-FN Recovery",
                "model_type": "Step 28 Dual Meta-Learner + Severe False-Negative Recovery Rule",
                "accuracy": 95.79,
                "precision": 88.17,
                "recall": 85.42,
                "f1_score": 86.77,
                "roc_auc": 94.55,
                "pr_auc": 85.75,
                "errors": 25,
                "status": "Verified Precursor Champion",
                "is_final_champion": False,
                "dataset": "Locked Legacy Holdout (N=594)",
                "confusion_matrix": {"TN": 487, "FP": 11, "FN": 14, "TP": 82}
            },
            {
                "rank": 4,
                "version": "Step 28 Dual Meta-Learner",
                "model_type": "Logistic Meta-Learner over CatBoost + LightGBM + Step 27 Anchor",
                "accuracy": 95.62,
                "precision": 88.04,
                "recall": 84.38,
                "f1_score": 86.17,
                "roc_auc": 94.55,
                "pr_auc": 85.40,
                "errors": 26,
                "status": "Verified Precursor Champion",
                "is_final_champion": False,
                "dataset": "Locked Legacy Holdout (N=594)",
                "confusion_matrix": {"TN": 487, "FP": 11, "FN": 15, "TP": 81}
            },
            {
                "rank": 5,
                "version": "Review-2 / Stage 6 Primary Champion",
                "model_type": "Tuned Single-Model CatBoost (Depth 6 / Balanced)",
                "accuracy": 94.95,
                "precision": 87.50,
                "recall": 80.21,
                "f1_score": 83.70,
                "roc_auc": 92.32,
                "pr_auc": 81.12,
                "errors": 30,
                "status": "Verified Foundational Champion",
                "is_final_champion": False,
                "dataset": "Locked Legacy Holdout (N=594)",
                "confusion_matrix": {"TN": 487, "FP": 11, "FN": 19, "TP": 77}
            }
        ]


def get_base_paper_comparison_data() -> Dict[str, Any]:
    """Returns verified base paper comparison figures (Talebi et al. 2025 vs Step 32)."""
    try:
        from step32_final import CHAMPION_COMPARISON_DATA
        return {
            "published_base_paper": CHAMPION_COMPARISON_DATA.get("published_base_paper", {}),
            "base_paper_on_our_dataset": CHAMPION_COMPARISON_DATA.get("base_paper_on_our_dataset", {}),
            "official_final_champion": CHAMPION_COMPARISON_DATA.get("official_final_champion", {})
        }
    except ImportError:
        return {
            "published_base_paper": {
                "title": "Developing a hybrid machine learning model for employee turnover prediction (Talebi et al. 2025)",
                "sample_size": "N = 2,932 records",
                "metrics": {
                    "accuracy": 92.00,
                    "precision": 89.00,
                    "recall": 85.00,
                    "f1_score": 87.00,
                    "roc_auc": 94.00,
                    "pr_auc": None,
                    "total_errors": None
                }
            },
            "base_paper_on_our_dataset": {
                "experiment_id": "Experiment B (GA + LightGBM Transfer)",
                "sample_size": "N = 2,969 unique profiles",
                "metrics": {
                    "accuracy": 88.01,
                    "precision": 74.80,
                    "recall": 38.96,
                    "f1_score": 51.23,
                    "roc_auc": 87.11,
                    "pr_auc": 66.13,
                    "total_errors": 356,
                    "error_rate": 11.99
                },
                "confusion_matrix": {"TN": 2426, "FP": 63, "FN": 293, "TP": 187}
            },
            "official_final_champion": {
                "version": "Step 32 Final Champion",
                "sample_size": "N = 594 records",
                "accuracy": 96.13,
                "precision": 88.42,
                "recall": 87.50,
                "f1_score": 87.96,
                "roc_auc": 94.56,
                "pr_auc": 86.89,
                "total_errors": 23,
                "error_rate": 3.87,
                "confusion_matrix": {"TN": 487, "FP": 11, "FN": 12, "TP": 84}
            }
        }


def get_ablation_results() -> pd.DataFrame:
    """Returns ablation study dataframe from research_gap_extension/ablation_results.csv."""
    if not os.path.exists(ABLATION_CSV):
        raise FileNotFoundError(f"Ablation results file not found at: {ABLATION_CSV}")
    return pd.read_csv(ABLATION_CSV)


def get_fairness_audit_data() -> Dict[str, Any]:
    """Returns fairness audit metrics, subgroup attrition rates, and ethical rules."""
    try:
        from step32_final import BENCHMARK_DATA
        return BENCHMARK_DATA.get("fairness_audit", {})
    except ImportError:
        return {
            "disparate_impact_ratio": 0.951,
            "four_fifths_rule_compliant": True,
            "subgroups": [
                {"group": "Female", "count": 234, "attrition_rate": 0.145, "wilson_ci_95": [0.106, 0.197]},
                {"group": "Male", "count": 360, "attrition_rate": 0.172, "wilson_ci_95": [0.136, 0.215]},
                {"group": "Age < 30", "count": 89, "attrition_rate": 0.326, "wilson_ci_95": [0.237, 0.430]},
                {"group": "Age 30-45", "count": 352, "attrition_rate": 0.134, "wilson_ci_95": [0.102, 0.174]},
                {"group": "Age > 45", "count": 153, "attrition_rate": 0.118, "wilson_ci_95": [0.076, 0.178]}
            ],
            "ethical_governance_rules": [
                "Prohibition of Automated Adverse Employment Actions: Model predictions shall never trigger automatic termination, demotion, salary reduction, or promotion denial.",
                "Mandatory Human-in-the-Loop Review: High-risk flags serve strictly as decision support signals for structured engagement conversations.",
                "Subgroup Sample Limitations: Subgroups with N < 50 require elevated caution due to wide statistical confidence intervals.",
                "Non-Punitive Intent: Interventions must focus on supportive resources, not punitive tracking."
            ]
        }


_HOLDOUT_DF: Optional[pd.DataFrame] = None


def get_holdout_dataframe() -> pd.DataFrame:
    """Returns the cached authoritative 594-record holdout dataframe."""
    global _HOLDOUT_DF
    if _HOLDOUT_DF is None:
        if not os.path.exists(HOLDOUT_CSV):
            raise FileNotFoundError(f"Holdout CSV file not found at: {HOLDOUT_CSV}")
        _HOLDOUT_DF = pd.read_csv(HOLDOUT_CSV)
    return _HOLDOUT_DF


def compute_record_fingerprint(raw_row: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 fingerprint for a holdout record."""
    import hashlib
    fp_fields = [
        str(raw_row.get("Holdout_ID", "")),
        str(raw_row.get("Age", "")),
        str(raw_row.get("MonthlyIncome", "")),
        str(raw_row.get("TotalWorkingYears", "")),
        str(raw_row.get("YearsAtCompany", "")),
        str(raw_row.get("JobRole", "")),
        str(raw_row.get("Department", "")),
        str(raw_row.get("Attrition", ""))
    ]
    fp_str = "|".join(fp_fields)
    return hashlib.sha256(fp_str.encode("utf-8")).hexdigest()[:12]


def get_single_employee_record(identifier: Any) -> Optional[Dict[str, Any]]:
    """
    Looks up a single employee record from the holdout evaluation cache.
    Deterministically resolves both synthetic reference IDs (e.g. 'EMP_HD_043')
    and integer indices (0-593), cross-validating against data/holdout_594.csv.
    Attaches authoritative raw attributes, CSV row numbers, and cryptographic fingerprint.
    """
    cache = load_cached_evaluation()
    records = cache.get("records", [])
    df = get_holdout_dataframe()

    target_idx: Optional[int] = None
    target_ref: Optional[str] = None

    if isinstance(identifier, int):
        if 0 <= identifier < len(records):
            target_idx = identifier
            target_ref = f"EMP_HD_{identifier + 1:03d}"
    else:
        str_id = str(identifier).strip().upper()
        # Case A: Synthetic reference format like 'EMP_HD_043'
        if str_id.startswith("EMP_HD_"):
            try:
                num = int(str_id.replace("EMP_HD_", ""))
                if 1 <= num <= len(records):
                    target_idx = num - 1
                    target_ref = str_id
            except ValueError:
                pass

        # Case B: Direct integer string index like '42'
        if target_idx is None:
            try:
                idx = int(str_id)
                if 0 <= idx < len(records):
                    target_idx = idx
                    target_ref = f"EMP_HD_{idx + 1:03d}"
            except ValueError:
                pass

        # Case C: Fallback linear search by employee_reference
        if target_idx is None:
            for i, r in enumerate(records):
                if r.get("employee_reference", "").upper() == str_id:
                    target_idx = i
                    target_ref = r.get("employee_reference")
                    break

    if target_idx is None or not (0 <= target_idx < len(records)):
        return None

    record = dict(records[target_idx])

    # Cross-validate against authoritative holdout CSV
    if 0 <= target_idx < len(df):
        raw_row = df.iloc[target_idx].to_dict()
        csv_id = str(raw_row.get("Holdout_ID", ""))
        expected_ref = target_ref or record.get("employee_reference", "")
        if csv_id != expected_ref:
            # Locate by Holdout_ID column directly if ordering ever differed
            matched_df = df[df["Holdout_ID"] == expected_ref]
            if not matched_df.empty:
                raw_row = matched_df.iloc[0].to_dict()
                target_idx = int(matched_df.index[0])

        record["raw_row"] = raw_row
        record["record_index"] = target_idx
        record["employee_reference"] = str(raw_row.get("Holdout_ID", expected_ref))
        record["csv_row_number"] = target_idx + 2  # 1-indexed, accounting for header
        record["row_fingerprint"] = compute_record_fingerprint(raw_row)

    return record
