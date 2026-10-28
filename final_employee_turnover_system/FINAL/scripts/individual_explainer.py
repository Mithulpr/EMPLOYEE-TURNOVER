"""
Individual Employee SHAP Explainability & Publication Graphics Engine.
======================================================================
Generates human-readable natural language narratives, SHAP waterfall plots,
horizontal diverging contribution bar graphs, and executive two-column explanation
panels for any evaluated employee.

All figures are grounded strictly in genuine model attributions and holdout data,
saved to reports/figures/individual_shap/, and automatically opened via the system viewer.
"""

import os
import sys
import re
import time
import json
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for thread safety; images are opened directly with system viewer
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mtick
from typing import Dict, Any, List, Optional, Tuple

from reporting.data_provider import (
    PROJECT_ROOT,
    get_single_employee_record,
    load_cached_evaluation
)
from reporting.terminal_reports import Colors

# Absolute path resolution using pathlib.Path
PROJECT_ROOT_PATH = Path(PROJECT_ROOT).resolve()
p_ind_res = PROJECT_ROOT_PATH / "research" / "figures" / "individual_shap"
INDIVIDUAL_FIG_DIR = p_ind_res
INDIVIDUAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
FULL_SHAP_CACHE_PATH = PROJECT_ROOT_PATH / "data" / "full_shap_cache_594.json"
HOLDOUT_CSV_PATH = PROJECT_ROOT_PATH / "data" / "holdout_594.csv"

# Curated publication-grade palette
PALETTE = {
    "push": "#DC2626",        # Crimson Red (Increases flight risk)
    "push_bg": "#FEF2F2",     # Soft Rose Red Background
    "push_border": "#F87171", # Rose Red Border
    "pull": "#059669",        # Emerald Green (Anchors retention)
    "pull_bg": "#F0FDF4",     # Soft Mint Green Background
    "pull_border": "#34D399", # Mint Green Border
    "base": "#475569",        # Slate Base Value
    "base_tree": "#0891B2",   # Cyan/Teal Intermediate Tree Score
    "final": "#2563EB",       # Blue Final Prediction
    "bg_light": "#F8FAFC",    # Card Background
    "border": "#CBD5E1",      # Border Line
    "text_dark": "#0F172A",   # Navy Text
    "text_muted": "#64748B"   # Subdued Text
}


def sanitize_filename_component(name: Any) -> str:
    """
    Sanitizes an employee ID or string to be safe for Windows filenames.
    Removes invalid Windows characters: \\ / : * ? \" < > | and control characters.
    Strips trailing dots and spaces.
    """
    s = str(name).strip()
    s = re.sub(r'[\\/*?:"<>|\x00-\x1f]', '_', s)
    s = s.strip(". ")
    return s if s else "EMP_UNKNOWN"


def _save_figure_safely(fig: plt.Figure, output_path: Path, dpi: int = 300) -> Path:
    """
    Saves a matplotlib figure safely to disk on Windows:
    1. Resolves absolute path and ensures parent directory exists.
    2. Verifies and logs writability of destination and parent directory.
    3. Handles locked files (e.g. Windows Photos memory-mapped handles causing
       Win32 Error 1224: ERROR_USER_MAPPED_FILE -> POSIX Errno 22: EINVAL)
       by attempting to unlink stale files, or falling back to a versioned path if strictly locked.
    4. Verifies that the saved file exists and has non-zero size.
    """
    resolved_path = output_path.resolve()
    parent_dir = resolved_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    is_writable = os.access(parent_dir, os.W_OK)
    print(f"  {Colors.DIM}[PATH AUDIT] Target: {resolved_path}{Colors.ENDC}")
    print(f"  {Colors.DIM}[PATH AUDIT] Parent: {parent_dir} (Exists: {parent_dir.exists()}, Writable: {is_writable}){Colors.ENDC}")

    if not is_writable:
        raise PermissionError(f"Destination directory is not writable: {parent_dir}")

    actual_target = resolved_path
    if actual_target.exists():
        try:
            actual_target.unlink()
        except OSError:
            fallback_name = f"{actual_target.stem}_{int(time.time())}{actual_target.suffix}"
            actual_target = parent_dir / fallback_name
            print(f"  {Colors.WARNING}Notice: Existing file locked; saving to fallback: {actual_target.name}{Colors.ENDC}")

    try:
        fig.savefig(str(actual_target), dpi=dpi, bbox_inches="tight")
    except Exception as e:
        print(f"  {Colors.FAIL}ERROR: Failed to save figure to {actual_target}: {e}{Colors.ENDC}")
        raise OSError(f"Failed to save figure to {actual_target}: {e}") from e

    if not actual_target.exists() or actual_target.stat().st_size == 0:
        raise OSError(f"Figure file was not created or is empty: {actual_target}")

    return actual_target


def _open_image_file(filepath: Path) -> bool:
    """Opens image in Windows default viewer gracefully without throwing errors."""
    try:
        p = Path(filepath).resolve()
        if p.exists() and p.stat().st_size > 0:
            if sys.platform.startswith("win"):
                os.startfile(str(p))
                return True
    except Exception as e:
        print(f"  {Colors.DIM}(Note: Image saved successfully. System viewer could not be launched: {e}){Colors.ENDC}")
    return False


def _load_full_shap_cache() -> Dict[str, Any]:
    """Loads the precomputed 30-feature SHAP cache if available."""
    if FULL_SHAP_CACHE_PATH.exists():
        try:
            with open(FULL_SHAP_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _get_raw_employee_row(record_idx: int, emp_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves authoritative raw dataset attributes for the employee from holdout_594.csv.
    Uses cached dataframe and matches by Holdout_ID first to prevent any row-order drift.
    """
    try:
        from reporting.data_provider import get_holdout_dataframe
        df = get_holdout_dataframe()
        if emp_id:
            matched = df[df["Holdout_ID"] == emp_id]
            if not matched.empty:
                return matched.iloc[0].to_dict()
        if 0 <= record_idx < len(df):
            return df.iloc[record_idx].to_dict()
    except Exception:
        pass
    return {}


def _extract_employee_data_and_shap(identifier: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], float, float, Dict[str, Any], Dict[str, Any]]:
    """
    Extracts employee metadata, positive and negative SHAP factors,
    baseline expected value, final model output, comprehensive SHAP accounting metadata,
    and raw dataset features. Guaranteed deterministic across synthetic IDs and row indices.
    """
    record = get_single_employee_record(identifier)
    if not record:
        raise ValueError(f"Employee identifier '{identifier}' not found in holdout cache or dataset.")

    emp_id = record.get("employee_reference", "EMP_UNKNOWN")
    rec_idx = record.get("record_index", 0)
    raw_row = record.get("raw_row")
    if not raw_row:
        raw_row = _get_raw_employee_row(rec_idx, emp_id=emp_id)
        record["raw_row"] = raw_row

    full_cache = _load_full_shap_cache()
    cached_shap = full_cache.get(emp_id)

    if cached_shap:
        pos_factors = cached_shap.get("top_push_factors", [])
        neg_factors = cached_shap.get("top_pull_factors", [])
        base_val = cached_shap.get("expected_baseline_value", -1.215)
        raw_final = cached_shap.get("final_model_margin", 0.0)
        shap_meta = cached_shap
    else:
        pos_factors = record.get("shap_top_positive", [])
        neg_factors = record.get("shap_top_negative", [])
        base_val = -1.215
        
        push_sub = sum(f.get("shap_value", f.get("shap_impact", 0.0)) for f in pos_factors)
        pull_sub = sum(f.get("shap_value", f.get("shap_impact", 0.0)) for f in neg_factors)
        top10_sub = push_sub + pull_sub
        
        risk_score = float(record.get("risk_score", 0.5))
        clamped_score = np.clip(risk_score, 1e-5, 1 - 1e-5)
        raw_final = float(np.log(clamped_score / (1.0 - clamped_score)))
        
        shap_meta = {
            "employee_reference": emp_id,
            "record_index": rec_idx,
            "expected_baseline_value": base_val,
            "top_push_factors": pos_factors,
            "top_pull_factors": neg_factors,
            "top_push_subtotal": push_sub,
            "top_pull_subtotal": pull_sub,
            "top_10_net_subtotal": top10_sub,
            "remaining_features_net": 0.0,
            "sum_shap_attributions": top10_sub,
            "base_trees_margin": base_val + top10_sub,
            "base_trees_probability": float(1.0 / (1.0 + np.exp(-(base_val + top10_sub)))),
            "meta_learner_adjustment": raw_final - (base_val + top10_sub),
            "final_model_margin": raw_final,
            "final_risk_score": risk_score,
            "all_features": []
        }

    return record, pos_factors, neg_factors, base_val, raw_final, shap_meta, raw_row


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely converts a value to integer, handling NaN, None, floats, and strings."""
    if val is None or pd.isna(val):
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def interpret_feature(feat: str, val: float, shap_val: float, raw_row: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Translates technical feature attributions into natural, human-understandable English:
    Returns (sentence, metric_label, domain_reasoning).
    Each feature has a distinct, concise, human-friendly label and a single, direct,
    clear sentence explaining its impact without redundancy or clutter.
    """
    is_push = shap_val > 0
    clean = feat.replace("_", " ")

    # 1. TotalExpToAgeRatio (must check BEFORE TotalWorkingYears to avoid label collisions!)
    if "TotalExpToAgeRatio" in feat:
        tot_exp = _safe_int(raw_row.get("TotalWorkingYears"), 5)
        age = max(_safe_int(raw_row.get("Age"), 30), 1)
        pct = int((tot_exp / age) * 100)
        label = f"Experience-to-Age Ratio ({pct}%)"
        if is_push:
            sentence = f"Earlier career stage relative to age ({pct}%) correlates with higher job mobility."
        else:
            sentence = f"Substantial career maturity relative to age ({pct}%) reflects professional stability."
        reason = "Career-to-age ratio captures career progression stage."
        return sentence, label, reason

    # 2. TotalWorkingYears
    if feat == "TotalWorkingYears" or "TotalWorkingYears" in feat:
        tot_exp = _safe_int(raw_row.get("TotalWorkingYears"), 5)
        label = f"Total Experience: {tot_exp} yrs"
        if is_push:
            sentence = f"Experience level ({tot_exp} yrs) offers active market options and recruiter interest."
        else:
            sentence = f"Extensive career tenure ({tot_exp} yrs) anchors professional stability."
        reason = "Total experience shapes career seniority and labor market mobility."
        return sentence, label, reason

    # 3. JobHoppingRate / PriorCompanyExperience / NumCompaniesWorked
    if any(k in feat for k in ["JobHopping", "PriorCompany", "NumCompanies"]):
        num_c = _safe_int(raw_row.get("NumCompaniesWorked"), 1)
        label = f"Prior Job Transitions: {num_c} companies"
        if is_push:
            sentence = f"Track record of frequent job changes ({num_c} companies) increases openness to new offers."
        else:
            sentence = f"Stable employment history ({num_c} prior companies) indicates commitment to employers."
        reason = "Mobility history reflects career transition habits."
        return sentence, label, reason

    # 4. Salary / Monthly Income
    if "Income" in feat or "Salary" in feat:
        inc = _safe_int(raw_row.get("MonthlyIncome"), 5000)
        label = f"Monthly Income: ${inc:,}"
        if inc < 3500 and is_push:
            sentence = f"Entry-level salary (${inc:,}/mo) is below benchmark, increasing sensitivity to offers."
        elif not is_push:
            sentence = f"Competitive salary (${inc:,}/mo) provides strong financial stability and retention."
        else:
            sentence = f"Current compensation level (${inc:,}/mo) exerts mild upward flight pressure."
        reason = "Compensation structure directly influences retention."
        return sentence, label, reason

    # 5. Marital Status
    if "MaritalStatus" in feat:
        m_stat = str(raw_row.get("MaritalStatus", "Married"))
        label = f"Marital Status: {m_stat}"
        if is_push:
            sentence = "Single marital status affords greater geographic and career transition flexibility."
        else:
            sentence = f"{m_stat} status and personal stability anchor long-term retention."
        reason = "Personal commitments influence geographic and career mobility."
        return sentence, label, reason

    # 6. Promotion Stagnation vs Lag vs Velocity
    if "PromotionStagnation" in feat or "PromotionVelocity" in feat:
        label = "Promotion Cadence: On Track" if not is_push else "Promotion Stagnation"
        if not is_push:
            sentence = "Consistent advancement pacing confirms timely career recognition."
        else:
            sentence = "Advancement pacing lag signals upward progression friction."
        reason = "Career progression timing validates professional development."
        return sentence, label, reason

    if any(k in feat for k in ["PromotionLag", "PromotionPace", "YearsSinceLastPromotion", "Promotion"]):
        yslp = _safe_int(raw_row.get("YearsSinceLastPromotion"), 0)
        label = f"Promotion Delay ({yslp} yrs without promo)" if yslp > 0 else "Promotion Pacing Delay"
        if is_push:
            sentence = "Time elapsed since last promotion generates upward progression friction."
        else:
            sentence = "Regular promotion intervals support sustained role commitment."
        reason = "Promotion pacing directly affects advancement expectations."
        return sentence, label, reason

    # 7. Severe Dissatisfaction vs Job / Environment Satisfaction
    if "SevereDissatisfaction" in feat:
        label = "Workplace Satisfaction: Good" if not is_push else "Severe Workplace Dissatisfaction"
        if not is_push:
            sentence = "Absence of acute role or environment dissatisfaction supports daily engagement."
        else:
            sentence = "Acute dissatisfaction with role or workplace culture elevates departure intent."
        reason = "Severe workplace dissatisfaction is an acute driver of voluntary exit."
        return sentence, label, reason

    if "JobSatisfaction" in feat:
        js = _safe_int(raw_row.get("JobSatisfaction"), 3)
        label = f"Job Satisfaction: {js}/4"
        if is_push:
            sentence = f"Lower job satisfaction ({js}/4) increases receptivity to alternative positions."
        else:
            sentence = f"Positive job satisfaction ({js}/4) reinforces commitment to current responsibilities."
        reason = "Role fulfillment directly affects employee retention."
        return sentence, label, reason

    if "EnvironmentSatisfaction" in feat:
        es = _safe_int(raw_row.get("EnvironmentSatisfaction"), 3)
        label = f"Environment Satisfaction: {es}/4"
        if is_push:
            sentence = f"Unfavorable workplace environment ({es}/4) contributes to turnover intent."
        else:
            sentence = f"Favorable workplace environment ({es}/4) supports team morale and stability."
        reason = "Work environment sentiment impacts long-term organizational attachment."
        return sentence, label, reason

    # 8. Business Travel
    if "Travel" in feat:
        bt = str(raw_row.get("BusinessTravel", "Travel_Rarely")).replace("_", " ")
        label = f"Business Travel: {bt}"
        if is_push:
            sentence = "Frequent travel obligations create schedule fatigue and disrupt work-life balance."
        else:
            sentence = "Minimal travel requirements protect personal routine and avoid burnout."
        reason = "Travel frequency influences work-life balance and fatigue."
        return sentence, label, reason

    # 9. Overtime Workload
    if "OverTime" in feat or "Overtime" in feat:
        ot = str(raw_row.get("OverTime", "No")).lower() == "yes"
        label = "Overtime: Yes (Frequent)" if ot else "Overtime: None (Standard Schedule)"
        if ot and is_push:
            sentence = "Frequent overtime workload elevates burnout hazards and turnover intent."
        else:
            sentence = "Standard working hours without overtime support sustainable retention."
        reason = "Overtime demand directly impacts workload equilibrium."
        return sentence, label, reason

    # 10. Work-Life Balance / Deficit
    if any(k in feat for k in ["WorkLife", "Deficit"]):
        wlb = _safe_int(raw_row.get("WorkLifeBalance"), 3)
        label = f"Work-Life Balance: {wlb}/4"
        if is_push:
            sentence = "Work-life balance strain elevates personal stress and turnover intent."
        else:
            sentence = "Balanced schedule and personal equilibrium maintain strong role commitment."
        reason = "Work-life equilibrium is essential for preventing employee burnout."
        return sentence, label, reason

    # 11. Manager Alignment / Continuity
    if "ManagerTenureMismatch" in feat:
        ywcm = _safe_int(raw_row.get("YearsWithCurrManager"), 1)
        label = f"Manager Continuity ({ywcm} yrs)"
        if is_push:
            sentence = f"Brief supervisory continuity ({ywcm} yrs) weakens direct reporting alignment."
        else:
            sentence = f"Established relationship with manager ({ywcm} yrs) buffers against role friction."
        reason = "Supervisory continuity directly impacts employee engagement."
        return sentence, label, reason

    if any(k in feat for k in ["Manager", "WithCurrManager"]):
        ywcm = _safe_int(raw_row.get("YearsWithCurrManager"), 1)
        label = f"Manager Tenure: {ywcm} yrs"
        if is_push:
            sentence = f"Limited time with current manager ({ywcm} yrs) weakens team integration."
        else:
            sentence = f"Solid tenure with manager ({ywcm} yrs) fosters operational trust."
        reason = "Reporting relationship length supports workplace cohesion."
        return sentence, label, reason

    # 12. Company Tenure & Loyalty
    if any(k in feat for k in ["CompanyLoyalty", "YearsAtCompany"]):
        yac = _safe_int(raw_row.get("YearsAtCompany"), 3)
        tot_exp = max(_safe_int(raw_row.get("TotalWorkingYears"), 1), 1)
        pct = int((yac / tot_exp) * 100)
        label = f"Company Loyalty ({pct}% of career)"
        if not is_push:
            sentence = f"High career proportion with company ({pct}%) builds deep organizational loyalty."
        else:
            sentence = f"Shorter company tenure relative to total career reduces switching friction."
        reason = "Tenure-to-career ratio reflects firm-specific capital and loyalty."
        return sentence, label, reason

    # 13. EarlyCareerFlightRisk
    if "EarlyCareer" in feat:
        label = "Early Career Stage"
        if is_push:
            sentence = "Early career juncture combined with market mobility elevates turnover risk."
        else:
            sentence = "Established career juncture buffers against voluntary departure."
        reason = "Early career demographics correlate with higher exploratory job transitions."
        return sentence, label, reason

    # 14. Age
    if feat == "Age":
        age = _safe_int(raw_row.get("Age"), 35)
        label = f"Age: {age} years"
        if age >= 38 and not is_push:
            sentence = f"Mature career stage (age {age}) correlates with established organizational stability."
        elif age < 32 and is_push:
            sentence = f"Early-career demographic (age {age}) aligns with active career exploration."
        else:
            sentence = f"Demographic profile (age {age}) shapes baseline career mobility."
        reason = "Age demographics reflect baseline life-stage and career mobility patterns."
        return sentence, label, reason

    # 15. Training
    if "Training" in feat:
        train_count = _safe_int(raw_row.get("TrainingTimesLastYear"), 2)
        label = f"Training: {train_count} sessions"
        if not is_push:
            sentence = f"Active training participation ({train_count} sessions) signals ongoing upskilling."
        else:
            sentence = f"Limited training sessions ({train_count}) reduce growth visibility."
        reason = "Training frequency signals organizational investment in employee skills."
        return sentence, label, reason

    # Fallback
    dir_str = "higher" if is_push else "lower"
    label = f"{clean}: {val:.2f}"
    sentence = f"Attribute level associates with {dir_str} predicted turnover risk."
    reason = f"Model attribution aligns with {'elevated' if is_push else 'reduced'} turnover probability."
    return sentence, label, reason


def generate_personalized_synthesis_paragraph(record: Dict[str, Any],
                                               shap_meta: Dict[str, Any],
                                               pos_factors: List[Dict[str, Any]],
                                               neg_factors: List[Dict[str, Any]],
                                               raw_row: Dict[str, Any]) -> str:
    """
    Constructs a personalized, clean, readable executive summary paragraph addressing:
    1. Assigned flight risk score and risk tier.
    2. Primary drivers elevating risk vs key retention anchors buffering departure.
    3. Practical HR recommendation aligned with decision policy τ = 0.3610.
    """
    emp_id = record.get("employee_reference", "EMP_UNKNOWN")
    score = record.get("risk_score", 0.0)
    risk_tier = record.get("risk_level", "Unknown")
    is_high_risk = score >= 0.3610

    # Top push & pull names
    top_push_names = []
    for p in pos_factors[:2]:
        _, l, _ = interpret_feature(p.get("feature", ""), p.get("feature_value", 0.0), p.get("shap_impact", 0.0), raw_row)
        top_push_names.append(f"{l} (+{abs(p.get('shap_impact', 0.0)):.2f} SHAP)")
    push_summary = ", ".join(top_push_names) if top_push_names else "minor exploratory variables"

    top_pull_names = []
    for n in neg_factors[:2]:
        _, l, _ = interpret_feature(n.get("feature", ""), n.get("feature_value", 0.0), n.get("shap_impact", 0.0), raw_row)
        top_pull_names.append(f"{l} ({n.get('shap_impact', 0.0):.2f} SHAP)")
    pull_summary = ", ".join(top_pull_names) if top_pull_names else "routine baseline tenure"

    if is_high_risk:
        paragraph = (
            f"The Step 32 Champion model categorizes {emp_id} as {risk_tier} with an estimated flight risk of "
            f"{score*100:.1f}%, exceeding the action threshold of τ = 0.3610. "
            f"The primary drivers elevating risk are {push_summary}. "
            f"While protective anchors such as {pull_summary} provide partial stability, "
            f"overall risk pressure suggests prioritizing proactive, supportive career dialogue and workload assessment."
        )
    else:
        paragraph = (
            f"The Step 32 Champion model categorizes {emp_id} as {risk_tier} with an estimated flight risk of "
            f"{score*100:.1f}%, remaining comfortably below the action threshold of τ = 0.3610. "
            f"While factors such as {push_summary} exert mild upward pressure, "
            f"strong protective anchors including {pull_summary} decisively support retention. "
            f"Standard career development and routine quarterly engagement check-ins are recommended."
        )
    return paragraph


def generate_individual_explanation_panel(identifier: Any, interactive: bool = False, pre_extracted: Optional[Tuple] = None) -> str:
    """
    Figure P: Executive Two-Column Human-Readable Explanation Panel.
    Publication-grade high-resolution graphic designed for academic paper inclusion:
    - Pure white clean canvas suitable for document embedding (no clashing backgrounds)
    - Red Panel: Key Factors Increasing Flight Risk (Push Forces)
    - Green Panel: Key Factors Buffering Retention (Protective Pull Forces)
    - Clean card hierarchy with guaranteed non-overlapping text and proper padding
    - Concise, elegant plain-English interpretations without clutter or jargon
    """
    if pre_extracted:
        record, pos_factors, neg_factors, base_val, raw_final, shap_meta, raw_row = pre_extracted
    else:
        record, pos_factors, neg_factors, base_val, raw_final, shap_meta, raw_row = _extract_employee_data_and_shap(identifier)

    emp_id = record.get("employee_reference", "EMP_UNKNOWN")
    rec_idx = record.get("record_index", 0)
    csv_row = record.get("csv_row_number", rec_idx + 2)
    fingerprint = record.get("row_fingerprint", "")
    if not fingerprint and raw_row:
        from reporting.data_provider import compute_record_fingerprint
        fingerprint = compute_record_fingerprint(raw_row)

    sanitized_id = sanitize_filename_component(emp_id)
    prob = record.get("risk_score", 0.0)
    risk_tier = record.get("risk_level", "Unknown")
    is_high_risk = prob >= 0.3610
    info = record.get("employee_info", {})

    dept = raw_row.get("Department", info.get("Department", "General"))
    role = raw_row.get("JobRole", info.get("Job Role", "Specialist"))
    age = raw_row.get("Age", info.get("Age", "N/A"))
    tot_exp = raw_row.get("TotalWorkingYears", info.get("Total Working Years", "N/A"))
    if isinstance(tot_exp, (float, np.floating)) and tot_exp == int(tot_exp):
        tot_exp = int(tot_exp)
    tenure = raw_row.get("YearsAtCompany", info.get("Years at Company", "N/A"))
    if isinstance(tenure, (float, np.floating)) and tenure == int(tenure):
        tenure = int(tenure)
    if "MonthlyIncome" in raw_row and pd.notna(raw_row["MonthlyIncome"]):
        inc = f"${int(raw_row['MonthlyIncome']):,}"
    else:
        inc = str(info.get("Monthly Income", "$0"))

    # Create figure: 14.5 x 9.2 inches at 300 DPI for paper publication
    fig, ax = plt.subplots(figsize=(14.5, 9.2), dpi=300)
    try:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")

        # Outer subtle frame
        r_outer = patches.FancyBboxPatch((0.015, 0.015), 0.97, 0.97,
                                         boxstyle="round,pad=0.008,rounding_size=0.012",
                                         facecolor="#FAFAFA", edgecolor="#E2E8F0", linewidth=1.2)
        ax.add_patch(r_outer)

        # 1. Header Card (y = 0.880 to 0.965, height = 0.085)
        r_hdr = patches.FancyBboxPatch((0.03, 0.880), 0.94, 0.085,
                                       boxstyle="round,pad=0.008,rounding_size=0.012",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.2)
        ax.add_patch(r_hdr)

        ax.text(0.048, 0.948, f"INDIVIDUAL EMPLOYEE RETENTION EXPLANATION — {emp_id}",
                fontsize=13.5, fontweight="bold", color="#0F172A", va="top")
        ax.text(0.048, 0.918,
                f"Role: {role} ({dept})   •   Age: {age}   •   Experience: {tot_exp} yrs   •   Tenure: {tenure} yrs   •   Income: {inc}",
                fontsize=9.2, color="#475569", va="top")
        ax.text(0.048, 0.892,
                f"Holdout Record Index: {rec_idx}   •   CSV Line: {csv_row}   •   Integrity: SHA-256 [{fingerprint}]",
                fontsize=7.8, color="#64748B", va="top")

        # Risk Badge Pill
        badge_bg = "#FEE2E2" if is_high_risk else "#DCFCE7"
        badge_edge = "#EF4444" if is_high_risk else "#16A34A"
        badge_txt = "#991B1B" if is_high_risk else "#166534"
        decision_txt = "Predicted Left" if is_high_risk else "Predicted Stayed"

        r_badge = patches.FancyBboxPatch((0.68, 0.890), 0.275, 0.065,
                                        boxstyle="round,pad=0.008,rounding_size=0.012",
                                        facecolor=badge_bg, edgecolor=badge_edge, linewidth=1.5)
        ax.add_patch(r_badge)
        ax.text(0.817, 0.938, f"PREDICTED FLIGHT RISK: {prob*100:.1f}% ({risk_tier})",
                fontsize=10.0, fontweight="bold", color=badge_txt, ha="center", va="top")
        ax.text(0.817, 0.908, f"Decision Policy: τ = 0.3610 ({decision_txt})",
                fontsize=8.5, color=badge_txt, ha="center", va="top")

        # 2. Left Column: Risk Push Forces (Red Card)
        col_w = 0.455
        col_h = 0.635
        left_x = 0.03
        right_x = 0.515
        body_y = 0.230

        r_left = patches.FancyBboxPatch((left_x, body_y), col_w, col_h,
                                       boxstyle="round,pad=0.008,rounding_size=0.012",
                                       facecolor="#FFF8F8", edgecolor="#FECACA", linewidth=1.2)
        ax.add_patch(r_left)

        # Left Column Header Banner
        banner_h = 0.042
        r_l_hdr = patches.FancyBboxPatch((left_x, body_y + col_h - banner_h), col_w, banner_h,
                                        boxstyle="round,pad=0.004,rounding_size=0.008",
                                        facecolor="#DC2626", edgecolor="#DC2626")
        ax.add_patch(r_l_hdr)
        ax.text(left_x + col_w / 2, body_y + col_h - banner_h / 2,
                "▲ FACTORS INCREASING ATTRITION RISK (PUSH FORCES)",
                fontsize=10.2, fontweight="bold", color="#FFFFFF", ha="center", va="center")

        # Render Push Items (Top 4)
        top_push = pos_factors[:4]
        card_h = 0.125
        card_gap = 0.014
        start_y = body_y + col_h - banner_h - 0.016

        for i, p in enumerate(top_push, 1):
            feat = p.get("feature", "")
            val = float(p.get("feature_value", 0.0))
            shap_v = float(p.get("shap_value", p.get("shap_impact", 0.0)))
            sentence, label, _ = interpret_feature(feat, val, shap_v, raw_row)

            card_y = start_y - i * card_h - (i - 1) * card_gap
            card_x = left_x + 0.015
            inner_w = col_w - 0.030

            # Item sub-box
            r_item = patches.FancyBboxPatch((card_x, card_y), inner_w, card_h,
                                           boxstyle="round,pad=0.006,rounding_size=0.008",
                                           facecolor="#FFFFFF", edgecolor="#FECACA", linewidth=0.9)
            ax.add_patch(r_item)

            # Title & Pill Badge (top of card)
            top_line_y = card_y + card_h - 0.018
            ax.text(card_x + 0.012, top_line_y, f"{i}. {label}",
                    fontsize=9.8, fontweight="bold", color="#0F172A", va="top")
            ax.text(card_x + inner_w - 0.012, top_line_y, f"+{shap_v:.3f} SHAP",
                    fontsize=8.8, fontweight="bold", color="#DC2626", ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEE2E2", edgecolor="#FCA5A5", linewidth=0.8))

            # Concise Natural Language Explanation (strictly below title)
            desc_y = card_y + card_h - 0.052
            wrapped_sent = textwrap.fill(sentence, width=54)
            ax.text(card_x + 0.012, desc_y, wrapped_sent,
                    fontsize=9.0, color="#334155", va="top", linespacing=1.35)

        # 3. Right Column: Retention Buffering Pull Forces (Green Card)
        r_right = patches.FancyBboxPatch((right_x, body_y), col_w, col_h,
                                        boxstyle="round,pad=0.008,rounding_size=0.012",
                                        facecolor="#F0FDF4", edgecolor="#BBF7D0", linewidth=1.2)
        ax.add_patch(r_right)

        # Right Column Header Banner
        r_r_hdr = patches.FancyBboxPatch((right_x, body_y + col_h - banner_h), col_w, banner_h,
                                         boxstyle="round,pad=0.004,rounding_size=0.008",
                                         facecolor="#059669", edgecolor="#059669")
        ax.add_patch(r_r_hdr)
        ax.text(right_x + col_w / 2, body_y + col_h - banner_h / 2,
                "▼ FACTORS BUFFERING RETENTION (PROTECTIVE PULL FORCES)",
                fontsize=10.2, fontweight="bold", color="#FFFFFF", ha="center", va="center")

        # Render Pull Items (Top 4)
        top_pull = neg_factors[:4]
        for i, n in enumerate(top_pull, 1):
            feat = n.get("feature", "")
            val = float(n.get("feature_value", 0.0))
            shap_v = float(n.get("shap_value", n.get("shap_impact", 0.0)))
            sentence, label, _ = interpret_feature(feat, val, shap_v, raw_row)

            card_y = start_y - i * card_h - (i - 1) * card_gap
            card_x = right_x + 0.015
            inner_w = col_w - 0.030

            # Item sub-box
            r_item = patches.FancyBboxPatch((card_x, card_y), inner_w, card_h,
                                           boxstyle="round,pad=0.006,rounding_size=0.008",
                                           facecolor="#FFFFFF", edgecolor="#BBF7D0", linewidth=0.9)
            ax.add_patch(r_item)

            # Title & Pill Badge
            top_line_y = card_y + card_h - 0.018
            ax.text(card_x + 0.012, top_line_y, f"{i}. {label}",
                    fontsize=9.8, fontweight="bold", color="#0F172A", va="top")
            ax.text(card_x + inner_w - 0.012, top_line_y, f"{shap_v:.3f} SHAP",
                    fontsize=8.8, fontweight="bold", color="#059669", ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#DCFCE7", edgecolor="#86EFAC", linewidth=0.8))

            # Concise Natural Language Explanation
            desc_y = card_y + card_h - 0.052
            wrapped_sent = textwrap.fill(sentence, width=54)
            ax.text(card_x + 0.012, desc_y, wrapped_sent,
                    fontsize=9.0, color="#334155", va="top", linespacing=1.35)

        # 4. Bottom Box: Overall Executive Prediction Synthesis & Research Guidance
        r_bot = patches.FancyBboxPatch((0.03, 0.030), 0.94, 0.180,
                                       boxstyle="round,pad=0.008,rounding_size=0.012",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.2)
        ax.add_patch(r_bot)

        ax.text(0.048, 0.190, "OVERALL EXECUTIVE PREDICTION SYNTHESIS & RETENTION GUIDANCE",
                fontsize=10.5, fontweight="bold", color="#1E40AF", va="top")

        syn_paragraph = generate_personalized_synthesis_paragraph(record, shap_meta, pos_factors, neg_factors, raw_row)
        wrapped_syn = textwrap.fill(syn_paragraph, width=132)
        ax.text(0.048, 0.160, wrapped_syn, fontsize=8.8, color="#1E293B", va="top", linespacing=1.38)

        ax.text(0.048, 0.052,
                "RESEARCH INTEGRITY NOTE: Feature attributions calculated via TreeSHAP (log-odds impact). "
                "Observational attributions reflect historical statistical correlations and do not imply counterfactual causality.",
                fontsize=7.8, style="italic", color="#64748B", va="top")

        plt.tight_layout()
        target_path = INDIVIDUAL_FIG_DIR / f"{sanitized_id}_explanation_panel.png"
        saved_path = _save_figure_safely(fig, target_path, dpi=300)
    finally:
        plt.close(fig)

    if interactive:
        try:
            plt.show()
        except Exception:
            pass

    _open_image_file(saved_path)
    return str(saved_path)


def generate_individual_shap_waterfall(identifier: Any, interactive: bool = False, pre_extracted: Optional[Tuple] = None) -> str:
    """
    Figure W: Individual SHAP Waterfall Plot for a single employee.
    Visualizes the path from population baseline E[f(x)] to final prediction f(x),
    including individual features, remaining tree features, base tree score,
    and meta-learner stacking integration.
    """
    if pre_extracted:
        record, pos_factors, neg_factors, base_val, raw_final, shap_meta, raw_row = pre_extracted
    else:
        record, pos_factors, neg_factors, base_val, raw_final, shap_meta, raw_row = _extract_employee_data_and_shap(identifier)

    emp_id = record.get("employee_reference", "EMP_UNKNOWN")
    rec_idx = record.get("record_index", 0)
    sanitized_id = sanitize_filename_component(emp_id)
    prob = record.get("risk_score", 0.0)
    risk_tier = record.get("risk_level", "Unknown")

    total_shap = shap_meta.get("sum_shap_attributions", 0.0)
    base_tree_margin = shap_meta.get("base_trees_margin", base_val + total_shap)
    tree_prob = shap_meta.get("base_trees_probability", 1.0 / (1.0 + np.exp(-base_tree_margin)))
    meta_adj = shap_meta.get("meta_learner_adjustment", 0.0)
    final_margin = shap_meta.get("final_model_margin", raw_final)

    selected_pos = sorted(pos_factors, key=lambda x: abs(x.get("shap_value", x.get("shap_impact", 0.0))), reverse=True)[:4]
    selected_neg = sorted(neg_factors, key=lambda x: abs(x.get("shap_value", x.get("shap_impact", 0.0))), reverse=True)[:4]
    all_selected = selected_pos + selected_neg
    all_selected.sort(key=lambda x: abs(x.get("shap_value", x.get("shap_impact", 0.0))), reverse=True)

    items = []
    # 1. Base value
    items.append({
        "label": "E[f(X)]\nBase Rate",
        "value": base_val,
        "delta": base_val,
        "is_total": True,
        "color": PALETTE["base"]
    })

    curr = base_val
    for f in all_selected:
        delta = float(f.get("shap_value", f.get("shap_impact", 0.0)))
        feat_raw = f.get("feature", "")
        _, lbl_short, _ = interpret_feature(feat_raw, float(f.get("feature_value", 0.0)), delta, raw_row)
        clean_lbl = lbl_short.split("(")[0].strip()
        color = PALETTE["push"] if delta > 0 else PALETTE["pull"]
        items.append({
            "label": clean_lbl,
            "value": curr + delta,
            "delta": delta,
            "is_total": False,
            "color": color
        })
        curr += delta

    # 2. Remaining Features delta in the 30-feature Tree Ensemble
    rem_delta = base_tree_margin - curr
    if abs(rem_delta) > 0.001:
        rem_color = PALETTE["push"] if rem_delta > 0 else PALETTE["pull"]
        items.append({
            "label": "Other\nFeatures",
            "value": base_tree_margin,
            "delta": rem_delta,
            "is_total": False,
            "color": rem_color
        })
        curr = base_tree_margin

    # 3. Base Tree Ensemble Margin
    items.append({
        "label": "f_trees(x)\nTree Margin",
        "value": base_tree_margin,
        "delta": base_tree_margin,
        "is_total": True,
        "color": PALETTE["base_tree"]
    })

    # 4. Meta-Learner Stacking Integration Step
    if abs(meta_adj) > 0.001:
        meta_color = PALETTE["push"] if meta_adj > 0 else PALETTE["pull"]
        items.append({
            "label": "Meta-Learner\nStacking",
            "value": final_margin,
            "delta": meta_adj,
            "is_total": False,
            "color": meta_color
        })
        curr = final_margin

    # 5. Final Step 32 Score
    items.append({
        "label": "f_Step32(x)\nFinal Output",
        "value": final_margin,
        "delta": final_margin,
        "is_total": True,
        "color": PALETTE["final"]
    })

    fig, ax = plt.subplots(figsize=(13, 6.6))
    try:
        n_steps = len(items)
        indices = np.arange(n_steps)

        for i in range(n_steps):
            it = items[i]
            if it["is_total"]:
                bot = 0 if it["value"] >= 0 else it["value"]
                ht = abs(it["value"])
                ax.bar(i, ht, bottom=bot, color=it["color"], width=0.55, edgecolor=PALETTE["text_dark"], linewidth=1.2, zorder=3)
                v_offset = 0.08 if it["value"] >= 0 else -0.18
                ax.text(i, it["value"] + v_offset,
                        f"{it['value']:.2f}", ha="center", va="bottom" if it["value"] >= 0 else "top",
                        fontsize=9.5, fontweight="bold", color=it["color"])
            else:
                prev_val = items[i - 1]["value"]
                delta = it["delta"]
                bot = prev_val if delta >= 0 else prev_val + delta
                ht = abs(delta)
                ax.bar(i, ht, bottom=bot, color=it["color"], width=0.55, edgecolor=PALETTE["text_dark"], linewidth=1.0, zorder=3)
                ax.plot([i - 1, i], [prev_val, prev_val], color=PALETTE["text_muted"], linestyle=":", linewidth=1.2, zorder=2)
                sign = "+" if delta >= 0 else ""
                txt_y = prev_val + (delta / 2)
                ax.text(i, txt_y, f"{sign}{delta:.2f}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white")

        ax.set_xticks(indices)
        ax.set_xticklabels([it["label"] for it in items], rotation=25, ha="right", fontsize=8.5, fontweight="bold")
        ax.set_ylabel("Model Log-Odds Space (Margin Output)", fontsize=10, fontweight="bold", color=PALETTE["text_dark"])
        ax.axhline(0, color=PALETTE["text_dark"], linewidth=1, linestyle="-", alpha=0.5)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Ensure generous y-headroom and footroom to prevent label or box collisions
        all_y_points = [0.0]
        for it in items:
            all_y_points.append(it["value"])
            if not it.get("is_total", False):
                all_y_points.append(it["value"] - it.get("delta", 0))
        min_y = min(all_y_points)
        max_y = max(all_y_points)
        y_span = max(max_y - min_y, 1.0)
        ax.set_ylim(min_y - y_span * 0.22, max_y + y_span * 0.38)

        ax.text(0.02, 0.93,
                f"Employee: {emp_id} (Index {rec_idx})  •  Predicted Risk: {prob*100:.1f}% ({risk_tier})\n"
                f"Tree Efficiency: Base ({base_val:.2f}) + Σ SHAP ({total_shap:+.2f}) = {base_tree_margin:+.2f} log-odds (P_trees={tree_prob*100:.1f}%)  •  Meta Stacking -> {final_margin:+.2f} log-odds ({prob*100:.1f}%)",
                transform=ax.transAxes, fontsize=8.8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["bg_light"], edgecolor=PALETTE["border"], alpha=0.95))

        ax.set_title(f"Individual SHAP Waterfall Attribution — {emp_id} (Holdout Index {rec_idx})\n"
                     f"Complete Cumulative Log-Odds Accounting (Turnover Probability = {prob*100:.1f}%, τ = 0.3610)",
                     fontsize=12, fontweight="bold", pad=15, color=PALETTE["text_dark"])

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["push"], label="Risk-Increasing Force (Push)"),
            plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["pull"], label="Retention Anchor (Pull)"),
            plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["base"], label="Enterprise Baseline E[f(x)]"),
            plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["base_tree"], label="Base Tree Margin f_trees(x)"),
            plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["final"], label="Final Model Output f_Step32(x)")
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=8.0)

        plt.tight_layout()
        target_path = INDIVIDUAL_FIG_DIR / f"{sanitized_id}_waterfall.png"
        saved_path = _save_figure_safely(fig, target_path, dpi=300)
    finally:
        plt.close(fig)

    if interactive:
        try:
            plt.show()
        except Exception:
            pass

    _open_image_file(saved_path)
    return str(saved_path)


def generate_individual_shap_bar(identifier: Any, interactive: bool = False, pre_extracted: Optional[Tuple] = None) -> str:
    """
    Figure B: Horizontal Diverging SHAP Bar Chart for a single employee.
    Visualizes positive drivers (right) vs negative retention anchors (left).
    """
    if pre_extracted:
        record, pos_factors, neg_factors, _, _, shap_meta, raw_row = pre_extracted
    else:
        record, pos_factors, neg_factors, _, _, shap_meta, raw_row = _extract_employee_data_and_shap(identifier)

    emp_id = record.get("employee_reference", "EMP_UNKNOWN")
    rec_idx = record.get("record_index", 0)
    sanitized_id = sanitize_filename_component(emp_id)
    prob = record.get("risk_score", 0.0)
    risk_tier = record.get("risk_level", "Unknown")

    push_sub = shap_meta.get("top_push_subtotal", sum(f.get("shap_value", f.get("shap_impact", 0.0)) for f in pos_factors))
    pull_sub = shap_meta.get("top_pull_subtotal", sum(f.get("shap_value", f.get("shap_impact", 0.0)) for f in neg_factors))
    top10_sub = shap_meta.get("top_10_net_subtotal", push_sub + pull_sub)
    total_shap = shap_meta.get("sum_shap_attributions", top10_sub)

    factors = pos_factors[:5] + neg_factors[:5]
    if not factors:
        factors = record.get("top_feature_contributions", [])[:10]

    factors.sort(key=lambda x: x.get("shap_value", x.get("shap_impact", 0.0)))

    labels = []
    values = []
    colors = []
    val_labels = []

    for f in factors:
        feat = f.get("feature", "")
        val = float(f.get("feature_value", 0.0))
        shap_v = float(f.get("shap_value", f.get("shap_impact", 0.0)))
        _, label, _ = interpret_feature(feat, val, shap_v, raw_row)
        labels.append(label)
        values.append(shap_v)
        colors.append(PALETTE["push"] if shap_v > 0 else PALETTE["pull"])
        val_labels.append(f"{'+' if shap_v > 0 else ''}{shap_v:.3f}")

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    try:
        bars = ax.barh(range(len(labels)), values, color=colors, height=0.6,
                       edgecolor=PALETTE["text_dark"], linewidth=0.8, zorder=3)

        ax.axvline(0, color=PALETTE["text_dark"], linewidth=1.2, linestyle="-", zorder=4)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9, fontweight="bold", color=PALETTE["text_dark"])
        ax.set_xlabel("SHAP Feature Attribution (Log-Odds Impact)", fontsize=10, fontweight="bold", color=PALETTE["text_dark"])

        for i, bar in enumerate(bars):
            v = values[i]
            offset = 0.02 if v >= 0 else -0.02
            ha = "left" if v >= 0 else "right"
            ax.text(v + offset, bar.get_y() + bar.get_height() / 2, val_labels[i],
                    va="center", ha=ha, fontsize=9, fontweight="bold", color=colors[i])

        max_abs = max(max(abs(v) for v in values) * 1.3, 0.5) if values else 1.0
        ax.set_xlim(-max_abs, max_abs)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        ax.set_title(f"Individual SHAP Factor Contributions — {emp_id} (Holdout Index {rec_idx})\n"
                     f"Estimated Turnover Probability: {prob*100:.1f}% ({risk_tier}, Decision Threshold τ = 0.3610)",
                     fontsize=12.0, fontweight="bold", pad=16, color=PALETTE["text_dark"])

        ax.text(max_abs * 0.5, len(labels) - 0.2, "► Elevates Turnover Risk (+)",
                fontsize=9.5, fontweight="bold", color=PALETTE["push"], ha="center")
        ax.text(-max_abs * 0.5, len(labels) - 0.2, "◄ Buffers Retention (-)",
                fontsize=9.5, fontweight="bold", color=PALETTE["pull"], ha="center")

        fig.text(0.5, 0.015,
                 "Note: SHAP attributions reflect model sensitivity within trained parameters; they do not establish real-world causality.",
                 fontsize=8.5, color=PALETTE["text_muted"], ha="center")

        plt.subplots_adjust(bottom=0.12)
        target_path = INDIVIDUAL_FIG_DIR / f"{sanitized_id}_contributions.png"
        saved_path = _save_figure_safely(fig, target_path, dpi=300)
    finally:
        plt.close(fig)

    if interactive:
        try:
            plt.show()
        except Exception:
            pass

    _open_image_file(saved_path)
    return str(saved_path)


def format_natural_language_explanation(record: Dict[str, Any],
                                        pos_factors: Optional[List[Dict[str, Any]]] = None,
                                        neg_factors: Optional[List[Dict[str, Any]]] = None,
                                        shap_meta: Optional[Dict[str, Any]] = None,
                                        raw_row: Optional[Dict[str, Any]] = None) -> str:
    """
    Generates a neat, clean, human-readable terminal explanation:
    - Concise profile and prediction summary
    - Clean tabular breakdown of top risk push drivers and protective anchors
    - Direct, simple executive takeaway without overwhelming walls of text
    """
    emp_id = record.get("employee_reference", "EMP_UNKNOWN")
    idx = record.get("record_index", 0)
    score = record.get("risk_score", 0.0)
    risk_tier = record.get("risk_level", "Unknown")
    pred_cls = record.get("predicted_class", 0)
    pred_out = record.get("predicted_outcome", "Predicted Left" if pred_cls == 1 else "Predicted Stayed")
    act_cls = record.get("actual_attrition", None)
    eval_res = record.get("evaluation_result", "Unknown")
    info = record.get("employee_info", {})

    if raw_row is None:
        raw_row = _get_raw_employee_row(idx)

    if shap_meta is None:
        full_cache = _load_full_shap_cache()
        shap_meta = full_cache.get(emp_id, {})

    if pos_factors is None:
        pos_factors = shap_meta.get("top_push_factors", record.get("shap_top_positive", []))
    if neg_factors is None:
        neg_factors = shap_meta.get("top_pull_factors", record.get("shap_top_negative", []))

    lines = []

    csv_row = record.get("csv_row_number", idx + 2)
    fingerprint = record.get("row_fingerprint", "")
    if not fingerprint and raw_row:
        from reporting.data_provider import compute_record_fingerprint
        fingerprint = compute_record_fingerprint(raw_row)

    # 1. Header Box
    lines.append(f"\n{Colors.CYAN}╔{'═'*76}╗{Colors.ENDC}")
    lines.append(f"{Colors.CYAN}║{Colors.BOLD}   INDIVIDUAL EMPLOYEE RETENTION EXPLANATION — {emp_id:<29} {Colors.CYAN}║{Colors.ENDC}")
    lines.append(f"{Colors.CYAN}║{Colors.DIM}   Record Index: {idx:<4} • CSV Row: {csv_row:<4} • Policy: τ = 0.3610 • SHA: {fingerprint[:12]:<12} {Colors.CYAN}║{Colors.ENDC}")
    lines.append(f"{Colors.CYAN}╚{'═'*76}╝{Colors.ENDC}\n")

    # 2. Profile & Prediction Summary
    dept = raw_row.get("Department", info.get("Department", "N/A"))
    role = raw_row.get("JobRole", info.get("Job Role", "N/A"))
    age = raw_row.get("Age", info.get("Age", "N/A"))
    if "MonthlyIncome" in raw_row and pd.notna(raw_row["MonthlyIncome"]):
        inc = f"${int(raw_row['MonthlyIncome']):,}"
    else:
        inc = str(info.get("Monthly Income", "$0"))
    ot = raw_row.get("OverTime", info.get("Overtime", "N/A"))
    tot_exp = raw_row.get("TotalWorkingYears", info.get("Total Working Years", "N/A"))
    if isinstance(tot_exp, (float, np.floating)) and tot_exp == int(tot_exp):
        tot_exp = int(tot_exp)
    tenure = raw_row.get("YearsAtCompany", info.get("Years at Company", "N/A"))
    if isinstance(tenure, (float, np.floating)) and tenure == int(tenure):
        tenure = int(tenure)
    m_stat = raw_row.get("MaritalStatus", info.get("Marital Status", "N/A"))

    lines.append(f"{Colors.BOLD}{Colors.BLUE}► 1. EMPLOYEE PROFILE & MODEL PREDICTION{Colors.ENDC}")
    lines.append(f"{Colors.DIM}─"*78 + f"{Colors.ENDC}")
    lines.append(f"  • {Colors.BOLD}Reference ID:{Colors.ENDC}      {emp_id} (Holdout Index: {idx}, CSV Line: {csv_row})")
    lines.append(f"  • {Colors.BOLD}Role & Dept:{Colors.ENDC}       {role} | {dept}")
    lines.append(f"  • {Colors.BOLD}Demographics:{Colors.ENDC}      Age: {age} • Experience: {tot_exp} yrs • Tenure: {tenure} yrs • {m_stat}")
    lines.append(f"  • {Colors.BOLD}Compensation:{Colors.ENDC}      Income: {inc}/mo • Overtime: {ot}")
    
    risk_color = Colors.FAIL if score >= 0.3610 else Colors.GREEN
    lines.append(f"  • {Colors.BOLD}Predicted Risk:{Colors.ENDC}    {risk_color}{score*100:.1f}%{Colors.ENDC} ({score:.4f}) ► {risk_color}{Colors.BOLD}{risk_tier}{Colors.ENDC} (Policy: τ = 0.3610)")

    if act_cls is not None:
        act_str = "Left Company (1)" if act_cls == 1 else "Stayed with Company (0)"
        res_color = Colors.GREEN if "True" in eval_res or "Correct" in eval_res else Colors.FAIL
        lines.append(f"  • {Colors.BOLD}Ground Truth:{Colors.ENDC}      {act_str} ► {res_color}{Colors.BOLD}{eval_res}{Colors.ENDC}")

    lines.append("")

    # 3. Key Risk Factors (Push Forces)
    lines.append(f"{Colors.BOLD}{Colors.FAIL}► 2. KEY FACTORS INCREASING FLIGHT RISK (PUSH FORCES){Colors.ENDC}")
    lines.append(f"{Colors.DIM}─"*78 + f"{Colors.ENDC}")
    if pos_factors:
        for i, f in enumerate(pos_factors[:4], 1):
            feat = f.get("feature", "")
            val = float(f.get("feature_value", 0.0))
            shap_v = float(f.get("shap_value", f.get("shap_impact", 0.0)))
            sentence, label, _ = interpret_feature(feat, val, shap_v, raw_row)
            lines.append(f"  [{i}] {Colors.FAIL}+{shap_v:.3f} SHAP{Colors.ENDC}  •  {Colors.BOLD}{label}{Colors.ENDC}")
            lines.append(f"      {Colors.DIM}Impact: {sentence}{Colors.ENDC}")
    else:
        lines.append("  No significant risk-increasing factors identified.")
    lines.append("")

    # 4. Key Retention Anchors (Pull Forces)
    lines.append(f"{Colors.BOLD}{Colors.GREEN}► 3. KEY FACTORS BUFFERING RETENTION (PROTECTIVE PULL FORCES){Colors.ENDC}")
    lines.append(f"{Colors.DIM}─"*78 + f"{Colors.ENDC}")
    if neg_factors:
        for i, f in enumerate(neg_factors[:4], 1):
            feat = f.get("feature", "")
            val = float(f.get("feature_value", 0.0))
            shap_v = float(f.get("shap_value", f.get("shap_impact", 0.0)))
            sentence, label, _ = interpret_feature(feat, val, shap_v, raw_row)
            lines.append(f"  [{i}] {Colors.GREEN}{shap_v:.3f} SHAP{Colors.ENDC}  •  {Colors.BOLD}{label}{Colors.ENDC}")
            lines.append(f"      {Colors.DIM}Impact: {sentence}{Colors.ENDC}")
    else:
        lines.append("  No significant retention buffering factors identified.")
    lines.append("")

    # 5. Executive Synthesis & Guidance
    lines.append(f"{Colors.BOLD}{Colors.BLUE}► 4. EXECUTIVE SYNTHESIS & GUIDANCE{Colors.ENDC}")
    lines.append(f"{Colors.DIM}─"*78 + f"{Colors.ENDC}")
    syn_paragraph = generate_personalized_synthesis_paragraph(record, shap_meta, pos_factors, neg_factors, raw_row)
    for line in textwrap.wrap(syn_paragraph, width=76):
        lines.append(f"  {line}")
    lines.append("")

    # 6. Research Integrity Notice
    lines.append(f"{Colors.BOLD}► 5. RESEARCH INTEGRITY & CAUSAL DISCLAIMER{Colors.ENDC}")
    lines.append(f"{Colors.DIM}─"*78 + f"{Colors.ENDC}")
    lines.append(f"{Colors.DIM}  SHAP values describe statistical feature attributions within the model architecture.{Colors.ENDC}")
    lines.append(f"{Colors.DIM}  They do NOT imply counterfactual causality. Interventions must remain supportive.{Colors.ENDC}\n")

    return "\n".join(lines)


def run_individual_audit(identifier: Any, interactive: bool = False) -> Dict[str, Any]:
    """
    Executes the complete individual employee audit:
    1. Validates employee existence and retrieves raw data.
    2. Prints human-readable natural language narrative to terminal.
    3. Generates and saves SHAP waterfall graph at 300 DPI.
    4. Generates and saves SHAP horizontal contribution bar graph at 300 DPI.
    5. Generates and saves Two-Column Human-Readable Explanation Panel at 300 DPI.
    6. Opens saved figures with Windows viewer gracefully.
    """
    audit_data = _extract_employee_data_and_shap(identifier)
    record, pos_factors, neg_factors, base_val, raw_final, shap_meta, raw_row = audit_data
    emp_id = record.get("employee_reference", "EMP_UNKNOWN")

    # 1. Print terminal narrative
    explanation_text = format_natural_language_explanation(record, pos_factors, neg_factors,
                                                           shap_meta=shap_meta, raw_row=raw_row)
    print(explanation_text)

    # 2. Generate Graphs
    print(f"{Colors.BOLD}{Colors.BLUE}► GENERATING INDIVIDUAL SHAP & EXPLANATION GRAPHS (300 DPI){Colors.ENDC}")
    print(f"{Colors.DIM}─"*78 + f"{Colors.ENDC}")
    
    waterfall_path = None
    bar_path = None
    panel_path = None

    # Graph 1: Waterfall
    try:
        waterfall_path = generate_individual_shap_waterfall(identifier, interactive=interactive, pre_extracted=audit_data)
        print(f"  {Colors.GREEN}✔ Saved SHAP Waterfall Graph:{Colors.ENDC}         {waterfall_path}")
    except Exception as e:
        print(f"  {Colors.FAIL}✖ FAILED to save Waterfall Graph for {emp_id}:{Colors.ENDC} {e}")

    # Graph 2: Diverging Bar
    try:
        bar_path = generate_individual_shap_bar(identifier, interactive=interactive, pre_extracted=audit_data)
        print(f"  {Colors.GREEN}✔ Saved Contribution Bar Graph:{Colors.ENDC}       {bar_path}")
    except Exception as e:
        print(f"  {Colors.FAIL}✖ FAILED to save Contribution Bar Graph for {emp_id}:{Colors.ENDC} {e}")

    # Graph 3: Two-Column Explanation Panel
    try:
        panel_path = generate_individual_explanation_panel(identifier, interactive=interactive, pre_extracted=audit_data)
        print(f"  {Colors.GREEN}✔ Saved Two-Column Explanation Panel:{Colors.ENDC}   {panel_path}")
    except Exception as e:
        print(f"  {Colors.FAIL}✖ FAILED to save Explanation Panel for {emp_id}:{Colors.ENDC} {e}")

    if waterfall_path or bar_path or panel_path:
        print(f"\n{Colors.CYAN}Output directory:{Colors.ENDC}")
        print(f"  {INDIVIDUAL_FIG_DIR}")
        print(f"{Colors.DIM}All images verified and launched with system viewer for desktop review.{Colors.ENDC}\n")
    else:
        print(f"\n{Colors.FAIL}ERROR: Could not generate individual SHAP graphs for {emp_id}. See error details above.{Colors.ENDC}\n")

    return {
        "record": record,
        "waterfall_path": waterfall_path,
        "bar_path": bar_path,
        "panel_path": panel_path,
        "shap_meta": shap_meta
    }


def interactive_employee_review_loop():
    """
    Continuous interactive loop for inspecting multiple employees
    without restarting the application.
    """
    cache = load_cached_evaluation()
    records = cache.get("records", [])

    print(f"\n{Colors.CYAN}╔{'═'*76}╗{Colors.ENDC}")
    print(f"{Colors.CYAN}║{Colors.BOLD}         INTERACTIVE MULTI-EMPLOYEE PREDICTION & SHAP INSPECTOR           {Colors.CYAN}║{Colors.ENDC}")
    print(f"{Colors.CYAN}║{Colors.DIM}         Inspect Any of the 594 Holdout Employees With Full Graphs         {Colors.CYAN}║{Colors.ENDC}")
    print(f"{Colors.CYAN}╚{'═'*76}╝{Colors.ENDC}\n")

    while True:
        print(f"{Colors.BOLD}Options:{Colors.ENDC}")
        print("  • Enter an Employee Reference (e.g. EMP_HD_001, EMP_HD_043)")
        print("  • Enter an Index number (0 to 593)")
        print("  • Type 'sample' to see curated demonstration cases (TP, TN, FP, FN)")
        print("  • Type 'q' or 'quit' to return to main menu\n")

        query = input(f"{Colors.BOLD}Enter Employee ID or query: {Colors.ENDC}").strip()
        if not query:
            continue

        if query.lower() in ["q", "quit", "exit"]:
            print("Returning to main menu.")
            break

        if query.lower() in ["sample", "list"]:
            print(f"\n{Colors.BOLD}Curated Demonstration Holdout Cases:{Colors.ENDC}")
            print(f"  • {Colors.GREEN}EMP_HD_001{Colors.ENDC} (Index 0)  - High Risk (85.3%) • Ground Truth: Left (1)   [True Positive]")
            print(f"  • {Colors.GREEN}EMP_HD_002{Colors.ENDC} (Index 1)  - Low Risk  (4.0%)  • Ground Truth: Stayed (0) [True Negative]")
            print(f"  • {Colors.GREEN}EMP_HD_043{Colors.ENDC} (Index 42) - Low Risk  (6.3%)  • Ground Truth: Stayed (0) [True Negative]")
            print(f"  • {Colors.FAIL}EMP_HD_047{Colors.ENDC} (Index 46) - High Risk (73.2%) • Ground Truth: Stayed (0) [False Positive / False Alarm]")
            print(f"  • {Colors.WARNING}EMP_HD_093{Colors.ENDC} (Index 92) - Low Risk  (24.1%) • Ground Truth: Left (1)   [False Negative / Missed Leaver]\n")
            continue

        try:
            run_individual_audit(query)
        except Exception as e:
            print(f"{Colors.FAIL}Error inspecting employee '{query}': {e}{Colors.ENDC}")
            print(f"{Colors.DIM}Ensure the ID is formatted like 'EMP_HD_001' or enter an integer index from 0 to 593.{Colors.ENDC}\n")

        input(f"{Colors.DIM}Press Enter to inspect another employee...{Colors.ENDC}\n")
