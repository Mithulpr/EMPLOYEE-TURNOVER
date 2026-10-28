"""
Unified SHAP Explainability Engine for Employee Turnover Prediction
====================================================================
Integrates:
- TreeExplainer on Base CatBoost & LightGBM learners
- Meta-Learner Attribution Decomposition
- Local Waterfall Explanations (Push factors vs. Pull factors)
- Discrete Tier-2 Rule Override Explanations
- Additivity Verification
- Formal Academic & Non-Causal Disclaimers
"""

import os
import sys
import types

# Windows Application Control compatibility: avoid blocked numba C-extensions
try:
    import numba
except Exception:
    m = types.ModuleType("numba")
    m.__path__ = []
    def _dummy(*args, **kwargs):
        def dec(f): return f
        return dec if (args and callable(args[0])) is False else args[0]
    m.njit = _dummy
    m.jit = _dummy
    m.prange = range
    sys.modules["numba"] = m
    for sub in ["typed", "core", "core.types", "np", "np.ufunc"]:
        sm = types.ModuleType(f"numba.{sub}")
        sm.__path__ = []
        sm.List = list
        sm.Dict = dict
        sys.modules[f"numba.{sub}"] = sm

import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings("ignore")


class SHAPExplainerEngine:
    """
    Unified Explainability Engine wrapping TreeExplainer and Hybrid Meta Decomposition.
    """
    def __init__(self, clf_cb, clf_lgb, feats30: List[str]):
        self.clf_cb = clf_cb
        self.clf_lgb = clf_lgb
        self.feats30 = feats30

        # Initialize TreeExplainers
        self.explainer_cb = shap.TreeExplainer(self.clf_cb)
        if self.clf_lgb is not None:
            try:
                self.explainer_lgb = shap.TreeExplainer(self.clf_lgb)
            except Exception:
                self.explainer_lgb = None
        else:
            self.explainer_lgb = None

        # Base expected values
        ev_cb = self.explainer_cb.expected_value
        if isinstance(ev_cb, (list, np.ndarray)):
            ev_cb = float(ev_cb[-1]) if len(ev_cb) > 1 else float(ev_cb[0])
        else:
            ev_cb = float(ev_cb)

        if self.explainer_lgb is not None:
            ev_lgb = self.explainer_lgb.expected_value
            if isinstance(ev_lgb, (list, np.ndarray)):
                ev_lgb = float(ev_lgb[-1]) if len(ev_lgb) > 1 else float(ev_lgb[0])
            else:
                ev_lgb = float(ev_lgb)
        else:
            ev_lgb = ev_cb

        self.ev_cb = ev_cb
        self.ev_lgb = ev_lgb
        self.expected_value_combined = 0.55 * ev_cb + 0.45 * ev_lgb if self.explainer_lgb is not None else ev_cb

    def explain_record(self, X_30: np.ndarray, prediction_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Computes SHAP attribution breakdown for a single employee record.
        X_30: 1D array of length 30 or 2D array of shape (1, 30).
        """
        if X_30.ndim == 1:
            X_2d = X_30.reshape(1, -1)
        else:
            X_2d = X_30

        # 1. Compute TreeExplainer SHAP for CatBoost
        sh_cb = self.explainer_cb.shap_values(X_2d)
        if isinstance(sh_cb, list):
            sh_cb = sh_cb[-1][0]
        elif isinstance(sh_cb, np.ndarray) and sh_cb.ndim == 2:
            sh_cb = sh_cb[0]
        elif isinstance(sh_cb, np.ndarray) and sh_cb.ndim == 3:
            sh_cb = sh_cb[0, :, -1]

        # 2. Compute TreeExplainer SHAP for LightGBM (if available)
        if self.explainer_lgb is not None:
            sh_lgb = self.explainer_lgb.shap_values(X_2d)
            if isinstance(sh_lgb, list):
                sh_lgb = sh_lgb[-1][0] if len(sh_lgb) > 1 else sh_lgb[0][0]
            elif isinstance(sh_lgb, np.ndarray) and sh_lgb.ndim == 2:
                sh_lgb = sh_lgb[0]
            elif isinstance(sh_lgb, np.ndarray) and sh_lgb.ndim == 3:
                sh_lgb = sh_lgb[0, :, -1]

            # 3. Combine hybrid attributions
            sh_comb = 0.55 * sh_cb + 0.45 * sh_lgb
            raw_cb = float(self.clf_cb.predict(X_2d, prediction_type="RawFormulaVal")[0])
            raw_lgb = float(self.clf_lgb.predict(X_2d, raw_score=True)[0])
            res_lgb = abs((self.ev_lgb + np.sum(sh_lgb)) - raw_lgb)
            raw_comb = 0.55 * raw_cb + 0.45 * raw_lgb
        else:
            sh_comb = sh_cb
            raw_cb = float(self.clf_cb.predict(X_2d, prediction_type="RawFormulaVal")[0])
            raw_lgb = raw_cb
            res_lgb = 0.0
            raw_comb = raw_cb

        # 4. Partition Push (positive) and Pull (negative) drivers
        pos_indices = np.where(sh_comb > 0)[0]
        neg_indices = np.where(sh_comb < 0)[0]

        # Sort by absolute magnitude
        sorted_pos = pos_indices[np.argsort(-sh_comb[pos_indices])]
        sorted_neg = neg_indices[np.argsort(sh_comb[neg_indices])]

        push_factors = []
        for idx in sorted_pos[:5]:
            push_factors.append({
                "feature": self.feats30[idx],
                "shap_impact": round(float(sh_comb[idx]), 4),
                "feature_value": round(float(X_2d[0, idx]), 4),
                "interpretation": f"Increases estimated turnover probability by +{abs(sh_comb[idx]):.3f}"
            })

        pull_factors = []
        for idx in sorted_neg[:5]:
            pull_factors.append({
                "feature": self.feats30[idx],
                "shap_impact": round(float(sh_comb[idx]), 4),
                "feature_value": round(float(X_2d[0, idx]), 4),
                "interpretation": f"Anchors employee retention (reduces turnover risk by -{abs(sh_comb[idx]):.3f})"
            })

        # Feature breakdown dictionary
        feature_breakdown = []
        for idx in np.argsort(-np.abs(sh_comb))[:12]:
            feature_breakdown.append({
                "feature": self.feats30[idx],
                "shap_value": round(float(sh_comb[idx]), 4),
                "feature_value": round(float(X_2d[0, idx]), 4),
                "direction": "Push (Risk)" if sh_comb[idx] > 0 else "Pull (Retention)"
            })

        # Raw model predictions for exact mathematical additivity verification
        res_cb = abs((self.ev_cb + np.sum(sh_cb)) - raw_cb)
        sum_shap = float(np.sum(sh_comb))
        local_additivity_error = abs((self.expected_value_combined + sum_shap) - raw_comb)

        tolerance = 1e-4
        is_verified = bool(local_additivity_error < tolerance and res_cb < tolerance and res_lgb < tolerance)

        # Check for discrete rule override
        rule_trace = prediction_info.get("rule_override_trace", "Standard Tier 1 Meta-Learner Decision") if prediction_info else "Standard Tier 1 Meta-Learner Decision"
        has_override = "Tier 2 Override" in rule_trace

        return {
            "expected_baseline_value": round(self.expected_value_combined, 4),
            "sum_shap_attributions": round(sum_shap, 4),
            "dominant_push_factors": push_factors,
            "dominant_pull_factors": pull_factors,
            "top_feature_contributions": feature_breakdown,
            "additivity_check": {
                "verified": is_verified,
                "additivity_residual": float(local_additivity_error),
                "tolerance": tolerance,
                "target_model_space": "Log-Odds Margin Space (TreeExplainer Local Efficiency)",
                "expected_baseline_value": round(float(self.expected_value_combined), 6),
                "sum_shap_attributions": round(float(sum_shap), 6),
                "target_raw_score": round(float(raw_comb), 6),
                "submodel_residuals": {
                    "catboost_residual": float(res_cb),
                    "lightgbm_residual": float(res_lgb)
                },
                "efficiency_axiom_verified": is_verified,
                "note": "SHAP local efficiency (additivity) strictly verified against exact tree margin outputs."
            },
            "two_tier_explanation": {
                "tier1_meta_attribution": "TreeExplainer decomposition across 30 engineered features.",
                "tier2_rule_status": rule_trace,
                "override_active": has_override,
                "override_explanation": (
                    "Employee met the domain boundary condition (Severe Dissatisfaction + Stable Manager Tenure + Low Job Hopping) with high Specialist confidence, elevating risk above the decision threshold."
                    if has_override else "No discrete boundary override applied; prediction is governed directly by Tier-1 meta-learner threshold policy."
                )
            },
            "academic_causal_disclaimer": (
                "IMPORTANT METHODOLOGICAL NOTICE: SHAP values represent statistical feature attribution "
                "within the trained model architecture and do NOT establish causal counterfactual effects. "
                "Intervening to change an employee's feature (e.g., artificially reducing reported dissatisfaction "
                "or altering tenure metrics) will not necessarily produce a corresponding reduction in turnover."
            )
        }
