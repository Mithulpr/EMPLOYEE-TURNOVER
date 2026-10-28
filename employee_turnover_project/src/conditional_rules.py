"""
Stage 5 & Stage 6: Conditional Error Correction and Domain Refinement Module
=============================================================================
Implements:
- Stage 5: False-Negative (FN) Recovery Rule
- Stage 5: False-Positive (FP) Suppression Veto
- Stage 6: Domain Boundary Specialist Refinement Rule
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def build_specialized_fn_features(
    p_meta: np.ndarray,
    p_cb: np.ndarray,
    p_lgb: np.ndarray,
    p_s27: np.ndarray,
    ref_cb: np.ndarray,
    ref_lgb: np.ndarray,
    ref_s27: np.ndarray,
    ref_pmeta: np.ndarray
) -> np.ndarray:
    """
    Constructs the 24-dimensional consensus feature representation for Stage 5 recovery.
    """
    stack3 = np.column_stack([p_cb, p_lgb, p_s27])
    f_mean = np.mean(stack3, axis=1)
    f_median = np.median(stack3, axis=1)
    f_max = np.max(stack3, axis=1)
    f_min = np.min(stack3, axis=1)
    f_std = np.std(stack3, axis=1)

    f_d_cb_lgb = p_cb - p_lgb
    f_d_cb_s27 = p_cb - p_s27
    f_d_lgb_s27 = p_lgb - p_s27

    f_cb_p28 = p_cb - p_meta
    f_lgb_p28 = p_lgb - p_meta
    f_s27_p28 = p_s27 - p_meta
    f_max_p28 = f_max - p_meta
    f_mean_p28 = f_mean - p_meta

    f_boost_supp = (p_cb + p_lgb) / 2.0 - p_s27
    f_dual_boost = np.minimum(p_cb, p_lgb)
    f_dual_boost_gap = f_dual_boost - p_meta

    rank_cb = np.array([np.searchsorted(np.sort(ref_cb), x) / len(ref_cb) for x in p_cb])
    rank_lgb = np.array([np.searchsorted(np.sort(ref_lgb), x) / len(ref_lgb) for x in p_lgb])
    rank_s27 = np.array([np.searchsorted(np.sort(ref_s27), x) / len(ref_s27) for x in p_s27])
    rank_p28 = np.array([np.searchsorted(np.sort(ref_pmeta), x) / len(ref_pmeta) for x in p_meta])

    X = np.column_stack([
        p_cb, p_lgb, p_s27, p_meta,
        f_mean, f_median, f_max, f_min, f_std,
        f_d_cb_lgb, f_d_cb_s27, f_d_lgb_s27,
        f_cb_p28, f_lgb_p28, f_s27_p28, f_max_p28, f_mean_p28,
        f_boost_supp, f_dual_boost, f_dual_boost_gap,
        rank_cb, rank_lgb, rank_s27, rank_p28
    ])
    return X


class ConditionalErrorCorrector:
    """
    Applies Stage 5 and Stage 6 conditional rules without replacing the meta-learner.
    Rules trigger strictly when validated contextual criteria are satisfied.
    """
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = MODELS_DIR
        self.models_dir = models_dir

        fn_path = os.path.join(self.models_dir, "false_negative_recovery_catboost.joblib")
        fp_path = os.path.join(self.models_dir, "false_positive_recovery_catboost.joblib")
        ref_path = os.path.join(self.models_dir, "reference_pool.npz")
        spec_path = os.path.join(self.models_dir, "specialist_catboost.joblib")

        self.clf_fn = joblib.load(fn_path)
        self.clf_fp = joblib.load(fp_path)
        self.ref_pool = np.load(ref_path)
        self.clf_spec = joblib.load(spec_path)

        self.spec_cols = [
            'SevereDissatisfactionFlag', 'YearsWithCurrManager', 'JobHoppingRate',
            'TotalExpToAgeRatio', 'PromotionPaceStrain', 'TotalWorkingYears',
            'P_cb', 'P_lgb', 'P_28'
        ]

    def apply_corrections(
        self,
        pred_base: np.ndarray,
        p_meta: np.ndarray,
        p_cb: np.ndarray,
        p_lgb: np.ndarray,
        p_anchor: np.ndarray,
        df_full: pd.DataFrame,
        features_30: list,
        tau: float = 0.361
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes Stage 5 FN/FP corrections and Stage 6 Boundary Specialist refinement.
        Returns:
            pred_final: corrected binary predictions
            audit_log: detailed dictionary of which rules activated for which samples
        """
        n_samples = len(pred_base)

        # Build Stage 5 feature matrix
        X_fn = build_specialized_fn_features(
            p_meta, p_cb, p_lgb, p_anchor,
            self.ref_pool['p_cb'], self.ref_pool['p_lgb'], self.ref_pool['p_s27'], self.ref_pool['p_meta']
        )
        p_fn = self.clf_fn.predict_proba(X_fn)[:, 1]
        p_fp = self.clf_fp.predict_proba(X_fn)[:, 1]

        # Stage 5 Rules
        cond_fn = (pred_base == 0) & (p_meta < tau) & (p_fn >= 0.35) & (np.minimum(p_cb, p_lgb) >= 0.10)
        cond_fp = (pred_base == 1) & (p_fp >= 0.65) & (p_cb <= 0.25)

        pred_s30 = pred_base.copy()
        pred_s30[cond_fn] = 1
        pred_s30[cond_fp] = 0

        # Stage 6 Domain Boundary Specialist Refinement
        df_sel = df_full[features_30].copy()
        df_sel['P_cb'] = p_cb
        df_sel['P_lgb'] = p_lgb
        df_sel['P_28'] = p_meta
        X_sel = df_sel[self.spec_cols].values
        p_cb_spec = self.clf_spec.predict_proba(X_sel)[:, 1]

        mask_boundary_zone = (pred_s30 == 0) & (p_meta >= 0.15) & (p_meta < tau)
        cond_spec = (
            mask_boundary_zone &
            (df_sel['SevereDissatisfactionFlag'] > 0.5) &
            (df_sel['YearsWithCurrManager'] > -0.5) &
            (df_sel['JobHoppingRate'] <= 1.0) &
            (df_sel['TotalExpToAgeRatio'] > -0.20) &
            (p_cb_spec >= 0.40)
        )

        pred_final = pred_s30.copy()
        pred_final[cond_spec] = 1

        rule_descriptions = []
        for i in range(n_samples):
            c_spec = cond_spec.iloc[i] if hasattr(cond_spec, 'iloc') else cond_spec[i]
            c_fn = cond_fn.iloc[i] if hasattr(cond_fn, 'iloc') else cond_fn[i]
            c_fp = cond_fp.iloc[i] if hasattr(cond_fp, 'iloc') else cond_fp[i]

            if c_spec:
                rule_descriptions.append("Tier 2 Override: Boundary Domain Specialist Refinement")
            elif c_fn:
                rule_descriptions.append("Tier 2 Override: Severe FN Recovery Rule")
            elif c_fp:
                rule_descriptions.append("Tier 2 Override: FP Suppression Veto")
            else:
                rule_descriptions.append("Standard Tier 1 Meta-Learner Decision")

        audit_log = {
            "p_fn": p_fn,
            "p_fp": p_fp,
            "p_specialist": p_cb_spec,
            "cond_fn_count": int(np.sum(cond_fn)),
            "cond_fp_count": int(np.sum(cond_fp)),
            "cond_spec_count": int(np.sum(cond_spec)),
            "rule_descriptions": rule_descriptions
        }
        return pred_final, audit_log
