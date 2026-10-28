"""
Unified Drift Monitoring Module
================================
Implements:
- Population Stability Index (PSI) using reference decile quantiles
- Two-Sample Kolmogorov-Smirnov (KS) test against reference training distributions
- Benjamini-Hochberg False Discovery Rate (FDR) p-value adjustment
- Sample Size Safeguards (N < 30 alert)
- Evaluates drift across the 30 model features
- Clear distinction between feature drift and performance degradation
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional


class UnifiedDriftMonitor:
    """
    Offline Batch Drift Monitoring Engine comparing incoming batches
    against reference distributions from the frozen Training Pool (N=1,900).
    """
    def __init__(self, reference_path: Optional[str] = None):
        if reference_path is None:
            p1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "reference_pool_summary.json")
            p2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reference_pool_summary.json")
            reference_path = p1 if os.path.exists(p1) else p2
        self.reference_path = reference_path

        with open(self.reference_path, "r") as f:
            self.ref_data = json.load(f)

        self.ref_features = self.ref_data.get("features", {})
        self.feature_names = self.ref_data.get("feature_names", list(self.ref_features.keys()))
        self.n_reference = self.ref_data.get("total_reference_samples", 1900)

    def calculate_psi(self, current_series: pd.Series, feature_name: str) -> Dict[str, Any]:
        """
        Calculates Population Stability Index (PSI) using reference decile cutoffs.
        """
        ref_feat = self.ref_features.get(feature_name)
        if not ref_feat or "decile_quantiles" not in ref_feat:
            return {"psi": 0.0, "status": "UNKNOWN", "label": "No reference data"}

        clean_curr = current_series.dropna().values
        if len(clean_curr) == 0:
            return {"psi": 0.0, "status": "INSUFFICIENT_DATA", "label": "No data available"}

        cutoffs = sorted(list(set(ref_feat["decile_quantiles"])))
        if len(cutoffs) < 2:
            return {"psi": 0.0, "status": "STABLE", "label": "Low Variance"}

        bins = [-np.inf] + cutoffs[1:-1] + [np.inf]
        ref_expected_pct = 1.0 / (len(bins) - 1)

        counts, _ = np.histogram(clean_curr, bins=bins)
        curr_pct = counts / len(clean_curr)

        eps = 1e-4
        psi_val = 0.0
        for actual, expected in zip(curr_pct, [ref_expected_pct] * len(curr_pct)):
            a = max(actual, eps)
            e = max(expected, eps)
            psi_val += (a - e) * np.log(a / e)

        psi_val = float(np.round(max(0.0, psi_val), 4))
        if psi_val < 0.10:
            status = "STABLE"
            label = "Negligible Drift"
        elif psi_val <= 0.25:
            status = "MODERATE"
            label = "Moderate Shift — Review Recommended"
        else:
            status = "SIGNIFICANT"
            label = "Significant Drift Alert — Retraining / Audit Required"

        return {
            "psi": psi_val,
            "status": status,
            "label": label,
            "current_mean": round(float(np.mean(clean_curr)), 3),
            "reference_mean": round(float(ref_feat.get("mean", 0.0)), 3),
            "current_std": round(float(np.std(clean_curr)), 3),
            "reference_std": round(float(ref_feat.get("std", 0.0)), 3)
        }

    def calculate_batch_drift(self, df_30: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive PSI and KS-test drift diagnostics across 30 features.
        df_30 can be either a pre-transformed 30-feature DataFrame or a DataFrame
        containing matching column names.
        """
        n_current = len(df_30)
        has_sample_size_warning = n_current < 30

        feature_reports = []
        raw_p_values = []

        # Analyze features present in both df_30 and reference
        for feat in self.feature_names:
            if feat in df_30.columns:
                curr_s = df_30[feat].dropna()
                if len(curr_s) >= 5:
                    psi_res = self.calculate_psi(curr_s, feat)
                    ref_feat = self.ref_features[feat]

                    # KS test against reference raw sample if available, else synthetic normal
                    ref_sample = ref_feat.get("raw_values_sample")
                    if ref_sample and len(ref_sample) > 10:
                        ref_arr = np.array(ref_sample)
                    else:
                        ref_mean = ref_feat.get("mean", 0.0)
                        ref_std = max(ref_feat.get("std", 1.0), 1e-4)
                        np.random.seed(42)
                        ref_arr = np.random.normal(ref_mean, ref_std, size=min(self.n_reference, 1000))

                    ks_stat, ks_pval = stats.ks_2samp(curr_s.values, ref_arr)
                    raw_p_values.append(ks_pval)

                    feature_reports.append({
                        "feature": feat,
                        "psi": psi_res["psi"],
                        "psi_status": psi_res["status"],
                        "psi_label": psi_res["label"],
                        "ks_statistic": round(float(ks_stat), 4),
                        "ks_raw_pvalue": float(ks_pval),
                        "current_mean": psi_res.get("current_mean"),
                        "reference_mean": psi_res.get("reference_mean")
                    })

        # Benjamini-Hochberg FDR correction on KS p-values
        m = len(raw_p_values)
        if m > 0:
            sorted_indices = np.argsort(raw_p_values)
            sorted_pvals = np.array(raw_p_values)[sorted_indices]
            fdr_adjusted = np.zeros(m)
            for rank, (orig_idx, p) in enumerate(zip(sorted_indices, sorted_pvals), start=1):
                adj_p = min(1.0, p * m / rank)
                fdr_adjusted[orig_idx] = adj_p

            for i, rep in enumerate(feature_reports):
                rep["ks_fdr_adjusted_pvalue"] = round(float(fdr_adjusted[i]), 4)
                rep["ks_drift_detected"] = bool(fdr_adjusted[i] < 0.05)
        else:
            for rep in feature_reports:
                rep["ks_fdr_adjusted_pvalue"] = 1.0
                rep["ks_drift_detected"] = False

        # Summary Metrics
        sig_psi_count = sum(1 for r in feature_reports if r["psi_status"] == "SIGNIFICANT")
        mod_psi_count = sum(1 for r in feature_reports if r["psi_status"] == "MODERATE")
        ks_drift_count = sum(1 for r in feature_reports if r.get("ks_drift_detected", False))

        overall_status = "STABLE"
        if sig_psi_count >= 3 or ks_drift_count >= 5:
            overall_status = "SIGNIFICANT_DRIFT_ALERT"
        elif mod_psi_count >= 3 or ks_drift_count >= 2:
            overall_status = "MODERATE_DRIFT_DETECTED"

        return {
            "overall_status": overall_status,
            "sample_size": n_current,
            "sample_size_safeguard": {
                "active": has_sample_size_warning,
                "message": (
                    f"Sample size warning: N={n_current} is under the N=30 threshold for reliable statistical drift estimation."
                    if has_sample_size_warning else f"Sample size N={n_current} meets statistical minimum threshold (N >= 30)."
                )
            },
            "reference_population_size": self.n_reference,
            "significant_psi_features_count": sig_psi_count,
            "moderate_psi_features_count": mod_psi_count,
            "ks_drift_features_count": ks_drift_count,
            "feature_details": sorted(feature_reports, key=lambda x: x["psi"], reverse=True),
            "offline_monitoring_notice": (
                "NOTE: This evaluation represents offline, batch-based statistical distribution comparison. "
                "It does NOT indicate a live, continuous production sensor feed."
            ),
            "drift_vs_accuracy_disclaimer": (
                "CAUTION: Feature distribution shift (covariate shift) does not necessarily imply immediate degradation "
                "in model accuracy. A formal performance audit requires ground-truth outcome labels."
            )
        }
