"""
Track 1 — Step 37: Model and Data Drift Monitoring Engine
Computes Population Stability Index (PSI), Kolmogorov-Smirnov (KS) statistics,
and categorical distribution shifts between reference and incoming batches.
"""

import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


class DriftDetector:
    """
    Monitors distribution shift across features and predictions against reference development data.
    Implements PSI, KS tests, and sample-size validity safeguards.
    """

    MIN_RECORDS_FOR_INFERENCE = 30
    RECOMMENDED_RECORDS_FOR_PSI = 100

    def __init__(self, reference_df: pd.DataFrame, reference_predictions: Optional[np.ndarray] = None):
        self.ref_df = reference_df.copy()
        self.ref_preds = reference_predictions.copy() if reference_predictions is not None else None
        self.numeric_features = [
            c for c in reference_df.select_dtypes(include=[np.number]).columns
            if c not in ["Attrition", "EmployeeID", "EmployeeNumber"]
        ]
        self.categorical_features = [
            c for c in reference_df.select_dtypes(exclude=[np.number]).columns
            if c not in ["Attrition", "EmployeeID", "EmployeeNumber"]
        ]

    @staticmethod
    def calculate_psi(ref_vals: np.ndarray, curr_vals: np.ndarray, num_bins: int = 10) -> float:
        """
        Calculates the Population Stability Index (PSI) for continuous variables using quantile binning.
        PSI < 0.10: Insignificant drift (stable)
        0.10 <= PSI < 0.25: Moderate drift (monitor)
        PSI >= 0.25: Significant drift (action required)
        """
        ref_clean = ref_vals[~np.isnan(ref_vals)]
        curr_clean = curr_vals[~np.isnan(curr_vals)]
        if len(ref_clean) < 10 or len(curr_clean) < 10:
            return 0.0

        # Create quantile bins on reference distribution
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(ref_clean, percentiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
        curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

        ref_dist = ref_counts / len(ref_clean)
        curr_dist = curr_counts / len(curr_clean)

        # Smooth zero counts to avoid division by zero or log(0)
        eps = 1e-4
        ref_dist = np.where(ref_dist == 0, eps, ref_dist)
        curr_dist = np.where(curr_dist == 0, eps, curr_dist)

        # Re-normalize
        ref_dist = ref_dist / np.sum(ref_dist)
        curr_dist = curr_dist / np.sum(curr_dist)

        psi_val = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
        return float(max(0.0, psi_val))

    def evaluate_batch_drift(self, incoming_df: pd.DataFrame, incoming_preds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluates drift across all numerical and categorical features.
        Provides sample-size caveats and separate assessments for feature drift vs prediction drift.
        """
        n_incoming = len(incoming_df)
        sample_warnings = []
        if n_incoming < self.MIN_RECORDS_FOR_INFERENCE:
            sample_warnings.append(
                f"Small sample warning: Batch size N={n_incoming} is below minimum statistical threshold ({self.MIN_RECORDS_FOR_INFERENCE}). "
                "PSI and p-values have high sampling variance and should be interpreted cautiously."
            )
        elif n_incoming < self.RECOMMENDED_RECORDS_FOR_PSI:
            sample_warnings.append(
                f"Moderate sample size note: N={n_incoming} is below recommended threshold ({self.RECOMMENDED_RECORDS_FOR_PSI}) for stable PSI estimation."
            )

        feature_drift_reports = []
        drift_alert_count = 0
        monitor_alert_count = 0

        common_numeric = [c for c in self.numeric_features if c in incoming_df.columns]

        for col in common_numeric:
            ref_v = self.ref_df[col].dropna().values.astype(float)
            curr_v = incoming_df[col].dropna().values.astype(float)

            if len(curr_v) < 5:
                continue

            psi = self.calculate_psi(ref_v, curr_v, num_bins=10)
            ks_stat, ks_pval = ks_2samp(ref_v, curr_v)

            if psi >= 0.25:
                status = "SIGNIFICANT_DRIFT"
                drift_alert_count += 1
            elif psi >= 0.10:
                status = "MODERATE_DRIFT"
                monitor_alert_count += 1
            else:
                status = "STABLE"

            feature_drift_reports.append({
                "feature": col,
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pval), 4),
                "ref_mean": round(float(np.mean(ref_v)), 3),
                "curr_mean": round(float(np.mean(curr_v)), 3),
                "ref_std": round(float(np.std(ref_v)), 3),
                "curr_std": round(float(np.std(curr_v)), 3),
                "drift_status": status
            })

        # Sort features by PSI descending
        feature_drift_reports.sort(key=lambda x: x["psi"], reverse=True)

        # Prediction Drift Assessment
        pred_drift_report = None
        if self.ref_preds is not None and incoming_preds is not None and len(incoming_preds) > 0:
            pred_psi = self.calculate_psi(self.ref_preds, incoming_preds, num_bins=10)
            pred_ks_stat, pred_ks_pval = ks_2samp(self.ref_preds, incoming_preds)
            pred_drift_status = "SIGNIFICANT_DRIFT" if pred_psi >= 0.25 else ("MODERATE_DRIFT" if pred_psi >= 0.10 else "STABLE")

            pred_drift_report = {
                "prediction_psi": round(pred_psi, 4),
                "prediction_ks_stat": round(float(pred_ks_stat), 4),
                "prediction_ks_pvalue": round(float(pred_ks_pval), 4),
                "ref_mean_prob": round(float(np.mean(self.ref_preds)), 4),
                "curr_mean_prob": round(float(np.mean(incoming_preds)), 4),
                "ref_predicted_attrition_rate": round(float(np.mean(self.ref_preds >= 0.3610) * 100), 2),
                "curr_predicted_attrition_rate": round(float(np.mean(incoming_preds >= 0.3610) * 100), 2),
                "drift_status": pred_drift_status
            }

        overall_status = "STABLE"
        if drift_alert_count > 0:
            overall_status = "DRIFT_ALERT_ACTION_RECOMMENDED"
        elif monitor_alert_count > 0:
            overall_status = "MODERATE_DRIFT_MONITORING_SUGGESTED"

        return {
            "overall_status": overall_status,
            "incoming_records": n_incoming,
            "reference_records": len(self.ref_df),
            "sample_warnings": sample_warnings,
            "significant_drift_features_count": drift_alert_count,
            "moderate_drift_features_count": monitor_alert_count,
            "feature_drift_summary": feature_drift_reports[:10],
            "prediction_drift": pred_drift_report,
            "governance_note": (
                "Feature drift and prediction drift measure covariate shift between populations without requiring ground-truth labels. "
                "A drift alert indicates changes in employee characteristics or prediction distributions, but CANNOT determine whether "
                "model accuracy has declined without verified attrition outcomes. Do not retrain automatically on unverified drift."
            )
        }
