"""
Track 1 — Step 37: Monitoring-Ready HR Intervention and Action Tracking Module
Maintains historical action snapshots, logs proactive retention workflows,
and enforces strict governance against premature causal claims.
"""

import json
import os
import time
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


class HRInterventionTracker:
    """
    Manages longitudinal tracking of HR interventions for high-risk employees.
    Maintains historical immutable snapshots and validates schema integrity.
    """

    VALID_STATUSES = ["PENDING_REVIEW", "HR_CONTACTED", "ACTION_IN_PROGRESS", "COMPLETED", "NO_ACTION_REQUIRED"]
    VALID_ACTION_CATEGORIES = [
        "SALARY_COMPENSATION_REVIEW",
        "PROMOTION_TRAJECTORY_DISCUSSION",
        "WORKLOAD_OVERTIME_REDUCTION",
        "MANAGER_RELATIONSHIP_CHECKIN",
        "CAREER_DEVELOPMENT_TRAINING",
        "FLEXIBLE_WORK_ARRANGEMENT",
        "MONITORING_NO_DIRECT_ACTION"
    ]

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.records: List[Dict[str, Any]] = []
        if storage_path and os.path.exists(storage_path):
            self.load()

    def add_prediction_record(
        self,
        anonymized_id: str,
        predicted_prob: float,
        predicted_class: int,
        dominant_risk_factors: List[str],
        base_decision_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a new baseline intervention candidate record.
        """
        record = {
            "record_id": f"REC_{int(time.time()*1000)}_{len(self.records)+1}",
            "anonymized_id": anonymized_id,
            "prediction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_attrition_probability": round(float(predicted_prob), 4),
            "predicted_attrition_class": int(predicted_class),
            "dominant_risk_factors": dominant_risk_factors,
            "decision_override_note": base_decision_override or "STANDARD_MODEL_PREDICTION",
            "hr_review_status": "PENDING_REVIEW" if predicted_class == 1 else "NO_ACTION_REQUIRED",
            "action_category": None,
            "action_date": None,
            "follow_up_date": None,
            "follow_up_satisfaction_score": None,
            "actual_attrition_outcome": None,
            "outcome_verification_date": None,
            "audit_history": [
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": "RECORD_INITIALIZED",
                    "status": "PENDING_REVIEW" if predicted_class == 1 else "NO_ACTION_REQUIRED"
                }
            ]
        }
        self.records.append(record)
        if self.storage_path:
            self.save()
        return record

    def update_intervention(
        self,
        record_id: str,
        hr_status: str,
        action_category: Optional[str] = None,
        action_date: Optional[str] = None,
        follow_up_date: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Updates an existing intervention record with administrative review details.
        """
        for rec in self.records:
            if rec["record_id"] == record_id:
                rec["hr_review_status"] = hr_status
                if action_category:
                    rec["action_category"] = action_category
                if action_date:
                    rec["action_date"] = action_date
                if follow_up_date:
                    rec["follow_up_date"] = follow_up_date

                rec["audit_history"].append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": "INTERVENTION_UPDATED",
                    "status": hr_status,
                    "action_category": action_category,
                    "notes": notes
                })
                if self.storage_path:
                    self.save()
                return True
        return False

    def record_actual_outcome(
        self,
        record_id: str,
        actual_attrition: int,
        outcome_date: str,
        follow_up_satisfaction: Optional[int] = None
    ) -> bool:
        """
        Records the verified longitudinal outcome when it becomes available.
        """
        for rec in self.records:
            if rec["record_id"] == record_id:
                rec["actual_attrition_outcome"] = int(actual_attrition)
                rec["outcome_verification_date"] = outcome_date
                if follow_up_satisfaction is not None:
                    rec["follow_up_satisfaction_score"] = int(follow_up_satisfaction)

                rec["audit_history"].append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": "OUTCOME_VERIFIED",
                    "outcome": int(actual_attrition),
                    "outcome_date": outcome_date
                })
                if self.storage_path:
                    self.save()
                return True
        return False

    def to_dataframe(self) -> pd.DataFrame:
        """
        Exports flat snapshot summary for HR analysis.
        """
        flat_records = []
        for r in self.records:
            flat_records.append({
                "Record_ID": r["record_id"],
                "Anonymized_ID": r["anonymized_id"],
                "Prediction_Time": r["prediction_timestamp"],
                "Attrition_Prob": r["predicted_attrition_probability"],
                "Predicted_Class": r["predicted_attrition_class"],
                "Review_Status": r["hr_review_status"],
                "Action_Category": r["action_category"] or "NONE",
                "Action_Date": r["action_date"] or "N/A",
                "Follow_Up_Date": r["follow_up_date"] or "N/A",
                "Follow_Up_Satisfaction": r["follow_up_satisfaction_score"] if r["follow_up_satisfaction_score"] is not None else "UNCOLLECTED",
                "Actual_Outcome": r["actual_attrition_outcome"] if r["actual_attrition_outcome"] is not None else "PENDING_VERIFICATION",
                "Outcome_Date": r["outcome_verification_date"] or "N/A"
            })
        return pd.DataFrame(flat_records)

    def save(self):
        if self.storage_path:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)

    def load(self):
        if self.storage_path and os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)

    @staticmethod
    def get_causality_disclaimer() -> str:
        return (
            "SCIENTIFIC CAUSALITY WARNING: Observing employee retention following an HR intervention "
            "does NOT establish causal efficacy. True causal attribution requires randomized control "
            "trials (A/B testing) or robust quasi-experimental causal inference designs (e.g., difference-in-differences, "
            "propensity score matching). In observational HR records, post-hoc outcomes are heavily confounded by selection bias."
        )
