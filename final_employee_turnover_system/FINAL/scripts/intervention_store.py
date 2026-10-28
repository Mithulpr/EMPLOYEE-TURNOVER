"""
Unified HR Intervention Tracking Module
=======================================
Provides persistent JSON-backed storage and management of HR retention interventions:
- Anonymized employee references
- Pre-intervention model turnover risk
- Intervention categories & workflows
- Genuine observational outcome tracking
- Rigorous academic causal disclaimer banner
"""

import os
import json
import time
import random
from datetime import datetime
from typing import Dict, Any, List, Optional


class HRInterventionStore:
    """
    Manages persistent storage and querying of HR intervention records.
    """
    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            p1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "hr_interventions.json")
            p2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hr_interventions.json")
            data_path = p1 if os.path.exists(p1) else p2
        self.data_path = data_path
        self._ensure_storage_file()

    def _ensure_storage_file(self):
        """Ensures the storage JSON file exists."""
        if not os.path.exists(self.data_path):
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            initial_records = [
                {
                    "id": "INT-2026-001",
                    "employee_ref": "EMP_ANON_1042",
                    "department": "Research & Development",
                    "turnover_probability": 0.785,
                    "risk_tier": "Critical",
                    "prediction_date": "2026-08-15",
                    "hr_partner": "Sarah Jenkins (Senior HRBP)",
                    "intervention_type": "Career Growth Pathway & Promotion Fast-Track",
                    "action_status": "Intervention Ongoing",
                    "follow_up_date": "2026-10-15",
                    "observed_outcome": "Pending Evaluation",
                    "intervention_notes": "Identified high promotion lag score. Scheduled review with VP of Engineering for technical track advancement.",
                    "last_updated": "2026-09-10 14:30:00"
                },
                {
                    "id": "INT-2026-002",
                    "employee_ref": "EMP_ANON_2891",
                    "department": "Sales",
                    "turnover_probability": 0.692,
                    "risk_tier": "High",
                    "prediction_date": "2026-07-20",
                    "hr_partner": "Marcus Reed (People Partner)",
                    "intervention_type": "Workload & Travel Rebalancing",
                    "action_status": "Completed & Monitored",
                    "follow_up_date": "2026-09-20",
                    "observed_outcome": "Retained (Active)",
                    "intervention_notes": "Employee exhibited extreme travel frequency. Reassigned domestic accounts to reduce travel overhead by 40%.",
                    "last_updated": "2026-09-21 09:15:00"
                },
                {
                    "id": "INT-2026-003",
                    "employee_ref": "EMP_ANON_3319",
                    "department": "Human Resources",
                    "turnover_probability": 0.724,
                    "risk_tier": "Critical",
                    "prediction_date": "2026-06-10",
                    "hr_partner": "Elena Rostova (HR Director)",
                    "intervention_type": "Compensation & Market Adjustment",
                    "action_status": "Completed & Monitored",
                    "follow_up_date": "2026-08-10",
                    "observed_outcome": "Retained (Active)",
                    "intervention_notes": "Comp-ratio review indicated compensation below 35th percentile for job role. Base salary corrected by +12%.",
                    "last_updated": "2026-08-12 16:45:00"
                },
                {
                    "id": "INT-2026-004",
                    "employee_ref": "EMP_ANON_0754",
                    "department": "Sales",
                    "turnover_probability": 0.812,
                    "risk_tier": "Critical",
                    "prediction_date": "2026-05-18",
                    "hr_partner": "Marcus Reed (People Partner)",
                    "intervention_type": "Manager Mediation & Role Realignment",
                    "action_status": "Completed & Monitored",
                    "follow_up_date": "2026-07-18",
                    "observed_outcome": "Resigned",
                    "intervention_notes": "Severe dissatisfaction flag active. Mediation conducted but external offer accepted.",
                    "last_updated": "2026-07-22 11:00:00"
                }
            ]
            with open(self.data_path, "w") as f:
                json.dump(initial_records, f, indent=2)

    def load_all(self, status_filter: Optional[str] = None, risk_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Loads all records, optionally filtering by status or risk tier."""
        with open(self.data_path, "r") as f:
            records = json.load(f)

        if status_filter:
            records = [r for r in records if r.get("action_status") == status_filter]
        if risk_filter:
            records = [r for r in records if r.get("risk_tier") == risk_filter]

        return records

    def add_intervention(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates and appends a new intervention record."""
        records = self.load_all()

        next_idx = len(records) + 1
        new_id = f"INT-2026-{next_idx:03d}"

        new_record = {
            "id": new_id,
            "employee_ref": data.get("employee_ref", f"EMP_ANON_{random.randint(1000, 9999)}"),
            "department": data.get("department", "Unassigned"),
            "turnover_probability": float(data.get("turnover_probability", 0.5)),
            "risk_tier": data.get("risk_tier", "Moderate"),
            "prediction_date": data.get("prediction_date", datetime.now().strftime("%Y-%m-%d")),
            "hr_partner": data.get("hr_partner", "HR Operations"),
            "intervention_type": data.get("intervention_type", "Standard Engagement Review"),
            "action_status": data.get("action_status", "Action Planned"),
            "follow_up_date": data.get("follow_up_date", ""),
            "observed_outcome": data.get("observed_outcome", "Pending Evaluation"),
            "intervention_notes": data.get("intervention_notes", ""),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        records.insert(0, new_record)
        with open(self.data_path, "w") as f:
            json.dump(records, f, indent=2)

        return new_record

    def update_intervention(self, record_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates an existing intervention record."""
        records = self.load_all()
        updated_rec = None

        for rec in records:
            if rec["id"] == record_id:
                for k, v in updates.items():
                    if k != "id":
                        rec[k] = v
                rec["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                updated_rec = rec
                break

        if updated_rec:
            with open(self.data_path, "w") as f:
                json.dump(records, f, indent=2)

        return updated_rec

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates operational summary statistics across all logged interventions."""
        records = self.load_all()
        total = len(records)

        status_counts = {}
        outcome_counts = {}
        type_counts = {}

        for r in records:
            st = r.get("action_status", "Unknown")
            status_counts[st] = status_counts.get(st, 0) + 1

            out = r.get("observed_outcome", "Unknown")
            outcome_counts[out] = outcome_counts.get(out, 0) + 1

            it = r.get("intervention_type", "Unknown")
            type_counts[it] = type_counts.get(it, 0) + 1

        completed_count = status_counts.get("Completed & Monitored", 0)
        retained_count = outcome_counts.get("Retained (Active)", 0)
        obs_retention_rate = (retained_count / (retained_count + outcome_counts.get("Resigned", 0))) * 100 if (retained_count + outcome_counts.get("Resigned", 0)) > 0 else 0.0

        return {
            "total_interventions": total,
            "status_breakdown": status_counts,
            "outcome_breakdown": outcome_counts,
            "intervention_types": type_counts,
            "completed_interventions": completed_count,
            "observed_retention_rate_pct": round(obs_retention_rate, 1),
            "causality_disclaimer": (
                "METHODOLOGICAL NOTICE: Recorded retention rates represent observational follow-up outcomes only. "
                "They do NOT prove that the retention was caused by the HR intervention. Without a randomized control "
                "group or formal quasi-experimental propensity score matching, confounding factors cannot be ruled out."
            )
        }
