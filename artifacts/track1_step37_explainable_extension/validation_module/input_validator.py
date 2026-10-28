"""
Track 1 — Step 37: Reusable Input Data Validation and Survey Response Bias Audit Module
Ensures data quality, schema integrity, and detects potential response bias before prediction.
"""

import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class HRDataValidator:
    """
    Validates employee record schemas, data types, value bounds, and inspects
    satisfaction survey indicators for potential response anomalies or bias.
    """

    EXPECTED_RAW_COLUMNS = [
        "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
        "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
        "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
        "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
        "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
        "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StandardHours", "StockOptionLevel",
        "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

    EXPECTED_30_FEATURES = [
        "Age", "EnvironmentSatisfaction", "JobSatisfaction", "TotalWorkingYears",
        "TrainingTimesLastYear", "WorkLifeBalance", "YearsSinceLastPromotion",
        "YearsWithCurrManager", "TotalExpToAgeRatio", "CompanyLoyaltyRatio",
        "PriorCompanyExperience", "JobHoppingRate", "PromotionStagnationRatio",
        "PromotionLagScore", "ManagerTenureMismatch", "WorkLifeDeficit",
        "SevereDissatisfactionFlag", "EarlyCareerFlightRisk", "PromotionVelocity",
        "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Frequently",
        "BusinessTravel_Travel_Rarely", "EducationField_Life Sciences",
        "EducationField_Marketing", "EducationField_Other", "Gender_Female",
        "Gender_Male", "JobRole_Human Resources", "MaritalStatus_Single",
        "PromotionPaceStrain"
    ]

    NUMERICAL_BOUNDS = {
        "Age": (18, 70),
        "TotalWorkingYears": (0, 50),
        "YearsAtCompany": (0, 50),
        "YearsWithCurrManager": (0, 30),
        "YearsSinceLastPromotion": (0, 30),
        "JobSatisfaction": (1, 4),
        "EnvironmentSatisfaction": (1, 4),
        "WorkLifeBalance": (1, 4),
        "JobInvolvement": (1, 4),
        "MonthlyIncome": (1000, 25000),
        "NumCompaniesWorked": (0, 20)
    }

    def __init__(self):
        pass

    def validate_schema(self, df: pd.DataFrame, expected_type: str = "30_features") -> Dict[str, Any]:
        """
        Validates column names, missing columns, and unexpected schemas.
        expected_type: '30_features' or 'raw_columns'
        """
        target_cols = self.EXPECTED_30_FEATURES if expected_type == "30_features" else self.EXPECTED_RAW_COLUMNS
        present_cols = set(df.columns)
        missing_cols = [c for c in target_cols if c not in present_cols and c != "Attrition"]
        extra_cols = [c for c in df.columns if c not in target_cols and c not in ["EmployeeID", "EmployeeNumber", "Attrition"]]

        status = "PASSED" if len(missing_cols) == 0 else "FAILED"
        return {
            "status": status,
            "expected_type": expected_type,
            "total_columns": len(df.columns),
            "missing_columns_count": len(missing_cols),
            "missing_columns": missing_cols,
            "extra_columns_count": len(extra_cols),
            "extra_columns": extra_cols[:10]
        }

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Inspects missing values, duplicates, and range violations.
        """
        issues = []
        # Missing values
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        if cols_with_nulls:
            issues.append(f"Missing values detected in {len(cols_with_nulls)} columns: {cols_with_nulls}")

        # Duplicates
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            issues.append(f"Duplicate rows detected: {duplicate_count} identical records found.")

        # Range violations
        range_violations = {}
        for col, (min_v, max_v) in self.NUMERICAL_BOUNDS.items():
            if col in df.columns:
                out_of_bounds = df[(df[col] < min_v) | (df[col] > max_v)]
                if len(out_of_bounds) > 0:
                    range_violations[col] = {
                        "expected_range": [min_v, max_v],
                        "violating_records": len(out_of_bounds),
                        "observed_min": float(df[col].min()),
                        "observed_max": float(df[col].max())
                    }
                    issues.append(f"Column '{col}' has {len(out_of_bounds)} values outside valid bounds [{min_v}, {max_v}].")

        status = "CLEAN" if len(issues) == 0 else "WARNINGS_DETECTED"
        return {
            "status": status,
            "total_records": len(df),
            "issues_count": len(issues),
            "issues": issues,
            "missing_values_by_column": cols_with_nulls,
            "duplicate_records": duplicate_count,
            "range_violations": range_violations
        }

    def audit_survey_response_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Conducts limitation-aware survey response consistency checks.
        Detects conflicting signals without attributing intent or dishonesty.
        """
        bias_flags = []
        n_total = len(df)

        # Signal 1: Extreme Satisfaction Paradox
        # High stated satisfaction (JobSatisfaction=4, EnvSatisfaction=4) alongside rapid promotion lag and high stagnation
        if "JobSatisfaction" in df.columns and "EnvironmentSatisfaction" in df.columns:
            if "YearsSinceLastPromotion" in df.columns and "YearsAtCompany" in df.columns:
                paradox_mask = (
                    (df["JobSatisfaction"] == 4) &
                    (df["EnvironmentSatisfaction"] == 4) &
                    (df["YearsSinceLastPromotion"] >= 7) &
                    (df["YearsAtCompany"] >= 10)
                )
                count_p = int(paradox_mask.sum())
                if count_p > 0:
                    bias_flags.append({
                        "flag_id": "SATISFACTION_STAGNATION_DIVERGENCE",
                        "severity": "INFORMATIONAL",
                        "description": "High stated satisfaction despite prolonged stagnation (>=7 years without promotion in >=10 year tenure).",
                        "affected_count": count_p,
                        "percentage": round(count_p / n_total * 100, 2),
                        "hr_recommendation": "Conduct exploratory 1-on-1 check-in; do not assume retention based solely on survey score."
                    })

        # Signal 2: Dissatisfaction without Administrative Signs
        # Lowest satisfaction (1/1) alongside very recent promotion (<= 1 yr) and short manager tenure
        if "JobSatisfaction" in df.columns and "YearsSinceLastPromotion" in df.columns:
            if "YearsWithCurrManager" in df.columns:
                recent_promo_dissat = (
                    (df["JobSatisfaction"] == 1) &
                    (df["YearsSinceLastPromotion"] <= 1) &
                    (df["YearsWithCurrManager"] <= 1)
                )
                count_d = int(recent_promo_dissat.sum())
                if count_d > 0:
                    bias_flags.append({
                        "flag_id": "NEW_ROLE_TRANSITION_DISSATISFACTION",
                        "severity": "REVIEW_RECOMMENDED",
                        "description": "Low job satisfaction reported immediately following a recent promotion or manager change (<=1 year).",
                        "affected_count": count_d,
                        "percentage": round(count_d / n_total * 100, 2),
                        "hr_recommendation": "Review role adjustment and onboarding support; dissatisfaction may reflect temporary transition stress."
                    })

        # Signal 3: Uniform Response Bias (Straight-lining across survey scales)
        survey_cols = [c for c in ["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance", "JobInvolvement", "RelationshipSatisfaction"] if c in df.columns]
        if len(survey_cols) >= 3:
            # Check if all survey items have the identical rating
            survey_vals = df[survey_cols].values
            is_straightline = np.all(survey_vals == survey_vals[:, [0]], axis=1)
            count_sl = int(is_straightline.sum())
            if count_sl > 0:
                bias_flags.append({
                    "flag_id": "UNIFORM_RESPONSE_STRAIGHTLINING",
                    "severity": "DATA_QUALITY_NOTE",
                    "description": "Identical rating submitted across all satisfaction and involvement questions (potential survey fatigue).",
                    "affected_count": count_sl,
                    "percentage": round(count_sl / n_total * 100, 2),
                    "hr_recommendation": "Flag for qualitative verification. Use multi-signal HR metrics rather than survey scores alone."
                })

        return {
            "survey_audit_status": "COMPLETED",
            "records_audited": n_total,
            "anomaly_flags_identified": len(bias_flags),
            "flags": bias_flags,
            "scientific_disclaimer": (
                "Data quality flags identify divergence between self-reported survey indicators and administrative records. "
                "They reflect survey response patterns or role transition effects, NOT evidence of employee dishonesty. "
                "Because historical benchmark data is cross-sectional (single-observation per employee), temporal "
                "response consistency cannot be tested without longitudinal survey panels."
            )
        }
