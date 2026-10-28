"""
Research Gap Extension: Advanced HR-Domain Feature Engineering Module
======================================================================
Addresses the future research gaps highlighted in Talebi et al. (2025):
- Domain-driven interaction features
- Non-linear composite career progression & compensation indicators
- Mathematically guarded denominators (zero-division safe)
- Zero label leakage: completely independent of target (Attrition) labels
- Scikit-learn compatible transformer fitted strictly on training partitions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin


# List of the official Step 32 baseline features (30 features)
STEP32_BASELINE_FEATURES = [
    "Age",
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TotalExpToAgeRatio",
    "CompanyLoyaltyRatio",
    "PriorCompanyExperience",
    "JobHoppingRate",
    "PromotionStagnationRatio",
    "PromotionLagScore",
    "ManagerTenureMismatch",
    "WorkLifeDeficit",
    "SevereDissatisfactionFlag",
    "EarlyCareerFlightRisk",
    "PromotionVelocity",
    "BusinessTravel_Non-Travel",
    "BusinessTravel_Travel_Frequently",
    "BusinessTravel_Travel_Rarely",
    "EducationField_Life Sciences",
    "EducationField_Marketing",
    "EducationField_Other",
    "Gender_Female",
    "Gender_Male",
    "JobRole_Human Resources",
    "MaritalStatus_Single",
    "PromotionPaceStrain",
]

# Newly engineered candidate HR-domain features
CANDIDATE_HR_FEATURES = [
    "ExperienceToAgeRatio",
    "PromotionCadence",
    "Overtime_JobSatisfaction_Interaction",
    "WorkLifeBalance_OverTime_Interaction",
    "CompensationCareerParity",
    "MeritHikeCadence",
    "DualDissatisfactionRisk",
    "CompensationPerTenureYear",
    "ManagerTenureRatio",
    "StockStabilityIndex",
]


class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-safe HR feature engineering transformer.
    Fitted statistics (medians for missing value handling) are learned exclusively on training data.
    """

    def __init__(self, feature_set: str = "enriched"):
        """
        feature_set:
            - 'baseline': original Step 32 feature set
            - 'enriched': Step 32 + all candidate HR features
            - 'selected': Step 32 + validated subset of candidate HR features
        """
        self.feature_set = feature_set
        self.fitted_medians_: Dict[str, float] = {}
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y=None):
        """Learns training set distributions (medians for missing values)."""
        df = X.copy()
        for col in ["YearsWithCurrManager", "MonthlyIncome", "TotalWorkingYears", "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance"]:
            if col in df.columns:
                self.fitted_medians_[col] = float(df[col].median(skipna=True))
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw cleaned DataFrame into engineered feature representation."""
        if not self.is_fitted_:
            raise ValueError("HRFeatureEngineer must be fitted on training data before transform.")

        df = X.copy()

        # Handle numeric OverTime representation
        if "OverTime" in df.columns:
            if df["OverTime"].dtype == object:
                ot_num = (df["OverTime"].astype(str).str.strip().str.lower() == "yes").astype(float)
            else:
                ot_num = (df["OverTime"] > 0).astype(float)
        else:
            ot_num = pd.Series(0.0, index=df.index)

        # Impute missing values with learned training medians
        curr_mgr = df["YearsWithCurrManager"].fillna(self.fitted_medians_.get("YearsWithCurrManager", 3.0)) if "YearsWithCurrManager" in df.columns else pd.Series(3.0, index=df.index)
        income = df["MonthlyIncome"].fillna(self.fitted_medians_.get("MonthlyIncome", 4900.0)) if "MonthlyIncome" in df.columns else pd.Series(4900.0, index=df.index)
        twy = df["TotalWorkingYears"].fillna(self.fitted_medians_.get("TotalWorkingYears", 10.0)) if "TotalWorkingYears" in df.columns else pd.Series(10.0, index=df.index)
        job_sat = df["JobSatisfaction"].fillna(self.fitted_medians_.get("JobSatisfaction", 3.0)) if "JobSatisfaction" in df.columns else pd.Series(3.0, index=df.index)
        env_sat = df["EnvironmentSatisfaction"].fillna(self.fitted_medians_.get("EnvironmentSatisfaction", 3.0)) if "EnvironmentSatisfaction" in df.columns else pd.Series(3.0, index=df.index)
        wlb = df["WorkLifeBalance"].fillna(self.fitted_medians_.get("WorkLifeBalance", 3.0)) if "WorkLifeBalance" in df.columns else pd.Series(3.0, index=df.index)
        yac = df["YearsAtCompany"].fillna(5.0) if "YearsAtCompany" in df.columns else pd.Series(5.0, index=df.index)
        ysp = df["YearsSinceLastPromotion"].fillna(1.0) if "YearsSinceLastPromotion" in df.columns else pd.Series(1.0, index=df.index)
        job_level = df["JobLevel"].fillna(2.0) if "JobLevel" in df.columns else pd.Series(2.0, index=df.index)
        age = df["Age"].fillna(36.0) if "Age" in df.columns else pd.Series(36.0, index=df.index)
        hike = df["PercentSalaryHike"].fillna(14.0) if "PercentSalaryHike" in df.columns else pd.Series(14.0, index=df.index)
        stock = df["StockOptionLevel"].fillna(1.0) if "StockOptionLevel" in df.columns else pd.Series(1.0, index=df.index)

        # 1. Step 32 baseline engineered features
        df["TotalExpToAgeRatio"] = twy / (age + 1.0)
        df["CompanyLoyaltyRatio"] = yac / (twy + 1.0)
        df["PriorCompanyExperience"] = np.maximum(0.0, twy - yac)
        num_comp = df["NumCompaniesWorked"].fillna(2.0) if "NumCompaniesWorked" in df.columns else pd.Series(2.0, index=df.index)
        df["JobHoppingRate"] = num_comp / (twy + 1.0)
        df["PromotionStagnationRatio"] = ysp / (yac + 1.0)
        df["PromotionLagScore"] = np.maximum(0.0, ysp - 3.0)
        df["ManagerTenureMismatch"] = np.abs(curr_mgr - (yac / 2.0))
        df["WorkLifeDeficit"] = np.maximum(0.0, 3.0 - wlb)
        df["SevereDissatisfactionFlag"] = ((job_sat <= 1.0) | (env_sat <= 1.0)).astype(float)
        df["EarlyCareerFlightRisk"] = ((age <= 30.0) & (twy <= 4.0)).astype(float)
        df["PromotionVelocity"] = ysp / (yac + 1.0)
        burnout_raw = (ot_num * 2.0) + df["WorkLifeDeficit"] + (5.0 - job_sat)
        df["PromotionPaceStrain"] = df["PromotionVelocity"] * burnout_raw

        # Categorical one-hot features if present
        if "BusinessTravel" in df.columns:
            bt = df["BusinessTravel"].astype(str).str.strip()
            df["BusinessTravel_Non-Travel"] = (bt == "Non-Travel").astype(float)
            df["BusinessTravel_Travel_Frequently"] = (bt == "Travel_Frequently").astype(float)
            df["BusinessTravel_Travel_Rarely"] = (bt == "Travel_Rarely").astype(float)
        else:
            df["BusinessTravel_Non-Travel"] = 0.0
            df["BusinessTravel_Travel_Frequently"] = 0.0
            df["BusinessTravel_Travel_Rarely"] = 1.0

        if "EducationField" in df.columns:
            ef = df["EducationField"].astype(str).str.strip()
            df["EducationField_Life Sciences"] = (ef == "Life Sciences").astype(float)
            df["EducationField_Marketing"] = (ef == "Marketing").astype(float)
            df["EducationField_Other"] = (ef == "Other").astype(float)
        else:
            df["EducationField_Life Sciences"] = 0.0
            df["EducationField_Marketing"] = 0.0
            df["EducationField_Other"] = 0.0

        if "Gender" in df.columns:
            g = df["Gender"].astype(str).str.strip()
            df["Gender_Female"] = (g == "Female").astype(float)
            df["Gender_Male"] = (g == "Male").astype(float)
        else:
            df["Gender_Female"] = 0.0
            df["Gender_Male"] = 1.0

        if "JobRole" in df.columns:
            jr = df["JobRole"].astype(str).str.strip()
            df["JobRole_Human Resources"] = (jr == "Human Resources").astype(float)
        else:
            df["JobRole_Human Resources"] = 0.0

        if "MaritalStatus" in df.columns:
            ms = df["MaritalStatus"].astype(str).str.strip()
            df["MaritalStatus_Single"] = (ms == "Single").astype(float)
        else:
            df["MaritalStatus_Single"] = 0.0

        # 2. Newly engineered candidate HR-domain features
        # F1: ExperienceToAgeRatio - Adult career progression ratio
        df["ExperienceToAgeRatio"] = twy / np.maximum(age - 17.0, 1.0)

        # F2: PromotionCadence - Relative frequency of promotions over company tenure
        df["PromotionCadence"] = ysp / (yac + 1.0)

        # F3: Overtime_JobSatisfaction_Interaction - Compounding burnout risk
        df["Overtime_JobSatisfaction_Interaction"] = ot_num * (5.0 - job_sat)

        # F4: WorkLifeBalance_OverTime_Interaction - Acute work-life friction from mandatory overtime
        df["WorkLifeBalance_OverTime_Interaction"] = ot_num * (5.0 - wlb)

        # F5: CompensationCareerParity - Compensation normalized against career stage and seniority
        df["CompensationCareerParity"] = income / ((twy + 1.0) * (job_level + 0.5))

        # F6: MeritHikeCadence - Salary growth pace normalized by promotion latency
        df["MeritHikeCadence"] = hike / (ysp + 1.0)

        # F7: DualDissatisfactionRisk - Simultaneous job and environment dissatisfaction
        df["DualDissatisfactionRisk"] = (5.0 - job_sat) * (5.0 - env_sat)

        # F8: CompensationPerTenureYear - Direct compensation return on organizational loyalty
        df["CompensationPerTenureYear"] = income / (yac + 1.0)

        # F9: ManagerTenureRatio - Supervisory stability index
        df["ManagerTenureRatio"] = curr_mgr / (yac + 1.0)

        # F10: StockStabilityIndex - Long-term financial retention lock relative to job level
        df["StockStabilityIndex"] = stock / (job_level + 0.5)

        return df

    def get_feature_names(self) -> List[str]:
        """Returns the list of feature column names for the configured feature set."""
        if self.feature_set == "baseline":
            return STEP32_BASELINE_FEATURES
        elif self.feature_set == "enriched":
            # Combine baseline and all 10 candidate features (avoiding duplicates)
            combined = list(STEP32_BASELINE_FEATURES)
            for f in CANDIDATE_HR_FEATURES:
                if f not in combined:
                    combined.append(f)
            return combined
        elif self.feature_set == "selected":
            # Top-performing features identified through ablation screening
            selected_candidates = [
                "Overtime_JobSatisfaction_Interaction",
                "WorkLifeBalance_OverTime_Interaction",
                "DualDissatisfactionRisk",
                "ManagerTenureRatio",
                "CompensationCareerParity"
            ]
            combined = list(STEP32_BASELINE_FEATURES)
            for f in selected_candidates:
                if f not in combined:
                    combined.append(f)
            return combined
        else:
            raise ValueError(f"Unknown feature_set: {self.feature_set}")


def get_feature_registry_metadata() -> Dict[str, Any]:
    """Generates exhaustive metadata for the feature registry."""
    return {
        "registry_version": "1.0.0",
        "description": "HR-Domain Feature Engineering Registry for Research Gap Extension (Talebi et al. 2025)",
        "baseline_feature_count": len(STEP32_BASELINE_FEATURES),
        "candidate_feature_count": len(CANDIDATE_HR_FEATURES),
        "total_enriched_feature_count": len(STEP32_BASELINE_FEATURES) + len([f for f in CANDIDATE_HR_FEATURES if f not in STEP32_BASELINE_FEATURES]),
        "features": {
            "ExperienceToAgeRatio": {
                "formula": "TotalWorkingYears / max(Age - 17.0, 1.0)",
                "domain_rationale": "Quantifies career acceleration and early workforce entry versus late start or career gaps.",
                "input_columns": ["TotalWorkingYears", "Age"],
                "zero_guard": "max(Age - 17.0, 1.0)",
                "leakage_status": "PASSED (Zero target dependence)",
                "category": "Career Progression"
            },
            "PromotionCadence": {
                "formula": "YearsSinceLastPromotion / (YearsAtCompany + 1.0)",
                "domain_rationale": "Measures promotion latency proportional to organizational tenure.",
                "input_columns": ["YearsSinceLastPromotion", "YearsAtCompany"],
                "zero_guard": "YearsAtCompany + 1.0",
                "leakage_status": "PASSED (Zero target dependence)",
                "category": "Career Progression"
            },
            "Overtime_JobSatisfaction_Interaction": {
                "formula": "OverTime_binary * (5.0 - JobSatisfaction)",
                "domain_rationale": "Compounding burnout risk: evaluates severe dissatisfaction under extended overtime load.",
                "input_columns": ["OverTime", "JobSatisfaction"],
                "zero_guard": "Direct scalar multiplication",
                "leakage_status": "PASSED (Zero target dependence)",
                "category": "Burnout & Satisfaction"
            },
            "WorkLifeBalance_OverTime_Interaction": {
                "formula": "OverTime_binary * (5.0 - WorkLifeBalance)",
                "domain_rationale": "Captures personal life strain when mandatory overtime clashes with poor work-life balance.",
                "input_columns": ["OverTime", "WorkLifeBalance"],
                "zero_guard": "Direct scalar multiplication",
                "leakage_status": "PASSED (Zero target dependence)",
                "category": "Burnout & Satisfaction"
            },
            "CompensationCareerParity": {
                "formula": "MonthlyIncome / ((TotalWorkingYears + 1.0) * (JobLevel + 0.5))",
                "domain_rationale": "Normalizes income against cumulative career investment and organizational hierarchy.",
                "input_columns": ["MonthlyIncome", "TotalWorkingYears", "JobLevel"],
                "zero_guard": "(TotalWorkingYears + 1.0) * (JobLevel + 0.5)",
                "leakage_status": "PASSED (Zero target dependence)",
                "category": "Compensation Equity"
            },
            "MeritHikeCadence": {
                "formula": "PercentSalaryHike / (YearsSinceLastPromotion + 1.0)",
                "domain_rationale": "Assesses whether merit salary increases offset promotional stagnation.",
                "input_columns": ["PercentSalaryHike", "YearsSinceLastPromotion"],
                "zero_guard": "YearsSinceLastPromotion + 1.0",
                "leakage_status": "PASSED (Zero target dependence)",
                "category": "Compensation Equity"
            },
            "DualDissatisfactionRisk": {
                "formula": "(5.0 - JobSatisfaction) * (5.0 - EnvironmentSatisfaction)",
                "domain_rationale": "Quantifies compounding organizational dissatisfaction across work duties and physical culture.",
                "input_columns": ["JobSatisfaction", "EnvironmentSatisfaction"],
                "zero_guard": "Direct scalar multiplication",
                "leakage_status": "PASSED (Zero target dependence)",
                "category": "Burnout & Satisfaction"
            },
            "CompensationPerTenureYear": {
                "formula": "MonthlyIncome / (YearsAtCompany + 1.0)",
                "domain_rationale": "Measures compensation accrual rate relative to tenure with current employer.",
                "input_columns": ["MonthlyIncome", "YearsAtCompany"],
                "zero_guard": "YearsAtCompany + 1.0",
                "leakage_status": "PASSED (Zero target dependence)",
                "category": "Compensation Equity"
            },
            "ManagerTenureRatio": {
                "formula": "YearsWithCurrManager / (YearsAtCompany + 1.0)",
                "domain_rationale": "Supervisory stability index: measures continuity of reporting relationship relative to tenure.",
                "input_columns": ["YearsWithCurrManager", "YearsAtCompany"],
                "zero_guard": "YearsAtCompany + 1.0",
                "leakage_status": "PASSED (Zero target dependence)",
                "category": "Supervisory Stability"
            },
            "StockStabilityIndex": {
                "formula": "StockOptionLevel / (JobLevel + 0.5)",
                "domain_rationale": "Equity retention anchor: evaluates stock ownership relative to job hierarchy level.",
                "input_columns": ["StockOptionLevel", "JobLevel"],
                "zero_guard": "JobLevel + 0.5",
                "leakage_status": "PASSED (Zero target dependence)",
                "category": "Retention & Compensation"
            }
        }
    }
