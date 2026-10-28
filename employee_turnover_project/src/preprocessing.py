"""
Feature Preprocessing and Domain Transformation Module
======================================================
Implements structural feature engineering, domain interaction engineering,
and transformation pipelines matching the exact Step 32 30-feature schema.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")

# Ensure src is in sys.path and aliased for joblib unpickling
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import src.feature_engineering
    import src.feature_selection
    sys.modules["src.feature_engineering"] = src.feature_engineering
    sys.modules["src.feature_selection"] = src.feature_selection
except ImportError:
    pass


def add_structural_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Computes basic structural tenure and compensation ratios."""
    df_out = df_in.copy()
    if "MonthlyIncome" in df_out.columns and "TotalWorkingYears" in df_out.columns:
        df_out["IncomeToExpRatio"] = df_out["MonthlyIncome"] / (df_out["TotalWorkingYears"] + 1.0)
    if "YearsSinceLastPromotion" in df_out.columns and "YearsAtCompany" in df_out.columns:
        df_out["PromotionVelocity"] = df_out["YearsSinceLastPromotion"] / (df_out["YearsAtCompany"] + 1.0)
    if "MonthlyIncome" in df_out.columns and "JobLevel" in df_out.columns:
        df_out["IncomePerJobLevel"] = df_out["MonthlyIncome"] / (df_out["JobLevel"] + 0.5)
    if "EnvironmentSatisfaction" in df_out.columns and "JobSatisfaction" in df_out.columns:
        df_out["SatisfactionInteraction"] = df_out["EnvironmentSatisfaction"] * df_out["JobSatisfaction"]
    return df_out


def add_approved_domain_interactions(df_in: pd.DataFrame) -> pd.DataFrame:
    """Constructs complex domain interactions and strain indicators."""
    df_out = df_in.copy()
    if "PromotionStagnationRatio" in df_out.columns and "WorkLifeDeficit" in df_out.columns:
        df_out["StagnationStrain"] = df_out["PromotionStagnationRatio"] * (df_out["WorkLifeDeficit"] + 1.0)
    if "ManagerTenureMismatch" in df_out.columns and "CompPenetrationIndex" in df_out.columns:
        df_out["CompTenureMismatch"] = df_out["ManagerTenureMismatch"] / (df_out["CompPenetrationIndex"].abs() + 0.1)
    if "SevereDissatisfactionFlag" in df_out.columns and "JobInvolvement" in df_out.columns:
        df_out["DisengagementIndex"] = (
            (df_out["SevereDissatisfactionFlag"] > 0) & (df_out["JobInvolvement"] <= 0)
        ).astype(float)
    if "SupervisoryChurnFlag" in df_out.columns and "EarlyCareerFlightRisk" in df_out.columns:
        df_out["SupervisoryFlightRisk"] = df_out["SupervisoryChurnFlag"] * df_out["EarlyCareerFlightRisk"]
    if "YearsAtCompany" in df_out.columns and "PromotionLagScore" in df_out.columns:
        df_out["TenurePromotionInteraction"] = df_out["YearsAtCompany"] * df_out["PromotionLagScore"]
    if "OverTime" in df_out.columns and "DistanceFromHome" in df_out.columns:
        df_out["OverTimeCommuteBurden"] = df_out["OverTime"] * df_out["DistanceFromHome"]
    if "CompositeSatisfaction" in df_out.columns and "WorkLifeDeficit" in df_out.columns:
        df_out["SatisfactionDeficitRatio"] = df_out["CompositeSatisfaction"] / (df_out["WorkLifeDeficit"].abs() + 0.1)
    if "ManagerBondRatio" in df_out.columns and "PromotionVelocity" in df_out.columns:
        df_out["ManagerBondVelocity"] = df_out["ManagerBondRatio"] * df_out["PromotionVelocity"]
    if "MonthlyIncome" in df_out.columns and "JobLevel" in df_out.columns:
        df_out["IncomeJobLevelDisparity"] = df_out["MonthlyIncome"] / (df_out["JobLevel"].abs() + 0.5)
    if "CareerPaceIndex" in df_out.columns and "JobHoppingRate" in df_out.columns:
        df_out["CareerPaceHoppingRatio"] = df_out["CareerPaceIndex"] * (df_out["JobHoppingRate"] + 0.1)
    if "HolisticBurnoutHazard" in df_out.columns and "OverTime" in df_out.columns:
        df_out["BurnoutOverTimeInteraction"] = df_out["HolisticBurnoutHazard"] * df_out["OverTime"]
    if "EarlyCareerFlightRisk" in df_out.columns and "PromotionStagnationRatio" in df_out.columns:
        df_out["EarlyCareerStagnation"] = df_out["EarlyCareerFlightRisk"] * df_out["PromotionStagnationRatio"]
    if "CompanyLoyaltyRatio" in df_out.columns and "SevereDissatisfactionFlag" in df_out.columns:
        df_out["LoyaltyDisengagement"] = df_out["CompanyLoyaltyRatio"] * df_out["SevereDissatisfactionFlag"]
    if "ManagerTenureMismatch" in df_out.columns and "HolisticBurnoutHazard" in df_out.columns:
        df_out["ManagerMismatchBurnout"] = df_out["ManagerTenureMismatch"] * df_out["HolisticBurnoutHazard"]
    if "JobSatisfaction" in df_out.columns and "WorkLifeBalance" in df_out.columns:
        df_out["JobSatisfactionWorkLife"] = df_out["JobSatisfaction"] * df_out["WorkLifeBalance"]
    if "EnvironmentSatisfaction" in df_out.columns and "JobSatisfaction" in df_out.columns:
        df_out["EnvironmentJobSatisfaction"] = df_out["EnvironmentSatisfaction"] * df_out["JobSatisfaction"]
    if "TotalWorkingYears" in df_out.columns and "MonthlyIncome" in df_out.columns:
        df_out["ExperienceIncomeVelocity"] = df_out["TotalWorkingYears"] / (df_out["MonthlyIncome"].abs() + 0.1)
    if "JobInvolvement" in df_out.columns and "StockOptionLevel" in df_out.columns:
        df_out["JobInvolvementStock"] = df_out["JobInvolvement"] * (df_out["StockOptionLevel"].abs() + 0.1)
    if "YearsAtCompany" in df_out.columns and "Age" in df_out.columns:
        df_out["AgeTenureRatio"] = df_out["YearsAtCompany"] / (df_out["Age"].abs() + 1.0)
    if "PromotionVelocity" in df_out.columns and "HolisticBurnoutHazard" in df_out.columns:
        df_out["PromotionPaceStrain"] = df_out["PromotionVelocity"] * df_out["HolisticBurnoutHazard"]
    return df_out


class FeaturePreprocessingPipeline:
    """
    Encapsulates fitted feature engineering and preprocessing transformers
    to convert raw employee survey data into the exact 30-feature matrix.
    """
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = MODELS_DIR
        self.models_dir = models_dir

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.features_30 = cfg["features_30"]

        fe_path = os.path.join(self.models_dir, "hr_feature_engineer.joblib")
        prep_path = os.path.join(self.models_dir, "preprocessor.joblib")

        self.fe = joblib.load(fe_path)
        self.prep = joblib.load(prep_path)

    def transform(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Transforms raw DataFrame into full feature matrix and extracts the exact 30 features.
        Returns:
            df_full: DataFrame with all intermediate transformed features
            X_30: np.ndarray shape (n_samples, 30) strictly adhering to schema order
        """
        df_fe = self.fe.transform(df_raw)
        df_struct = add_structural_features(df_fe)
        df_prep = self.prep.transform(df_struct)
        df_full = add_approved_domain_interactions(df_prep)

        # Validate 30 features present
        missing = [c for c in self.features_30 if c not in df_full.columns]
        if missing:
            raise KeyError(f"Missing required 30-feature columns: {missing}")

        X_30 = df_full[self.features_30].values
        return df_full, X_30
