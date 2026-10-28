"""
Unified Multi-Stage Hybrid Pipeline Engine
==========================================
Coordinates the 5-stage inference sequence:
- Stage 1: CatBoost Anchor Model (Step 27 NSGA-II)
- Stage 2: CatBoost Base Learner
- Stage 3: LightGBM Base Learner
- Stage 4: Logistic Regression Meta-Learner (6 meta-features, operating threshold tau = 0.361)
- Stage 5: Conditional Error Correction (FN recovery, FP suppression veto)
- Stage 6: Domain Boundary Specialist Refinement
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")

from src.preprocessing import FeaturePreprocessingPipeline
from src.anchor_model import CatBoostAnchorModel
from src.base_models import BaseEnsembleClassifiers
from src.meta_model import LogisticRegressionMetaLearner
from src.conditional_rules import ConditionalErrorCorrector


class MultiStageHybridPipeline:
    """
    Main prediction engine executing the complete verified multi-stage workflow.
    """
    def __init__(self, models_dir: Optional[str] = None, threshold: Optional[float] = None):
        if models_dir is None:
            models_dir = MODELS_DIR
        self.models_dir = models_dir

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.features_30 = self.config["features_30"]
        self.tau = threshold if threshold is not None else self.config.get("operating_threshold", 0.361)

        self.prep = FeaturePreprocessingPipeline(models_dir=self.models_dir)
        self.anchor = CatBoostAnchorModel(models_dir=self.models_dir)
        self.base = BaseEnsembleClassifiers(models_dir=self.models_dir)
        self.meta = LogisticRegressionMetaLearner(models_dir=self.models_dir, threshold=self.tau)
        self.corrector = ConditionalErrorCorrector(models_dir=self.models_dir)

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Executes end-to-end inference over raw employee records.
        """
        n_samples = len(df_raw)

        # Preprocessing & 30 feature representation
        df_full, X_30 = self.prep.transform(df_raw)

        # Stage 1: Anchor model
        p_anchor = self.anchor.transform_and_predict(df_raw)

        # Stages 2 & 3: Base learners
        p_cb, p_lgb = self.base.predict_proba(X_30)

        # Stage 4: Meta-learner
        p_meta, pred_meta, X_meta = self.meta.predict(p_cb, p_lgb, p_anchor)

        # Stages 5 & 6: Conditional error correction & specialist refinement
        pred_final, audit_log = self.corrector.apply_corrections(
            pred_base=pred_meta,
            p_meta=p_meta,
            p_cb=p_cb,
            p_lgb=p_lgb,
            p_anchor=p_anchor,
            df_full=df_full,
            features_30=self.features_30,
            tau=self.tau
        )

        # Assign Risk Tiers:
        # Critical: >= 0.60, High: >= tau, Moderate: >= 0.20, Low: < 0.20
        risk_tiers = []
        for p in p_meta:
            if p >= 0.60:
                risk_tiers.append("Critical")
            elif p >= self.tau:
                risk_tiers.append("High")
            elif p >= 0.20:
                risk_tiers.append("Moderate")
            else:
                risk_tiers.append("Low")

        # Compile structured output dataframe
        df_out = pd.DataFrame({
            "Turnover_Score": np.round(p_meta, 4),
            "Binary_Decision": np.where(pred_final == 1, "Leaver", "Stayer"),
            "Predicted_Class": pred_final,
            "Risk_Tier": risk_tiers,
            "P_CatBoost": np.round(p_cb, 4),
            "P_LightGBM": np.round(p_lgb, 4),
            "P_Anchor": np.round(p_anchor, 4),
            "P_Specialist": np.round(audit_log["p_specialist"], 4),
            "Rule_Override_Trace": audit_log["rule_descriptions"],
            "Operating_Threshold": self.tau
        })

        if "Holdout_ID" in df_raw.columns:
            df_out.insert(0, "Employee_ID", df_raw["Holdout_ID"])
        elif "EmployeeID" in df_raw.columns:
            df_out.insert(0, "Employee_ID", df_raw["EmployeeID"])
        else:
            df_out.insert(0, "Employee_ID", [f"EMP_{i+1:04d}" for i in range(n_samples)])

        if "Attrition" in df_raw.columns:
            df_out["Actual_Label"] = df_raw["Attrition"]
            # Outcome category
            categories = []
            for y_true, y_pred in zip(df_raw["Attrition"], pred_final):
                if y_true == 1 and y_pred == 1:
                    categories.append("TP")
                elif y_true == 0 and y_pred == 0:
                    categories.append("TN")
                elif y_true == 0 and y_pred == 1:
                    categories.append("FP")
                else:
                    categories.append("FN")
            df_out["Outcome_Category"] = categories

        return df_out
