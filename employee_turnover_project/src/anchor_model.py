"""
Stage 1: CatBoost Anchor Model Module
=====================================
Encapsulates the Step 27 NSGA-II CatBoost Anchor model and pipeline.
Provides baseline retention anchor probability signals for meta-learning.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class CatBoostAnchorModel:
    """
    Stage 1 Anchor Model trained under NSGA-II multi-objective optimization.
    Applies separate feature pipeline and produces calibrated anchor probabilities.
    """
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = MODELS_DIR
        self.models_dir = models_dir

        model_path = os.path.join(self.models_dir, "anchor_catboost.joblib")
        pipe_path = os.path.join(self.models_dir, "anchor_pipeline.joblib")

        self.model = joblib.load(model_path)
        self.pipeline = joblib.load(pipe_path)
        self.features = self.pipeline["features"]

    def transform_and_predict(self, df_raw: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw data through anchor pipeline and generates P_Anchor probabilities.
        """
        from src.preprocessing import add_structural_features, add_approved_domain_interactions

        fe27 = self.pipeline["fe"]
        prep27 = self.pipeline["prep"]

        df_fe = fe27.transform(df_raw)
        df_struct = add_structural_features(df_fe)
        df_prep = prep27.transform(df_struct)
        df_full = add_approved_domain_interactions(df_prep)

        X_anchor = df_full[self.features]
        p_anchor = self.model.predict_proba(X_anchor)[:, 1]
        return p_anchor
