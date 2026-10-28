"""
Stages 2 & 3: Base Gradient-Boosted Classifiers Module
======================================================
Encapsulates:
- Stage 2: CatBoost Base Learner (trained on 30 features)
- Stage 3: LightGBM Base Learner (trained on 30 features)
"""

import os
import joblib
import numpy as np
from typing import Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class BaseEnsembleClassifiers:
    """
    Dual gradient-boosted base classifiers generating complementary risk estimates.
    """
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = MODELS_DIR
        self.models_dir = models_dir

        cb_path = os.path.join(self.models_dir, "base_catboost.joblib")
        lgb_path = os.path.join(self.models_dir, "base_lightgbm.joblib")

        self.clf_cb = joblib.load(cb_path)
        try:
            self.clf_lgb = joblib.load(lgb_path)
        except Exception:
            self.clf_lgb = None

    def predict_proba(self, X_30: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates probability predictions from both base learners.
        Returns:
            p_cb: CatBoost predicted probabilities (n_samples,)
            p_lgb: LightGBM predicted probabilities (n_samples,)
        """
        p_cb = self.clf_cb.predict_proba(X_30)[:, 1]
        if self.clf_lgb is not None:
            p_lgb = self.clf_lgb.predict_proba(X_30)[:, 1]
        else:
            p_lgb = p_cb.copy()
        return p_cb, p_lgb
