"""
Stage 4: Logistic Regression Meta-Learner Module
================================================
Constructs the exact 6-dimensional meta-feature representation and applies
the Level-2 Logistic Regression stacking classifier with decision threshold tau = 0.361.
"""

import os
import json
import joblib
import numpy as np
from typing import Optional, Tuple, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")


class LogisticRegressionMetaLearner:
    """
    Level-2 Logistic Regression meta-learner integrating base model probabilities
    and uncertainty metrics without arbitrary voting or averaging.
    """
    def __init__(self, models_dir: Optional[str] = None, threshold: Optional[float] = None):
        if models_dir is None:
            models_dir = MODELS_DIR
        self.models_dir = models_dir

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.default_tau = cfg.get("operating_threshold", 0.361)
        self.tau = threshold if threshold is not None else self.default_tau

        meta_path = os.path.join(self.models_dir, "meta_logistic_regression.joblib")
        self.meta_clf = joblib.load(meta_path)

    def build_meta_features(self, p_cb: np.ndarray, p_lgb: np.ndarray, p_anchor: np.ndarray) -> np.ndarray:
        """
        Builds the exact 6-element meta-feature vector:
        1. p_cb: CatBoost probability
        2. p_lgb: LightGBM probability
        3. p_anchor: Step 27 Anchor probability
        4. |p_cb - 0.5|: CatBoost decision margin
        5. |p_lgb - 0.5|: LightGBM decision margin
        6. |p_cb - p_lgb|: Inter-model discrepancy
        """
        X_meta = np.column_stack([
            p_cb,
            p_lgb,
            p_anchor,
            np.abs(p_cb - 0.5),
            np.abs(p_lgb - 0.5),
            np.abs(p_cb - p_lgb)
        ])
        return X_meta

    def predict(self, p_cb: np.ndarray, p_lgb: np.ndarray, p_anchor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates meta-features, predicts turnover score, and applies decision threshold tau.
        Returns:
            p_meta: model-estimated turnover score (n_samples,)
            pred_meta: binary decision (0 = Stayer, 1 = Leaver)
            X_meta: 6-feature meta representation matrix
        """
        X_meta = self.build_meta_features(p_cb, p_lgb, p_anchor)
        p_meta = self.meta_clf.predict_proba(X_meta)[:, 1]
        pred_meta = (p_meta >= self.tau).astype(int)
        return p_meta, pred_meta, X_meta
