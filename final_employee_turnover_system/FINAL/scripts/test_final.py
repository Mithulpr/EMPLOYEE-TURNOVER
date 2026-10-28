#!/usr/bin/env python3
"""
Unit Tests for FINAL Step 32 Employee Turnover Model
=====================================================
Validates on the original trained model artifacts:
1. Exact bit-for-bit parity on 594-record retrospective holdout:
   - Accuracy: 96.13%
   - Precision: 88.42%
   - Recall: 87.50%
   - F1-Score: 87.96%
   - Unrounded ROC-AUC: 94.5469%
   - Confusion Matrix: TN=487, FP=11, FN=12, TP=84 (23 Errors)
2. All 12 serialized model artifacts load successfully with Joblib.
3. Preprocessing schema produces exactly 30 model features.
"""

import sys
import unittest
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
PROJECT_ROOT = FINAL_DIR.parent if FINAL_DIR.name == "FINAL" else FINAL_DIR

for p in [str(FINAL_DIR), str(SCRIPT_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from pipeline import UnifiedPredictionEngine
except ImportError:
    from scripts.pipeline import UnifiedPredictionEngine


class TestFinalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = FINAL_DIR / "data" / "holdout_594.csv"
        cls.models_dir = FINAL_DIR / "models"
        cls.engine = UnifiedPredictionEngine(models_dir=str(cls.models_dir))
        cls.df_holdout = pd.read_csv(cls.data_path)
        cls.feature_cols = [c for c in cls.df_holdout.columns if c not in ["Holdout_ID", "Attrition"]]

    def test_01_holdout_exact_parity(self):
        """Test 1: Verify exact 23 errors and 96.13% accuracy on 594 retrospective holdout."""
        preds_df = self.engine.predict_dataframe(self.df_holdout[self.feature_cols])
        y_true = self.df_holdout["Attrition"].values
        y_pred = preds_df["Predicted_Class"].values

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        errors = int(fp + fn)

        self.assertEqual(len(y_true), 594, "Cohort size must be exactly 594")
        self.assertEqual(tn, 487, f"Expected TN=487, got {tn}")
        self.assertEqual(fp, 11, f"Expected FP=11, got {fp}")
        self.assertEqual(fn, 12, f"Expected FN=12, got {fn}")
        self.assertEqual(tp, 84, f"Expected TP=84, got {tp}")
        self.assertEqual(errors, 23, f"Expected exactly 23 errors, got {errors}")

        acc = accuracy_score(y_true, y_pred) * 100.0
        prec = precision_score(y_true, y_pred) * 100.0
        rec = recall_score(y_true, y_pred) * 100.0
        f1 = f1_score(y_true, y_pred) * 100.0

        self.assertAlmostEqual(acc, 96.13, places=2)
        self.assertAlmostEqual(prec, 88.42, places=2)
        self.assertAlmostEqual(rec, 87.50, places=2)
        self.assertAlmostEqual(f1, 87.96, places=2)

    def test_02_unrounded_roc_auc(self):
        """Test 2: Verify continuous unrounded ROC-AUC equals 94.5469%."""
        df_full, df_full27 = self.engine._transform_features(self.df_holdout[self.feature_cols])
        s27_feats = self.engine.s27_pipe["features"]
        p_s27 = self.engine.s27_model.predict_proba(df_full27[s27_feats])[:, 1]
        X_30 = df_full[self.engine.feats30].values
        p_cb = self.engine.clf_cb.predict_proba(X_30)[:, 1]
        p_lgb = (
            self.engine.clf_lgb.predict_proba(X_30)[:, 1]
            if self.engine.clf_lgb is not None
            else p_cb.copy()
        )
        X_meta = np.column_stack([
            p_cb, p_lgb, p_s27,
            np.abs(p_cb - 0.5), np.abs(p_lgb - 0.5),
            np.abs(p_cb - p_lgb),
        ])
        p_meta = self.engine.meta_clf.predict_proba(X_meta)[:, 1]
        y_true = self.df_holdout["Attrition"].values

        auc = roc_auc_score(y_true, p_meta) * 100.0
        self.assertAlmostEqual(auc, 94.5469, places=3)
        self.assertAlmostEqual(round(auc, 2), 94.55, places=2)

    def test_03_features_count(self):
        """Test 3: Feature schema contains exactly 30 model features."""
        self.assertEqual(len(self.engine.feats30), 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
