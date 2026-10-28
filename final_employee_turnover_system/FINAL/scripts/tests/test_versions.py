"""
Automated Test Suite for Five Official Versions
================================================
Verifies that all 5 official preserved versions reproduce their exact
retrospective classification metrics on the 594-record holdout.
"""

import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

for p in [str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class TestFiveVersions(unittest.TestCase):
    def test_01_final_version(self):
        """FINAL: 23 errors, 96.13% Acc, TN=487, FP=11, FN=12, TP=84."""
        p = PROJECT_ROOT / "FINAL" / "verification" / "retest_step32_594_results.csv"
        df = pd.read_csv(p)
        cm = confusion_matrix(df["Actual_Label"], df["Predicted_Class"]).ravel()
        self.assertEqual((cm[0], cm[1], cm[2], cm[3]), (487, 11, 12, 84))
        self.assertEqual(cm[1] + cm[2], 23)
        self.assertAlmostEqual(accuracy_score(df["Actual_Label"], df["Predicted_Class"]) * 100, 96.13, places=2)

    def test_02_champion_1_step31(self):
        """CHAMPION 1 (Step 31 v2): 24 errors, 95.96% Acc, TN=486, FP=12, FN=12, TP=84."""
        p = PROJECT_ROOT / "CHAMPIONS" / "CHAMPION_1_STEP31" / "predictions" / "step31_candidate_v2_holdout.csv"
        df = pd.read_csv(p)
        cm = confusion_matrix(df["True_Label"], df["Pred_Step31_v2"]).ravel()
        self.assertEqual((cm[0], cm[1], cm[2], cm[3]), (486, 12, 12, 84))
        self.assertEqual(cm[1] + cm[2], 24)
        self.assertAlmostEqual(accuracy_score(df["True_Label"], df["Pred_Step31_v2"]) * 100, 95.96, places=2)

    def test_03_champion_2_step30(self):
        """CHAMPION 2 (Step 30 Severe-FN): 25 errors, 95.79% Acc, TN=487, FP=11, FN=14, TP=82."""
        p = PROJECT_ROOT / "CHAMPIONS" / "CHAMPION_2_STEP30" / "predictions" / "step30_severe_fn_holdout.csv"
        df = pd.read_csv(p)
        cm = confusion_matrix(df["True_Label"], df["Pred_Step30_SevereFN"]).ravel()
        self.assertEqual((cm[0], cm[1], cm[2], cm[3]), (487, 11, 14, 82))
        self.assertEqual(cm[1] + cm[2], 25)
        self.assertAlmostEqual(accuracy_score(df["True_Label"], df["Pred_Step30_SevereFN"]) * 100, 95.79, places=2)

    def test_04_champion_3_step28(self):
        """CHAMPION 3 (Step 28 Dual Meta): 26 errors, 95.62% Acc, TN=487, FP=11, FN=15, TP=81."""
        p = PROJECT_ROOT / "CHAMPIONS" / "CHAMPION_3_STEP28" / "predictions" / "step28_holdout_predictions.csv"
        df = pd.read_csv(p)
        cm = confusion_matrix(df["True_Label"], df["Pred_Step28"]).ravel()
        self.assertEqual((cm[0], cm[1], cm[2], cm[3]), (487, 11, 15, 81))
        self.assertEqual(cm[1] + cm[2], 26)
        self.assertAlmostEqual(accuracy_score(df["True_Label"], df["Pred_Step28"]) * 100, 95.62, places=2)

    def test_05_champion_4_stage6(self):
        """CHAMPION 4 (Stage 6 / Review-2): 30 errors, 94.95% Acc, TN=487, FP=11, FN=19, TP=77."""
        p = PROJECT_ROOT / "CHAMPIONS" / "CHAMPION_4_STAGE6" / "predictions" / "stage6_review2_holdout_predictions.csv"
        df = pd.read_csv(p)
        cm = confusion_matrix(df["true_label"], df["pred_label"]).ravel()
        self.assertEqual((cm[0], cm[1], cm[2], cm[3]), (487, 11, 19, 77))
        self.assertEqual(cm[1] + cm[2], 30)
        self.assertAlmostEqual(accuracy_score(df["true_label"], df["pred_label"]) * 100, 94.95, places=2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
