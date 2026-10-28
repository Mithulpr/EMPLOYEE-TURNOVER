"""
Integration Tests for Prediction Pipeline
=========================================
Tests bulk evaluation on holdout cohort and verifies exact target metrics:
Accuracy: 96.13%, TN: 487, FP: 11, FN: 12, TP: 84 (23 errors).
"""

import unittest
from src.evaluate import evaluate_holdout


class TestPredictionPipeline(unittest.TestCase):
    def test_holdout_exact_parity(self):
        metrics, df_preds = evaluate_holdout(tau=0.361)
        cm = metrics["confusion_matrix"]

        self.assertEqual(metrics["total_records"], 594)
        self.assertEqual(metrics["total_errors"], 23)
        self.assertEqual(metrics["total_correct"], 571)

        self.assertEqual(cm["TN"], 487)
        self.assertEqual(cm["FP"], 11)
        self.assertEqual(cm["FN"], 12)
        self.assertEqual(cm["TP"], 84)

        self.assertAlmostEqual(metrics["accuracy_pct"], 96.13, places=1)
        self.assertAlmostEqual(metrics["precision_pct"], 88.42, places=1)
        self.assertAlmostEqual(metrics["recall_pct"], 87.50, places=1)
        self.assertAlmostEqual(metrics["f1_score_pct"], 87.96, places=1)


if __name__ == "__main__":
    unittest.main()
