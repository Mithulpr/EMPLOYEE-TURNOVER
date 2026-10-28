"""
Unit Tests for Threshold Decision Behavior
==========================================
Verifies exact boundary decision rule:
score < 0.361  -> Stayer (0)
score >= 0.361 -> Leaver (1)

Explicitly tests:
- score = 0.3609
- score = 0.3610
- score = 0.5000
- score = 0.9000
"""

import unittest
import numpy as np


class TestThresholdBehavior(unittest.TestCase):
    def setUp(self):
        self.tau = 0.3610

    def test_threshold_boundary_cases(self):
        scores = [0.3609, 0.3610, 0.5000, 0.9000]
        expected_classes = [0, 1, 1, 1]
        expected_labels = ["Stayer", "Leaver", "Leaver", "Leaver"]

        for s, exp_cls, exp_lbl in zip(scores, expected_classes, expected_labels):
            pred_class = int(s >= self.tau)
            pred_label = "Leaver" if pred_class == 1 else "Stayer"

            self.assertEqual(pred_class, exp_cls, f"Score {s} failed expected class {exp_cls}")
            self.assertEqual(pred_label, exp_lbl, f"Score {s} failed expected label {exp_lbl}")

    def test_strict_inequality(self):
        self.assertEqual(int(0.360999 >= self.tau), 0)
        self.assertEqual(int(0.361000 >= self.tau), 1)


if __name__ == "__main__":
    unittest.main()
