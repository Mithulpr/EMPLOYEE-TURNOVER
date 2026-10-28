"""
Unit Tests for Operational Risk Tiers
=====================================
Verifies exact operational risk tier boundaries:
- Low: [0.0, 0.20)
- Moderate: [0.20, 0.361)
- High: [0.361, 0.60)
- Critical: [0.60, 1.00]
"""

import unittest
from src.risk import assign_risk_tier, compute_risk_distribution
import numpy as np


class TestRiskTiers(unittest.TestCase):
    def setUp(self):
        self.tau = 0.361

    def test_tier_assignments(self):
        cases = [
            (0.05, "Low"),
            (0.1999, "Low"),
            (0.2000, "Moderate"),
            (0.3500, "Moderate"),
            (0.3609, "Moderate"),
            (0.3610, "High"),
            (0.5500, "High"),
            (0.5999, "High"),
            (0.6000, "Critical"),
            (0.8532, "Critical"),
            (1.0000, "Critical")
        ]
        for score, exp_tier in cases:
            tier = assign_risk_tier(score, self.tau)
            self.assertEqual(tier, exp_tier, f"Score {score} produced {tier}, expected {exp_tier}")

    def test_holdout_expected_distribution(self):
        from src.data_loader import load_cached_evaluation
        cache = load_cached_evaluation()
        scores = np.array([r["risk_score"] for r in cache["records"]])
        df_risk = compute_risk_distribution(scores, self.tau)

        counts = dict(zip(df_risk["Risk_Tier"], df_risk["Count"]))
        self.assertEqual(counts["Low"], 491)
        self.assertEqual(counts["Moderate"], 11)
        self.assertEqual(counts["High"], 14)
        self.assertEqual(counts["Critical"], 78)


if __name__ == "__main__":
    unittest.main()
