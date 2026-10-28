"""
Unit Tests for Feature Schema Validation
========================================
Verifies:
- Exactly 30 features
- Feature order preservation
- Schema validation fails cleanly on missing fields
"""

import unittest
import json
import os
import pandas as pd
from src.data_loader import load_config
from src.preprocessing import FeaturePreprocessingPipeline


class TestSchemaValidation(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_feature_count_and_order(self):
        feats = self.cfg["features_30"]
        self.assertEqual(len(feats), 30, f"Expected exactly 30 features, got {len(feats)}")
        self.assertEqual(feats[0], "Age")
        self.assertEqual(feats[-1], "PromotionPaceStrain")

    def test_missing_field_detection(self):
        pipe = FeaturePreprocessingPipeline()
        # Create defective dummy dataframe missing key fields
        df_bad = pd.DataFrame([{"Age": 30, "Gender": "Male"}])
        with self.assertRaises(Exception):
            pipe.transform(df_bad)


if __name__ == "__main__":
    unittest.main()
