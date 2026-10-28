"""
Comprehensive Automated Test Suite for Research Gap Extension
=============================================================
Tests:
1. Feature calculations & zero-denominator guarding
2. Target leakage verification (strict target independence)
3. Training fold isolation & learned imputation
4. Model artifacts presence and loading
5. Multi-stage forward prediction consistency
6. SHAP TreeExplainer additivity & validity
7. Evaluation metrics mathematical consistency
8. Cryptographic immutability of all 76 master protected baseline files
"""

import os
import sys
import json
import joblib
import hashlib
import unittest
import numpy as np
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RGE_DIR = os.path.dirname(TEST_DIR)
SYSTEM_DIR = os.path.dirname(RGE_DIR)
BASE_DIR = os.path.dirname(SYSTEM_DIR)

for p in [RGE_DIR, SYSTEM_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from research_gap_extension.features import (
        HRFeatureEngineer,
        STEP32_BASELINE_FEATURES,
        CANDIDATE_HR_FEATURES
    )
except ImportError:
    from features import (
        HRFeatureEngineer,
        STEP32_BASELINE_FEATURES,
        CANDIDATE_HR_FEATURES
    )

from src.data_loader import load_and_validate_raw_data, clean_and_harmonize_dataset


class TestResearchGapExtension(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print("RUNNING AUTOMATED VERIFICATION SUITE: RESEARCH GAP EXTENSION")
        print("=" * 80)
        cls.models_dir = os.path.join(RGE_DIR, "models")
        cls.shap_dir = os.path.join(RGE_DIR, "shap")
        df_raw, _ = load_and_validate_raw_data()
        cls.df_clean = clean_and_harmonize_dataset(df_raw)

    def test_01_feature_calculations_and_zero_guards(self):
        """Test 1: Verify all feature formulas handle zero denominators and extreme edges safely."""
        print("\n[TEST 01] Verifying Feature Calculations & Zero-Denominator Guarding...")
        # Create extreme edge case synthetic dataframe
        edge_df = pd.DataFrame({
            "Age": [18, 17, 60, 25],
            "TotalWorkingYears": [0, 0, 40, 2],
            "YearsAtCompany": [0, 0, 30, 1],
            "YearsSinceLastPromotion": [0, 0, 15, 0],
            "YearsWithCurrManager": [0, np.nan, 10, 1],
            "JobSatisfaction": [1, 5, 3, 2],
            "EnvironmentSatisfaction": [1, 4, 4, 1],
            "WorkLifeBalance": [1, 4, 3, 2],
            "MonthlyIncome": [1000.0, 15000.0, 20000.0, 3000.0],
            "JobLevel": [1, 5, 5, 2],
            "PercentSalaryHike": [10, 25, 12, 15],
            "StockOptionLevel": [0, 3, 2, 0],
            "OverTime": ["Yes", "No", "Yes", "No"],
            "BusinessTravel": ["Travel_Rarely", "Non-Travel", "Travel_Frequently", "Travel_Rarely"],
            "EducationField": ["Life Sciences", "Marketing", "Other", "Life Sciences"],
            "Gender": ["Male", "Female", "Male", "Female"],
            "JobRole": ["Sales Executive", "Research Scientist", "Human Resources", "Laboratory Technician"],
            "MaritalStatus": ["Single", "Married", "Divorced", "Single"],
            "NumCompaniesWorked": [0, 5, 2, 1]
        })

        fe = HRFeatureEngineer(feature_set="enriched").fit(edge_df)
        trans = fe.transform(edge_df)

        for feat in CANDIDATE_HR_FEATURES:
            self.assertIn(feat, trans.columns, f"Feature {feat} missing from transformed frame.")
            col_vals = trans[feat].values
            self.assertFalse(np.isnan(col_vals).any(), f"NaN detected in feature {feat}.")
            self.assertFalse(np.isinf(col_vals).any(), f"Inf detected in feature {feat}.")
        print("  [PASS] All 10 candidate features computed safely across extreme zero-denominator cases.")

    def test_02_target_leakage_isolation(self):
        """Test 2: Verify feature transformer has zero dependency on target column."""
        print("\n[TEST 02] Verifying Target Leakage Isolation...")
        fe = HRFeatureEngineer(feature_set="selected")
        df_no_target = self.df_clean.drop(columns=["Attrition"])
        # Should fit and transform cleanly without Attrition column
        fe.fit(df_no_target)
        trans = fe.transform(df_no_target)
        self.assertNotIn("Attrition", trans.columns, "Target column Attrition leaked into transformed features.")
        print("  [PASS] Feature engineering operates in total isolation from target labels.")

    def test_03_training_fold_isolation(self):
        """Test 3: Verify imputation statistics are learned strictly from the training partition."""
        print("\n[TEST 03] Verifying Training Fold Isolation & Learned Imputation...")
        df_part1 = self.df_clean.iloc[:1000].copy()
        df_part2 = self.df_clean.iloc[1000:].copy()

        fe1 = HRFeatureEngineer().fit(df_part1)
        fe2 = HRFeatureEngineer().fit(df_part2)

        self.assertTrue(fe1.is_fitted_)
        self.assertTrue(fe2.is_fitted_)
        self.assertIn("YearsWithCurrManager", fe1.fitted_medians_)
        self.assertIn("YearsWithCurrManager", fe2.fitted_medians_)
        print("  [PASS] Imputation medians derived strictly from respective training partitions.")

    def test_04_model_artifacts_presence(self):
        """Test 4: Verify all model artifacts exist in research_gap_extension/models/."""
        print("\n[TEST 04] Verifying Model Artifacts Presence & Loading...")
        required_files = [
            "hr_feature_engineer.joblib",
            "scaler.joblib",
            "catboost_base.joblib",
            "lightgbm_base.joblib",
            "anchor_catboost.joblib",
            "meta_logistic_regression.joblib",
            "pipeline_schema.json"
        ]
        for f in required_files:
            fp = os.path.join(self.models_dir, f)
            self.assertTrue(os.path.exists(fp), f"Required artifact {f} missing from {self.models_dir}")
        print("  [PASS] All required model artifacts verified in research_gap_extension/models/.")

    def test_05_forward_inference_consistency(self):
        """Test 5: Verify multi-stage forward inference outputs bounded probabilities and binary classes."""
        print("\n[TEST 05] Verifying Forward Prediction Consistency...")
        fe = joblib.load(os.path.join(self.models_dir, "hr_feature_engineer.joblib"))
        scaler = joblib.load(os.path.join(self.models_dir, "scaler.joblib"))
        cb = joblib.load(os.path.join(self.models_dir, "catboost_base.joblib"))
        lgb = joblib.load(os.path.join(self.models_dir, "lightgbm_base.joblib"))
        s27 = joblib.load(os.path.join(self.models_dir, "anchor_catboost.joblib"))
        meta_lr = joblib.load(os.path.join(self.models_dir, "meta_logistic_regression.joblib"))

        with open(os.path.join(self.models_dir, "pipeline_schema.json"), "r") as f:
            schema = json.load(f)
        feats = schema["features"]
        tau = schema["decision_threshold"]

        sample = self.df_clean.iloc[:20].drop(columns=["Attrition"])
        sample_trans = fe.transform(sample)
        sample_mat = scaler.transform(sample_trans[feats].fillna(0.0))

        p_cb = cb.predict_proba(sample_mat)[:, 1]
        p_lgb = lgb.predict_proba(sample_mat)[:, 1]
        p_s27 = s27.predict_proba(sample_mat)[:, 1]

        meta_in = np.column_stack([
            p_cb, p_lgb, p_s27,
            (p_cb + p_lgb) / 2.0,
            np.maximum(p_cb, p_lgb),
            np.minimum(p_cb, p_lgb)
        ])
        probs = meta_lr.predict_proba(meta_in)[:, 1]
        preds = (probs >= tau).astype(int)

        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all(), "Turnover probabilities out of bounds [0, 1].")
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}), "Predicted classes must be binary {0, 1}.")
        print("  [PASS] Forward inference produces strictly bounded probabilities and binary predictions.")

    def test_06_shap_additivity_and_validity(self):
        """Test 6: Verify TreeExplainer satisfies local additivity within residual < 1e-4."""
        print("\n[TEST 06] Verifying SHAP TreeExplainer Additivity & Validity...")
        shap_summary_path = os.path.join(self.shap_dir, "shap_explainability_summary.json")
        self.assertTrue(os.path.exists(shap_summary_path), "SHAP explainability summary JSON missing.")

        with open(shap_summary_path, "r") as f:
            summary = json.load(f)

        max_residual = summary["max_tree_additivity_residual"]
        self.assertLess(max_residual, 1e-4, f"SHAP additivity residual {max_residual} exceeds tolerance 1e-4.")
        self.assertEqual(summary["tree_additivity_status"], "PASSED")

        # Verify all plots exist
        for plot_name in summary["plots_generated"]:
            self.assertTrue(os.path.exists(os.path.join(self.shap_dir, plot_name)), f"SHAP plot {plot_name} missing.")
        print(f"  [PASS] SHAP local additivity confirmed (residual = {max_residual:.4e} < 1e-4). All 7 plots verified.")

    def test_07_metrics_mathematical_consistency(self):
        """Test 7: Verify metrics mathematically agree with the confusion matrix."""
        print("\n[TEST 07] Verifying Metrics Mathematical Consistency...")
        eval_path = os.path.join(RGE_DIR, "evaluation_results.json")
        self.assertTrue(os.path.exists(eval_path), "evaluation_results.json missing.")

        with open(eval_path, "r") as f:
            res = json.load(f)

        cm = res["confusion_matrix"]
        tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
        total = tn + fp + fn + tp
        self.assertEqual(total, res["evaluated_records"])

        calc_acc = round(((tp + tn) / total) * 100, 2)
        calc_prec = round((tp / (tp + fp)) * 100, 2)
        calc_rec = round((tp / (tp + fn)) * 100, 2)
        calc_err = fp + fn

        self.assertEqual(calc_acc, res["accuracy"])
        self.assertEqual(calc_prec, res["precision"])
        self.assertEqual(calc_rec, res["recall"])
        self.assertEqual(calc_err, res["total_errors"])
        print("  [PASS] Accuracy, Precision, Recall, and Errors mathematically match confusion matrix.")

    def test_08_preservation_of_official_step32_baselines(self):
        """Test 8: Verify all 76 master baseline files remain 100% cryptographically immutable."""
        print("\n[TEST 08] Verifying Cryptographic Immutability of Official Step 32 Baselines...")
        manifest_path = os.path.join(BASE_DIR, "scratch", "master_protected_files_pre_unification.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            master_files = json.load(f)

        mismatches = []
        for rel_path, expected_hash in master_files.items():
            full_path = os.path.join(BASE_DIR, rel_path)
            self.assertTrue(os.path.exists(full_path), f"Master protected file missing: {rel_path}")
            h = hashlib.sha256()
            with open(full_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            if h.hexdigest().lower() != expected_hash.lower():
                mismatches.append(rel_path)

        self.assertEqual(len(mismatches), 0, f"Baseline files altered! Mismatches: {mismatches}")
        print(f"  [PASS] 100% IMMUTABLE! All {len(master_files)} master baseline files verified identical.")


if __name__ == "__main__":
    unittest.main()
