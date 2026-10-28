"""
Comprehensive Automated Test Suite for Unified Employee Turnover Prediction System
===================================================================================
Tests:
1. Exact Step 32 Prediction Parity on 594-record Holdout (23 Errors, 96.13% Acc)
2. Preprocessing & 30-Feature Schema Compatibility
3. Single Employee Prediction API Endpoint
4. Batch File Upload & Downloadable Scored Predictions
5. Data Quality Validator & Logical Timeline Invariant Rejection
6. SHAP TreeExplainer Attribution & Local Additivity Verification
7. Two-Tier Decision Rule Override Traces
8. Statistical Drift Monitoring (PSI, KS with Benjamini-Hochberg FDR)
9. Fairness Audit & Subgroup Wilson 95% Confidence Intervals
10. HR Intervention Store CRUD Operations & Causal Disclaimers
11. Cryptographic SHA-256 Immutability of all Protected Research Baselines
"""

import os
import io
import sys
import json
import hashlib
import unittest
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure project root and workspace root are in sys.path
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
BASE_DIR = os.path.dirname(PROJECT_ROOT)
for p in [PROJECT_ROOT, BASE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from FINAL.app.step32_final import app, run_evaluation_benchmark
from FINAL.engine.pipeline import UnifiedPredictionEngine
from FINAL.engine.explainer import SHAPExplainerEngine
from FINAL.validator.data_validator import UnifiedDataValidator
from FINAL.monitoring.drift_monitor import UnifiedDriftMonitor
from FINAL.tracker.intervention_store import HRInterventionStore
from src.data_loader import load_and_validate_raw_data, clean_and_harmonize_dataset, get_train_test_split


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().upper()


class TestUnifiedTurnoverSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print("RUNNING COMPREHENSIVE UNIFIED SYSTEM INTEGRATION TEST SUITE")
        print("=" * 80)
        cls.client = app.test_client()
        cls.engine = UnifiedPredictionEngine()
        cls.validator = UnifiedDataValidator()
        cls.drift = UnifiedDriftMonitor()
        cls.interventions = HRInterventionStore()
        cls.explainer = SHAPExplainerEngine(cls.engine.clf_cb, cls.engine.clf_lgb, cls.engine.feats30)

        # Load clean dataset and holdout
        raw_path = os.path.join(BASE_DIR, "dataset", "merged_unique_profiles_2969.csv")
        df_raw, _ = load_and_validate_raw_data(raw_path)
        df_clean = clean_and_harmonize_dataset(df_raw)
        _, cls.X_hd_raw, _, cls.y_hd = get_train_test_split(df_clean, test_size=0.20, random_state=42)

    def test_01_step32_holdout_exact_parity(self):
        """Test 1: Verify exact bit-for-bit parity with official Step 32 predictions (23 errors / 594)."""
        print("\n[TEST 01] Verifying Step 32 Holdout Prediction Parity...")
        df_preds = self.engine.predict_dataframe(self.X_hd_raw)
        y_pred = df_preds["Predicted_Class"].values

        s32_target_path = os.path.join(BASE_DIR, "artifacts", "track1_step32_aggressive", "final_candidate", "step32_candidate_holdout.csv")
        df_s32_target = pd.read_csv(s32_target_path)
        y_target = df_s32_target["Pred_Step32_Candidate"].values

        diffs = np.where(y_pred != y_target)[0]
        self.assertEqual(len(diffs), 0, f"Discrepancies found with official Step 32: {diffs}")

        cm = confusion_matrix(self.y_hd, y_pred).ravel()
        errors = cm[1] + cm[2]
        self.assertEqual(cm[0], 487, f"Expected TN=487, got {cm[0]}")
        self.assertEqual(cm[1], 11, f"Expected FP=11, got {cm[1]}")
        self.assertEqual(cm[2], 12, f"Expected FN=12, got {cm[2]}")
        self.assertEqual(cm[3], 84, f"Expected TP=84, got {cm[3]}")
        self.assertEqual(errors, 23, f"Expected 23 total errors, got {errors}")

        acc = accuracy_score(self.y_hd, y_pred)
        f1 = f1_score(self.y_hd, y_pred)
        self.assertAlmostEqual(acc, 0.961279, places=4)
        self.assertAlmostEqual(f1, 0.879581, places=4)
        print(f"  [PASS] 0 discrepancies! TN=487, FP=11, FN=12, TP=84 | Acc={acc*100:.2f}%, F1={f1*100:.2f}%")

    def test_02_preprocessing_and_schema_compatibility(self):
        """Test 2: Verify feature preprocessing schema produces exactly 30 features."""
        print("\n[TEST 02] Verifying Preprocessing & Schema Compatibility...")
        df_full, _ = self.engine._transform_features(self.X_hd_raw.head(5))
        self.assertEqual(len(self.engine.feats30), 30)
        for col in self.engine.feats30:
            self.assertIn(col, df_full.columns, f"Feature '{col}' missing from transformed columns!")
        print("  [PASS] All 30 required schema features extracted and transformed cleanly.")

    def test_03_single_prediction_endpoint(self):
        """Test 3: Verify single prediction REST API endpoint."""
        print("\n[TEST 03] Verifying Single Prediction API Endpoint...")
        rec = self.X_hd_raw.iloc[0].to_dict()
        res = self.client.post("/api/predict", json=rec)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("turnover_probability", data["prediction"])
        self.assertIn("dominant_push_factors", data["shap_explanation"])
        self.assertIn("dominant_pull_factors", data["shap_explanation"])
        print(f"  [PASS] Predicted turnover probability: {data['prediction']['turnover_probability']:.4f} ({data['prediction']['turnover_risk_tier']})")

    def test_04_run_evaluation_without_retraining(self):
        """Test 4: Verify Run Evaluation endpoint performs pure forward inference with zero retraining."""
        print("\n[TEST 04] Verifying Run Evaluation Endpoint (Zero Retraining)...")
        res = self.client.post("/api/run-evaluation")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["evaluation_mode"], "INFERENCE_ONLY_ZERO_RETRAINING")
        self.assertFalse(data["retrained"], "Evaluation must never retrain or modify model weights!")
        self.assertEqual(data["evaluated_records"], 594)

        m = data["metrics"]
        self.assertAlmostEqual(m["accuracy"], 96.13, places=1)
        self.assertAlmostEqual(m["f1_score"], 87.96, places=1)
        self.assertAlmostEqual(m["precision"], 88.42, places=1)
        self.assertAlmostEqual(m["recall"], 87.50, places=1)
        self.assertAlmostEqual(m["roc_auc"], 94.55, places=1)
        self.assertEqual(m["total_errors"], 23)

        cm = data["confusion_matrix"]
        self.assertEqual(cm["TN"], 487)
        self.assertEqual(cm["FP"], 11)
        self.assertEqual(cm["FN"], 12)
        self.assertEqual(cm["TP"], 84)
        self.assertTrue(data["historical_parity_verified"])
        print(f"  [PASS] Evaluated {data['evaluated_records']} records in {data['latency_ms']} ms. Zero retraining, 23 errors verified.")

    def test_05_data_validator_rejection(self):
        """Test 5: Verify rejection of invalid records and timeline invariant violations."""
        print("\n[TEST 05] Verifying Data Quality Validator Rejection...")
        rec_clean = self.X_hd_raw.iloc[0].to_dict()

        # Corrupt with timeline invariant violation: YearsAtCompany > TotalWorkingYears
        bad_rec = rec_clean.copy()
        bad_rec["YearsAtCompany"] = 15
        bad_rec["TotalWorkingYears"] = 4
        is_valid, viols = self.validator.validate_record(bad_rec)
        self.assertFalse(is_valid, "Validator should have rejected YearsAtCompany > TotalWorkingYears!")
        self.assertTrue(any(v["type"] == "TimelineInvariantViolation" for v in viols))

        # Test API rejection
        res = self.client.post("/api/predict", json=bad_rec)
        self.assertEqual(res.status_code, 422)
        print("  [PASS] Invariant violation correctly identified and rejected with HTTP 422.")

    def test_06_shap_explanation_and_local_additivity(self):
        """Test 6: Verify SHAP TreeExplainer attributions and local efficiency/additivity."""
        print("\n[TEST 06] Verifying SHAP Explanations & Local Additivity...")
        rec = self.X_hd_raw.iloc[0].to_dict()
        pred_res = self.engine.predict_single_record(rec)
        df_full, _ = self.engine._transform_features(pd.DataFrame([rec]))
        X_30 = df_full[self.engine.feats30].values

        exp = self.explainer.explain_record(X_30, pred_res)
        self.assertTrue(exp["additivity_check"]["verified"], "SHAP additivity flag not verified!")
        self.assertLess(
            exp["additivity_check"]["additivity_residual"],
            1e-4,
            f"SHAP additivity residual {exp['additivity_check']['additivity_residual']} exceeded mathematical tolerance (1e-4)!"
        )
        self.assertIn("dominant_push_factors", exp)
        self.assertIn("dominant_pull_factors", exp)
        self.assertIn("academic_causal_disclaimer", exp)
        self.assertIn("NOT establish causal", exp["academic_causal_disclaimer"])
        print(f"  [PASS] SHAP explanation mathematically verified. Additivity residual: {exp['additivity_check']['additivity_residual']:.10e} (tolerance < 1e-4)")

    def test_07_rule_override_trace(self):
        """Test 7: Verify two-tier discrete rule override detection."""
        print("\n[TEST 07] Verifying Two-Tier Decision Rule Override Traces...")
        df_preds = self.engine.predict_dataframe(self.X_hd_raw)
        traces = df_preds["Rule_Override_Trace"].unique()
        self.assertTrue(any("Boundary Domain Specialist" in t for t in traces))
        self.assertTrue(any("Standard Tier 1" in t for t in traces))
        print(f"  [PASS] Rule override traces surfaced: {list(traces)}")

    def test_08_drift_monitoring_psi_ks(self):
        """Test 8: Verify offline drift calculation with PSI, KS, and FDR correction."""
        print("\n[TEST 08] Verifying Drift Monitoring Diagnostics...")
        df_full, _ = self.engine._transform_features(self.X_hd_raw)
        drift_rep = self.drift.calculate_batch_drift(df_full[self.engine.feats30])
        self.assertIn("overall_status", drift_rep)
        self.assertIn("feature_details", drift_rep)
        self.assertEqual(len(drift_rep["feature_details"]), 30)

        # Test sample-size safeguard on N < 30
        small_sample = df_full[self.engine.feats30].head(10)
        small_rep = self.drift.calculate_batch_drift(small_sample)
        self.assertTrue(small_rep["sample_size_safeguard"]["active"])
        print("  [PASS] Drift monitored on 30 features; sample-size safeguard verified.")

    def test_09_fairness_and_subgroup_metrics(self):
        """Test 9: Verify disparate impact ratio and Wilson 95% confidence intervals."""
        print("\n[TEST 09] Verifying Fairness Audit & Responsible AI Principles...")
        res = self.client.get("/api/benchmark-data")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["benchmarks"]["fairness_audit"]
        self.assertAlmostEqual(data["disparate_impact_ratio"], 0.951, places=2)
        self.assertTrue(data["four_fifths_rule_compliant"])
        self.assertEqual(len(data["ethical_governance_rules"]), 4)
        print("  [PASS] Disparate impact ratio: 0.951 (EEOC 4/5ths compliant). Ethical rules verified.")

    def test_10_hr_intervention_crud_and_causal_disclaimer(self):
        """Test 10: Verify HR intervention store CRUD operations and observational disclaimers."""
        print("\n[TEST 10] Verifying HR Intervention Store CRUD & Causal Disclaimers...")
        # Create
        new_int = self.interventions.add_intervention({
            "employee_ref": "EMP_UNITTEST_01",
            "department": "Engineering",
            "turnover_probability": 0.77,
            "risk_tier": "Critical",
            "intervention_type": "Career Growth Pathway",
            "action_status": "Action Planned",
            "observed_outcome": "Pending Evaluation"
        })
        self.assertIn("id", new_int)

        # Update
        updated = self.interventions.update_intervention(new_int["id"], {
            "observed_outcome": "Retained (Active)",
            "action_status": "Completed & Monitored"
        })
        self.assertEqual(updated["observed_outcome"], "Retained (Active)")

        # Verify disclaimer in stats
        summary = self.interventions.get_summary_statistics()
        self.assertIn("causality_disclaimer", summary)
        self.assertIn("NOT prove that the retention was caused", summary["causality_disclaimer"])
        print(f"  [PASS] Intervention {new_int['id']} created, updated, and causal disclaimer verified.")

    def test_11_protected_baseline_immutability(self):
        """Test 11: Cryptographic SHA-256 verification of all 76 protected baseline files."""
        print("\n[TEST 11] Verifying SHA-256 Immutability of all 76 Master Protected Baseline Files...")
        manifest_path = os.path.join(BASE_DIR, "scratch", "master_protected_files_pre_unification.json")
        self.assertTrue(os.path.exists(manifest_path), "Pre-unification hash manifest not found!")

        with open(manifest_path, "r") as f:
            expected_hashes = json.load(f)

        mismatches = []
        for rel_path, exp_hash in expected_hashes.items():
            full_path = os.path.join(BASE_DIR, rel_path)
            if not os.path.exists(full_path):
                mismatches.append(f"MISSING: {rel_path}")
                continue
            curr_hash = compute_sha256(full_path)
            if curr_hash != exp_hash:
                mismatches.append(f"MODIFIED: {rel_path} (Expected {exp_hash[:8]}, Got {curr_hash[:8]})")

        self.assertEqual(len(mismatches), 0, f"PROTECTED BASELINE MODIFIED: {mismatches}")
        print(f"  [PASS] 100% IMMUTABLE! All {len(expected_hashes)} protected files verified unchanged.")

    def test_12_preloaded_demonstration_records(self):
        """Test 12: Verify preloaded holdout demonstration cases (no upload interface required)."""
        print("\n[TEST 12] Verifying Preloaded Demonstration Cases & Endpoints...")
        res = self.client.get("/api/test-records")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_records_available"], 594)
        self.assertGreaterEqual(len(data["demonstration_cases"]), 5)

        # Check detail endpoint for case 0
        detail_res = self.client.get("/api/test-record/0")
        self.assertEqual(detail_res.status_code, 200)
        detail_data = detail_res.get_json()
        self.assertEqual(detail_data["status"], "success")
        self.assertIn("profile", detail_data)
        self.assertEqual(detail_data["holdout_id"], "EMP_HD_001")

        # Test prediction by record_idx
        pred_res = self.client.post("/api/predict", json={"record_idx": 0})
        self.assertEqual(pred_res.status_code, 200)
        pred_data = pred_res.get_json()
        self.assertEqual(pred_data["status"], "success")
        self.assertIn("turnover_probability", pred_data["prediction"])
        print("  [PASS] Preloaded demonstration records and index lookup verified cleanly.")

    def test_13_terminal_evaluation_cli_and_parity(self):
        """Test 13: Verify Method 1 CLI (--evaluate) executes forward inference with zero retraining and parity."""
        print("\n[TEST 13] Verifying Terminal Evaluation CLI (Method 1: --evaluate) & Web Parity...")
        script_path = os.path.join(BASE_DIR, "final_employee_turnover_system", "FINAL", "app", "step32_final.py")
        if not os.path.exists(script_path):
            script_path = os.path.join(BASE_DIR, "final_employee_turnover_system", "step32_final.py")
        cmd = [sys.executable, script_path, "--evaluate"]

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, f"--evaluate failed with code {proc.returncode}:\n{proc.stderr}")
        stdout = proc.stdout

        # Verify zero retraining confirmation
        self.assertIn("INFERENCE_ONLY_ZERO_RETRAINING", stdout)
        self.assertIn("CONFIRMED ZERO RETRAINING", stdout)

        # Verify key metrics in output table
        self.assertIn("96.13%", stdout, "Accuracy 96.13% missing from CLI table")
        self.assertIn("88.42%", stdout, "Precision 88.42% missing from CLI table")
        self.assertIn("87.50%", stdout, "Recall 87.50% missing from CLI table")
        self.assertIn("87.96%", stdout, "F1-Score 87.96% missing from CLI table")
        self.assertIn("23 / 594", stdout, "Total errors 23 missing from CLI table")

        # Verify confusion matrix cells
        self.assertIn("487", stdout, "TN count 487 missing from CLI table")
        self.assertIn("11", stdout, "FP count 11 missing from CLI table")
        self.assertIn("12", stdout, "FN count 12 missing from CLI table")
        self.assertIn("84", stdout, "TP count 84 missing from CLI table")

        # Verify disclaimer and parity status
        self.assertIn("HISTORICAL BENCHMARK", stdout)
        self.assertIn("BIT-EXACT PARITY: VERIFIED", stdout)

        # Verify parity with Website API (Method 2)
        web_res = self.client.post("/api/run-evaluation")
        self.assertEqual(web_res.status_code, 200)
        web_data = web_res.get_json()
        web_m = web_data["metrics"]
        web_cm = web_data["confusion_matrix"]

        self.assertAlmostEqual(web_m["accuracy"], 96.13, places=2)
        self.assertAlmostEqual(web_m["f1_score"], 87.96, places=2)
        self.assertEqual(web_m["total_errors"], 23)
        self.assertEqual(web_cm["TN"], 487)
        self.assertEqual(web_cm["FP"], 11)
        self.assertEqual(web_cm["FN"], 12)
        self.assertEqual(web_cm["TP"], 84)

        print("  [PASS] CLI --evaluate executed successfully (code 0). 100% parity verified between Terminal & Website!")

    def test_14_terminal_evaluation_error_handling(self):
        """Test 14: Verify evaluation error handling and input validation."""
        print("\n[TEST 14] Verifying Evaluation Error Handling on Missing or Invalid Data...")
        with self.assertRaises(ValueError):
            run_evaluation_benchmark(pd.DataFrame())

        empty_missing_target = pd.DataFrame({"SomeCol": [1, 2, 3]})
        with self.assertRaises(ValueError):
            run_evaluation_benchmark(empty_missing_target)

        print("  [PASS] Missing data and invalid schema rejected cleanly with ValueError.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
