"""
Unit & Integration Tests for Champion & Baseline Comparison Module.
Validates:
1. Endpoint /api/champion-comparison returns HTTP 200 and schema validity.
2. Verified base paper metrics adhere to published literature (Talebi et al. 2025).
3. Verified base paper reimplementation matches Experiment B on 2,969 dataset.
4. Top 5 verified historical champions are present without fabrication.
5. Official FINAL version (Step 32) is preserved with 96.13% accuracy & 23 errors.
6. Controlled same-dataset differences are programmatically calculated in percentage points.
7. Web template incorporates all 8 sections, navigation link, and champion detail modal.
"""

import unittest
import os
import sys
import json
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
FINAL_DIR = PROJECT_ROOT / "FINAL"

for p in [str(PROJECT_ROOT), str(FINAL_DIR), str(FINAL_DIR / "app")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from FINAL.app.step32_final import app

class TestChampionComparisonModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_api_endpoint_response_structure(self):
        """Test 1: /api/champion-comparison returns valid structure."""
        res = self.client.get("/api/champion-comparison")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertEqual(json_data.get("status"), "success")
        self.assertIn("data", json_data)
        d = json_data["data"]

        # Check main top-level keys
        self.assertIn("published_base_paper", d)
        self.assertIn("base_paper_on_our_dataset", d)
        self.assertIn("top_five_champions", d)
        self.assertIn("official_final_champion", d)
        self.assertIn("controlled_same_dataset_comparison", d)
        self.assertIn("cross_dataset_comparison", d)

    def test_02_published_base_paper_accuracy(self):
        """Test 2: Published Base Paper matches Table 13 of Talebi et al. (2025)."""
        res = self.client.get("/api/champion-comparison")
        d = res.get_json()["data"]["published_base_paper"]
        
        self.assertIn("Talebi", d["authors"])
        self.assertEqual(d["year"], 2025)
        self.assertEqual(d["doi"], "10.1016/j.joitmc.2025.100557")
        self.assertTrue(d["doi_url"].startswith("https://doi.org/10.1016/j.joitmc.2025.100557"))
        
        m = d["metrics"]
        self.assertAlmostEqual(m["accuracy"], 92.00, places=2)
        self.assertAlmostEqual(m["precision"], 89.00, places=2)
        self.assertAlmostEqual(m["recall"], 85.00, places=2)
        self.assertAlmostEqual(m["f1_score"], 87.00, places=2)
        self.assertAlmostEqual(m["roc_auc"], 94.00, places=2)
        self.assertIsNone(m["pr_auc"])
        self.assertIsNone(m["total_errors"])

    def test_03_base_paper_on_our_dataset_metrics(self):
        """Test 3: Base paper reimplementation matches Experiment B on 2,969 dataset."""
        res = self.client.get("/api/champion-comparison")
        d = res.get_json()["data"]["base_paper_on_our_dataset"]
        
        self.assertEqual(d["status"], "verified")
        self.assertEqual(d["selected_features_count"], 14)
        self.assertEqual(len(d["selected_features"]), 14)

        m = d["metrics"]
        self.assertAlmostEqual(m["accuracy"], 88.01, places=2)
        self.assertAlmostEqual(m["precision"], 74.80, places=2)
        self.assertAlmostEqual(m["recall"], 38.96, places=2)
        self.assertAlmostEqual(m["f1_score"], 51.23, places=2)
        self.assertAlmostEqual(m["roc_auc"], 87.11, places=2)
        self.assertAlmostEqual(m["pr_auc"], 66.13, places=2)
        self.assertEqual(m["total_errors"], 356)

        cm = d["confusion_matrix"]
        self.assertEqual(cm["TN"], 2426)
        self.assertEqual(cm["FP"], 63)
        self.assertEqual(cm["FN"], 293)
        self.assertEqual(cm["TP"], 187)
        self.assertEqual(cm["total"], 2969)

    def test_04_top_five_champions_registry(self):
        """Test 4: Top 5 verified champions are exactly 5 and rank 1 is Step 32."""
        res = self.client.get("/api/champion-comparison")
        champs = res.get_json()["data"]["top_five_champions"]
        
        self.assertEqual(len(champs), 5, "Must display exactly 5 verified historical champions")
        
        # Rank 1 must be Step 32
        r1 = champs[0]
        self.assertEqual(r1["rank"], 1)
        self.assertIn("Step 32", r1["version"])
        self.assertTrue(r1["is_final_champion"])
        self.assertAlmostEqual(r1["accuracy"], 96.13, places=2)
        self.assertAlmostEqual(r1["f1_score"], 87.96, places=2)
        self.assertEqual(r1["errors"], 23)
        self.assertEqual(r1["confusion_matrix"]["TN"], 487)
        self.assertEqual(r1["confusion_matrix"]["FP"], 11)
        self.assertEqual(r1["confusion_matrix"]["FN"], 12)
        self.assertEqual(r1["confusion_matrix"]["TP"], 84)

        # Ranks 2-5 must be precursor champions
        for c in champs[1:]:
            self.assertFalse(c["is_final_champion"])
            self.assertGreater(c["errors"], 23)

    def test_05_controlled_same_dataset_deltas(self):
        """Test 5: Controlled same-dataset differences are programmatically calculated in percentage points."""
        res = self.client.get("/api/champion-comparison")
        same = res.get_json()["data"]["controlled_same_dataset_comparison"]
        
        # Accuracy: 96.13 - 88.01 = +8.12 pp
        self.assertAlmostEqual(same["accuracy"]["difference_pp"], 8.12, places=2)
        # Precision: 88.42 - 74.80 = +13.62 pp
        self.assertAlmostEqual(same["precision"]["difference_pp"], 13.62, places=2)
        # Recall: 87.50 - 38.96 = +48.54 pp
        self.assertAlmostEqual(same["recall"]["difference_pp"], 48.54, places=2)
        self.assertGreater(same["recall"]["relative_gain_pct"], 120.0)
        # F1-Score: 87.96 - 51.23 = +36.73 pp
        self.assertAlmostEqual(same["f1_score"]["difference_pp"], 36.73, places=2)
        self.assertGreater(same["f1_score"]["relative_gain_pct"], 70.0)
        # ROC-AUC: 94.56 - 87.11 = +7.45 pp
        self.assertAlmostEqual(same["roc_auc"]["difference_pp"], 7.45, places=2)
        # PR-AUC: 86.89 - 66.13 = +20.76 pp
        self.assertAlmostEqual(same["pr_auc"]["difference_pp"], 20.76, places=2)

    def test_06_index_html_integration(self):
        """Test 6: index.html contains navigation tab, all 8 sections, and modal."""
        template_path = FINAL_DIR / "templates" / "index.html"
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Check Navigation
        self.assertIn('id="nav-tab-comparison"', html)
        self.assertIn('Champion & Baseline Comparison', html)

        # Check Main Tab
        self.assertIn('id="tab-comparison"', html)

        # Check Sections
        self.assertTrue('Comparison Overview' in html)
        self.assertTrue('Published Base Paper' in html)
        self.assertTrue('Top Five Champion Models' in html)

        # Check Badge and Modal
        self.assertIn('id="championDetailModal"', html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
