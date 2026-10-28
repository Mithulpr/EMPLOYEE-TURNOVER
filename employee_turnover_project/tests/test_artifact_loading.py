"""
Unit Tests for Artifact Loading
===============================
Verifies that all 11 model and data artifacts load without errors.
"""

import unittest
from src.validation import validate_system_integrity


class TestArtifactLoading(unittest.TestCase):
    def test_all_artifacts_load_successfully(self):
        status, issues = validate_system_integrity()
        self.assertTrue(status, f"Validation failed with issues: {issues}")


if __name__ == "__main__":
    unittest.main()
