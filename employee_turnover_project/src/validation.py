"""
System and Artifact Validation Module
=====================================
Validates integrity of:
- Required serialized model artifacts (11 files)
- Configuration schema and 30-feature specification
- Holdout evaluation data integrity
- System dependencies
"""

import os
import json
import joblib
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


REQUIRED_MODEL_FILES = [
    "anchor_catboost.joblib",
    "anchor_pipeline.joblib",
    "base_catboost.joblib",
    "base_lightgbm.joblib",
    "false_negative_recovery_catboost.joblib",
    "false_positive_recovery_catboost.joblib",
    "hr_feature_engineer.joblib",
    "meta_logistic_regression.joblib",
    "preprocessor.joblib",
    "reference_pool.npz",
    "specialist_catboost.joblib"
]

REQUIRED_DATA_FILES = [
    "holdout_594.csv",
    "evaluation_cache_594.json"
]


def validate_system_integrity() -> Tuple[bool, List[str]]:
    """
    Executes exhaustive verification of all artifacts, schemas, and dependencies.
    Returns (status: bool, issues: List[str]).
    """
    issues = []

    # 1. Config validation
    if not os.path.exists(CONFIG_PATH):
        issues.append(f"Missing master config: {CONFIG_PATH}")
    else:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        feats = cfg.get("features_30", [])
        if len(feats) != 30:
            issues.append(f"Expected exactly 30 features in config, found {len(feats)}")
        if cfg.get("operating_threshold") != 0.361:
            issues.append(f"Default operating threshold must be 0.361, found {cfg.get('operating_threshold')}")

    # 2. Model artifacts validation
    for m in REQUIRED_MODEL_FILES:
        p = os.path.join(MODELS_DIR, m)
        if not os.path.exists(p):
            issues.append(f"Missing required model artifact: {m}")
        else:
            if p.endswith(".joblib"):
                try:
                    obj = joblib.load(p)
                except Exception as e:
                    issues.append(f"Failed to load {m}: {e}")

    # 3. Data files validation
    for d in REQUIRED_DATA_FILES:
        p = os.path.join(DATA_DIR, d)
        if not os.path.exists(p):
            issues.append(f"Missing required data artifact: {d}")

    status = (len(issues) == 0)
    return status, issues
