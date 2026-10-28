"""
Data Loader & Dataset Specification Module
=========================================
Handles loading, schema validation, and cohort partitioning for:
- Full 2,969-record dataset (merged profiles)
- 594-record retrospective legacy holdout cohort
- Individual employee inference records
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "model_config.json")


def load_config() -> Dict[str, Any]:
    """Loads master configuration parameters."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_holdout_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the 594-record retrospective legacy holdout dataset.
    Note: Labelled strictly as a retrospective legacy holdout, not an untouched test set.
    """
    if file_path is None:
        file_path = os.path.join(DATA_DIR, "holdout_594.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Holdout dataset not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df


def load_full_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """Loads the full 2,969-record dataset."""
    if file_path is None:
        file_path = os.path.join(DATA_DIR, "merged_unique_profiles_2969.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Full dataset not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df


def load_cached_evaluation(cache_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads precomputed verification evaluation cache."""
    if cache_path is None:
        cache_path = os.path.join(DATA_DIR, "evaluation_cache_594.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Evaluation cache not found at: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_full_shap_cache(cache_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads precomputed complete SHAP explanations cache."""
    if cache_path is None:
        cache_path = os.path.join(DATA_DIR, "full_shap_cache_594.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"SHAP cache not found at: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_summary() -> Dict[str, Any]:
    """Returns official dataset demographic and partitioning statistics."""
    cfg = load_config()
    return cfg.get("dataset_metadata", {})
