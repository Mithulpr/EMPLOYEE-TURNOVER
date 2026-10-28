"""
Data Loading, Inspection, Validation, and Hygiene Module for Base Paper Reproduction.
Loads and preprocesses the 2,969-profile dataset using strictly self-contained, relative paths.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "merged_unique_profiles_2969.csv"

def load_and_validate_raw_data(filepath: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw dataset, performs integrity validation, and returns (df, summary diagnostics).
    Defaults to self-contained BASE_PAPER_REPRODUCE/data/merged_unique_profiles_2969.csv.
    """
    if filepath is None:
        target_path = DEFAULT_DATA_PATH
    else:
        target_path = Path(filepath)

    if not target_path.exists():
        raise FileNotFoundError(f"Dataset not found at {target_path}")

    df = pd.read_csv(target_path)

    diagnostics = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "target_distribution": df["Attrition"].value_counts(dropna=False).to_dict() if "Attrition" in df.columns else {},
        "target_proportions": df["Attrition"].value_counts(normalize=True).to_dict() if "Attrition" in df.columns else {},
        "source_breakdown": df["_dataset_source"].value_counts().to_dict() if "_dataset_source" in df.columns else {},
        "missing_summary": df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    }

    return df, diagnostics

def clean_and_harmonize_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Applies standard data hygiene as established for the 2,969 dataset:
    1. Harmonizes MonthlyIncome (Dataset_A had a 10x denomination multiplier relative to Dataset_B).
    2. Corrects categorical typos (e.g. 'TravelRarely' -> 'Travel_Rarely').
    3. Strips whitespace from all string columns.
    4. Drops non-predictive identifiers ('EmpID', 'EmployeeID', 'EmployeeNumber').
    5. Drops invariant/constant columns ('EmployeeCount', 'Over18', 'StandardHours').
    6. Drops redundant text labels that duplicate full-sample numeric columns.
    7. Drops partition marker '_dataset_source' to avoid source-specific leakage.
    8. Maps Attrition to binary 0/1.
    """
    df = df_raw.copy()

    # 1. Clean string values
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan})

    # 2. Harmonize MonthlyIncome 10x scale in Dataset_A
    if "_dataset_source" in df.columns and "MonthlyIncome" in df.columns:
        mask_a = df["_dataset_source"] == "Dataset_A"
        df.loc[mask_a, "MonthlyIncome"] = df.loc[mask_a, "MonthlyIncome"] / 10.0

    # 3. Correct categorical typos in BusinessTravel
    if "BusinessTravel" in df.columns:
        df["BusinessTravel"] = df["BusinessTravel"].replace({"TravelRarely": "Travel_Rarely"})

    # 4. Map Target
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"No": 0, "Yes": 1}).astype(int)

    # 5. Drop Identifiers
    id_cols = ["EmpID", "EmployeeID", "EmployeeNumber"]

    # 6. Drop Constant columns
    const_cols = ["EmployeeCount", "Over18", "StandardHours"]

    # 7. Drop Redundant Label columns that mirror universal numeric columns
    redundant_label_cols = [
        "AgeGroup",
        "Education_Label",
        "EnvironmentSatisfaction_Label",
        "JobSatisfaction_Label",
        "PerformanceRating_Label",
        "WorkLifeBalance_Label",
        "SalarySlab"
    ]

    # 8. Drop source marker
    source_cols = ["_dataset_source"]

    cols_to_drop = [c for c in id_cols + const_cols + redundant_label_cols + source_cols if c in df.columns]
    df_cleaned = df.drop(columns=cols_to_drop)

    return df_cleaned

def get_train_test_split(
    df_cleaned: pd.DataFrame, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified 80/20 train/test split.
    """
    X = df_cleaned.drop(columns=["Attrition"])
    y = df_cleaned["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
