"""
Inference & Prediction Interface Module
======================================
Provides programmatic entrypoints for single employee and bulk batch predictions.
"""

import os
import pandas as pd
from typing import Optional, Dict, Any

from src.pipeline import MultiStageHybridPipeline


def predict_single_employee(employee_dict: Dict[str, Any], tau: float = 0.361) -> Dict[str, Any]:
    """Generates turnover prediction for a single employee record."""
    df_single = pd.DataFrame([employee_dict])
    pipeline = MultiStageHybridPipeline(threshold=tau)
    df_pred = pipeline.predict(df_single)
    return df_pred.iloc[0].to_dict()


def predict_bulk_file(input_csv: str, output_csv: Optional[str] = None, tau: float = 0.361) -> pd.DataFrame:
    """Generates turnover predictions for a CSV batch file."""
    df_raw = pd.read_csv(input_csv)
    pipeline = MultiStageHybridPipeline(threshold=tau)
    df_pred = pipeline.predict(df_raw)
    
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df_pred.to_csv(output_csv, index=False)
        print(f"[OK] Saved batch predictions to: {output_csv}")
    
    return df_pred
