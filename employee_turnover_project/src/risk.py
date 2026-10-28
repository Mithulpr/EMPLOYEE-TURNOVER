"""
Operational Risk Classification Module
======================================
Implements operational 4-tier risk segmentation:
- Low Risk: [0.0, 0.20)
- Moderate Risk: [0.20, tau = 0.361)
- High Risk: [tau = 0.361, 0.60)
- Critical Risk: [0.60, 1.00]

Note: Risk tiers are distinct from the binary threshold (tau = 0.361).
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def assign_risk_tier(score: float, tau: float = 0.361) -> str:
    """Classifies a continuous model score into the operational 4-tier risk band."""
    if score >= 0.60:
        return "Critical"
    elif score >= tau:
        return "High"
    elif score >= 0.20:
        return "Moderate"
    else:
        return "Low"


def compute_risk_distribution(scores: np.ndarray, tau: float = 0.361) -> pd.DataFrame:
    """
    Computes aggregate counts and percentages across all 4 operational risk tiers.
    """
    tiers = [assign_risk_tier(s, tau) for s in scores]
    s_series = pd.Series(tiers)
    
    order = ["Low", "Moderate", "High", "Critical"]
    counts = s_series.value_counts().reindex(order, fill_value=0)
    total = len(scores)
    pcts = (counts / total * 100.0).round(2)
    
    df_dist = pd.DataFrame({
        "Risk_Tier": order,
        "Count": counts.values,
        "Percentage": pcts.values
    })
    return df_dist
