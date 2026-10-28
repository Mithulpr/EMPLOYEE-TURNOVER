#!/usr/bin/env python3
"""
Quick Terminal Results Viewer for Base Paper Reproduction (Talebi et al., 2025).
Displays the verified metrics table and confusion matrix directly in the terminal in 0.05s.
"""

import sys
from pathlib import Path

# Import the display function from run_base_paper
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from run_base_paper import print_terminal_results
import json

def main():
    metrics_path = BASE_DIR / "outputs" / "metrics.json"
    features_path = BASE_DIR / "outputs" / "selected_features.json"

    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found. Please run 'python run_base_paper.py' first.")
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    selected_features = None
    if features_path.exists():
        with open(features_path, "r", encoding="utf-8") as f:
            selected_features = json.load(f)

    print_terminal_results(metrics, selected_features)

if __name__ == "__main__":
    main()
