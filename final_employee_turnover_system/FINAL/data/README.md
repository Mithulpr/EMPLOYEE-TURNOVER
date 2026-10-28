# Data Directory

This directory contains the authoritative datasets and precomputed caches used by the **Explainable Employee Turnover Prediction System**.

---

## 1. Retrospective Holdout Cohort (`holdout_594.csv`)

> **File:** `holdout_594.csv`  
> **Size:** 594 employee records (32 columns: `Holdout_ID`, 30 domain features, `Attrition`)  
> **Class Breakdown:** 498 stayers (`Attrition = 0`), 96 leavers (`Attrition = 1`)  
> **Target Imbalance:** 16.16% positive rate  
> **Evaluation Protocol Note:** This dataset represents a **retrospective demonstration holdout cohort** that was frozen and reused during developmental benchmarking. It must be interpreted as retrospective performance rather than an untouched, independent prospective clinical validation cohort.

---

## 2. Precomputed Evaluation & Explanation Caches

- `evaluation_cache_594.json` (3.2 MB): Authoritative precomputed row-level predictions, risk tiers, and structured explanations across all 594 holdout cases for instant web dashboard loading.
- `full_shap_cache_594.json` (2.8 MB): Precomputed local SHAP attributions across all 30 features for all 594 holdout employees.
- `reference_pool_summary.json` (162.9 KB): Quantile distributions and statistical moments computed from the 1,900-sample training reference distribution for offline KS and PSI drift monitoring.
- `hr_interventions.json` (41.8 KB): Operational intervention audit store with explicit observational causality guardrails.
