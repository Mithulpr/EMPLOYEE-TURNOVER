# Explainable Employee Turnover Prediction — Workspace Repository

## Overview
This repository contains the complete research workspace for the **Two-Stage Selective Classification Employee Turnover Prediction System**.

The workspace is organized into modular directories supporting active model execution, raw data provenance, shared engineering libraries, academic publications, and historical research evidence.

---

## Directory Organization

```text
E:\R & D anitgravity/
│
├── final_employee_turnover_system/     [PRIMARY PRODUCTION SYSTEM — STRICTLY PROTECTED]
│   ├── FINAL/                          # Official 96.13% Production Implementation (Step 32)
│   ├── CHAMPIONS/                      # Predecessor versions (CHAMPIONS 1–4)
│   ├── data/                           # Retrospective holdout cohort (N=594) & SHAP cache
│   ├── research/                       # Methodology, ablation studies, audits & 10 figures
│   ├── scripts/                        # Standalone verification & comparison utilities
│   ├── tests/                          # 28-test automated integration suite
│   ├── archive/                        # Legacy model weights & unselected notes
│   ├── run.py                          # Unified launcher (--web, --cli, --verify, --compare)
│   └── requirements.txt                # Locked dependencies
│
├── dataset/                            [Master Research Dataset]
│   └── merged_unique_profiles_2969.csv # Standardized 2,969-profile HR employee dataset
│
├── src/                                [Shared Core Feature & Data Engineering Module]
│   ├── data_loader.py                  # Dataset loading, cleaning, and partitioning
│   ├── feature_engineering.py          # Domain feature transformation classes
│   ├── models/                         # Base model wrappers
│   └── evaluation/                     # Metric calculation utilities
│
├── artifacts/                          [Protected Baseline Reference Manifests]
│   ├── track1_step32_aggressive/       # Step 32 baseline reference predictions & configs
│   ├── track1_step36_aggressive_xgboost/# Step 36 ablation baselines
│   └── track1_step37_explainable_extension/# Step 37 validation reports & SHAP outputs
│
├── research_documents/                 [Academic Manuscripts, Reports & Literature]
│   ├── papers/                         # Final IEEE manuscript drafts (.docx)
│   ├── reports/                        # 24 formal R&D progress reports (v2–v12)
│   ├── references/                     # Published base paper PDF (Talebi et al., 2025)
│   └── other_research/                 # Independent research papers
│
├── scripts/                            [Core Benchmark & Research Utilities]
│   ├── run_track1_step37_*.py          # Explainability & scientific hardening runners
│   ├── generate_corrected_roc_chart.py # 300 DPI publication ROC curve generator
│   ├── run_experiment_b_base_paper.py  # Talebi et al. (2025) baseline benchmark
│   ├── run_experiment_c_adaptive_lgbm.py# Adaptive LightGBM comparative study
│   ├── run_benchmark_reproduction_audit.py # Multi-model reproduction audit
│   └── evaluate_step32.py              # Standalone Step 32 evaluator
│
├── scratch/                            [Baseline Manifest Registry]
│   └── master_protected_files_pre_unification.json # Cryptographic baseline manifest
│
├── archive/                            [Historical Provenance Registry]
│   └── legacy_experiments/             # Summary of Steps 1–32 R&D trajectory
│
├── requirements.txt                    # Workspace-level Python requirements
└── README.md                           # This navigation guide
```

---

## Quickstart & Primary Application

To interact with the official production model, launch web dashboards, or reproduce holdout evaluations, navigate into `final_employee_turnover_system/`:

```powershell
cd "E:\R & D anitgravity\final_employee_turnover_system"

# 1. Verify FINAL model on retrospective holdout (96.13% Acc, 23 Errors, 94.55% ROC-AUC)
python run.py --verify

# 2. Compare all 5 official versions (FINAL + CHAMPIONS 1-4)
python run.py --compare

# 3. Verify cryptographic SHA-256 immutability of protected artifacts
python run.py --integrity

# 4. Launch interactive Flask dashboard
python run.py --web --port 5000

# 5. Run full test suite (28/28 tests)
python -m unittest discover -s tests -p "test_*.py"
```

---

## External Backup
A full, byte-verified pre-cleanup backup of all historical experiments, raw outputs, and intermediate models was generated before workspace consolidation:
- **Location**: `E:\R & D anitgravity_backup_20261001_213158`
- **Total Archives**: 2,576 files (1,070.80 MB)
- **Integrity**: SHA-256 manifest recorded in `backup_manifest.json`
