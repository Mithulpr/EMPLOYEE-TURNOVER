# Precursor Version Collection: CHAMPIONS 1 through 4

This directory preserves the **four official predecessor champion versions** that established the milestone performance progression leading up to the **FINAL** model.

---

## 1. Retrospective Error-Reduction Progression

All evaluations were conducted on the same **retrospective 594-record holdout cohort** under single-blind conditions:

| Official Name | Original Version | Architecture Description | Retrospective Accuracy | Total Errors | Confusion Matrix (TN, FP, FN, TP) | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **FINAL** | Step 32 | Stacking Meta-Learner + Boundary Specialist | **96.13%** | **23** | 487, 11, 12, 84 | **87.96%** | **94.55%** |
| **CHAMPION 1** | Step 31 v2 | Step 28 Backbone + Precision FN Boundary Recovery | **95.96%** | **24** | 486, 12, 12, 84 | **87.50%** | **94.55%** |
| **CHAMPION 2** | Step 30 | Dual Meta-Learner + Severe-FN Recovery Rule | **95.79%** | **25** | 487, 11, 14, 82 | **86.77%** | **94.55%** |
| **CHAMPION 3** | Step 28 | Logistic Meta-Learner (C=1.0) over CatBoost + LightGBM + Step 27 | **95.62%** | **26** | 487, 11, 15, 81 | **86.17%** | **94.55%** |
| **CHAMPION 4** | Stage 6 / Review-2 | Tuned Single-Model CatBoost (Depth 6 / Balanced, tau=0.4850) | **94.95%** | **30** | 487, 11, 19, 77 | **83.70%** | **92.32%** |

*Note: The 594-record cohort represents a retrospective development holdout. Figures describe performance on this locked cohort rather than an independent prospectively enrolled cohort.*

---

## 2. Directory Contents

- `CHAMPION_1_STEP31/`: Step 31 Candidate v2 configuration, holdout predictions, and standalone evaluation verifier.
- `CHAMPION_2_STEP30/`: Step 30 Severe-FN Recovery configuration, holdout predictions, and standalone evaluation verifier.
- `CHAMPION_3_STEP28/`: Step 28 Dual Meta-Learner configuration, holdout predictions, and standalone evaluation verifier.
- `CHAMPION_4_STAGE6/`: Stage 6 / Review-2 Primary configuration, holdout predictions, and standalone evaluation verifier.

---

## 3. How to Verify All Champions

To verify the metrics of all versions simultaneously:
```powershell
python scripts/compare_versions.py
# or
python run.py --compare
```
To verify individual champions:
```powershell
python CHAMPIONS/CHAMPION_1_STEP31/evaluation/evaluate_champion1.py
python CHAMPIONS/CHAMPION_2_STEP30/evaluation/evaluate_champion2.py
python CHAMPIONS/CHAMPION_3_STEP28/evaluation/evaluate_champion3.py
python CHAMPIONS/CHAMPION_4_STAGE6/evaluation/evaluate_champion4.py
```
