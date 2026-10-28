# V12 Final Model Search Report

Generated: 2026-08-25 11:08:24 | Runtime: 4.7 min

## Official V2 Baseline
TP=71, FP=20, TN=478, FN=25, Errors=45 | Acc=92.42%, Prec=78.02%, Rec=73.96%, F1=75.94%, AUC=90.69%

## Model Competition (5-Fold OOF, N_train=2375)
| Model | AUC | F1 | Errors |
| :--- | :---: | :---: | :---: |
| V2 CatBoost | 89.42% | 68.37% | 247 |
| LGBM gbdt (V2-feat) | 88.32% | 66.48% | 243 |
| LGBM dart (V2-feat) | 86.28% | 57.94% | 408 |
| LGBM gbdt (Ext-feat) | 86.65% | 60.82% | 268 |
| CatBoost (Ext-feat) | 86.78% | 61.80% | 272 |
| XGBoost | 88.23% | 66.14% | 259 |
| HGB | 87.06% | 63.26% | 295 |

## Final Candidate: Ensemble(V2_CatBoost+LGBM_gbdt+XGBoost)

## Nested CV (25 Folds)

- Accuracy: 89.71%±1.42% CI=[89.15%,90.26%]
- Precision: 66.96%±3.86% CI=[65.44%,68.47%]
- Recall: 71.63%±5.97% CI=[69.28%,73.97%]
- F1_Score: 69.17%±4.62% CI=[67.36%,70.98%]
- ROC_AUC: 91.44%±2.27% CI=[90.55%,92.33%]
- PR_AUC: 74.36%±5.65% CI=[72.15%,76.58%]
- Balanced_Accuracy: 82.41%±3.19% CI=[81.16%,83.66%]
- Specificity: 93.19%±0.82% CI=[92.87%,93.51%]
- Brier_Score: 0.0733±0.0086

## Holdout (N=594) — FINAL_DEVELOPMENT_HOLDOUT_COMPARISON
TP=74, FP=29, TN=469, FN=22, Errors=51
Acc=91.41%, Prec=71.84%, Rec=77.08%, F1=74.37%, AUC=91.57%

## Error Ladder
40-ERROR MILESTONE NOT ACHIEVED
35-ERROR MILESTONE NOT ACHIEVED
30-ERROR MILESTONE NOT ACHIEVED
25-ERROR MILESTONE NOT ACHIEVED

## Base Paper Comparison
- Accuracy: 91.41% vs >92% → FAIL
- Precision: 71.84% vs >89% → FAIL
- Recall: 77.08% vs >85% → FAIL
- F1_Score: 74.37% vs >87% → FAIL
- ROC_AUC: 91.57% vs >94% → FAIL

## Status: CHAMPION NOT BEATEN
FULL BENCHMARK NOT YET EXCEEDED

RAW_DATA_INTEGRITY = PASS