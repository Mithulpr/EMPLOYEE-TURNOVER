"""
Optuna Bayesian Hyperparameter Optimization Module (Version 2).
Optimizes hyperparameters for top-performing non-LightGBM candidate architectures:
XGBoost, CatBoost, ExtraTrees, HistGradientBoosting, and EBM.
Uses 5-Fold Stratified Cross-Validation on the training partition only.
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.base import clone
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from interpret.glassbox import ExplainableBoostingClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

class BayesianOptimizer:
    def __init__(self, n_trials: int = 50, cv: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.studies: Dict[str, optuna.Study] = {}

    def _evaluate_cv_score(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        acc_scores = []
        f1_scores = []
        auc_scores = []
        pr_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            clf = clone(model)
            clf.fit(X_tr, y_tr)

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_val)[:, 1]
                preds = (probs >= 0.5).astype(int)
                try:
                    auc = roc_auc_score(y_val, probs)
                    pr = average_precision_score(y_val, probs)
                except Exception:
                    auc = 0.5
                    pr = 0.0
            else:
                preds = clf.predict(X_val)
                auc = 0.5
                pr = 0.0

            acc_scores.append(accuracy_score(y_val, preds))
            f1_scores.append(f1_score(y_val, preds, zero_division=0))
            auc_scores.append(auc)
            pr_scores.append(pr)

        mean_acc = np.mean(acc_scores)
        mean_f1 = np.mean(f1_scores)
        mean_pr = np.mean(pr_scores)

        # Version 2 composite objective: Acc + PR-AUC + F1
        composite_score = (0.40 * mean_acc) + (0.35 * mean_pr) + (0.25 * mean_f1)
        return float(composite_score)

    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], xgb.XGBClassifier]:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 450, step=25),
                'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.12, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
                'subsample': trial.suggest_float('subsample', 0.65, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.5),
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'verbosity': 0
            }
            model = xgb.XGBClassifier(**params)
            return self._evaluate_cv_score(model, X, y)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        self.studies['XGBoost'] = study

        best_params = study.best_params
        best_params['verbosity'] = 0
        best_params['eval_metric'] = 'logloss'
        best_params['random_state'] = self.random_state
        best_model = xgb.XGBClassifier(**best_params)
        return best_params, best_model

    def optimize_catboost(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], CatBoostClassifier]:
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 200, 500, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.12, log=True),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 15.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-3, 5.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced', 'SqrtBalanced', None]),
                'random_seed': self.random_state,
                'verbose': 0
            }
            model = CatBoostClassifier(**params)
            return self._evaluate_cv_score(model, X, y)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=min(self.n_trials, 40))
        self.studies['CatBoost'] = study

        best_params = study.best_params
        best_params['verbose'] = 0
        best_params['random_seed'] = self.random_state
        best_model = CatBoostClassifier(**best_params)
        return best_params, best_model

    def optimize_extratrees(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], ExtraTreesClassifier]:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 450, step=50),
                'max_depth': trial.suggest_int('max_depth', 8, 22),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            model = ExtraTreesClassifier(**params)
            return self._evaluate_cv_score(model, X, y)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=min(self.n_trials, 35))
        self.studies['ExtraTrees'] = study

        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = -1
        best_model = ExtraTreesClassifier(**best_params)
        return best_params, best_model

    def optimize_hist_gradient_boosting(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], HistGradientBoostingClassifier]:
        def objective(trial):
            params = {
                'max_iter': trial.suggest_int('max_iter', 150, 450, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.12, log=True),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'random_state': self.random_state
            }
            model = HistGradientBoostingClassifier(**params)
            return self._evaluate_cv_score(model, X, y)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=min(self.n_trials, 35))
        self.studies['HistGradientBoosting'] = study

        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_model = HistGradientBoostingClassifier(**best_params)
        return best_params, best_model
