"""
Candidate Machine Learning Architectures Suite (Version 2).
Defines algorithms across 6 diverse model families:
1. Gradient Boosted Decision Trees (XGBoost, CatBoost, HistGradientBoosting)
2. Ensemble Decision Trees (RandomForest, ExtraTrees, BalancedRandomForest)
3. Glassbox Additive Models (InterpretML Explainable Boosting Machine)
4. Regularized Linear Models (L1/L2/ElasticNet Logistic Regression)
5. Kernel Methods (Support Vector Classifier - RBF)
6. Neural Architectures (Multi-Layer Perceptron - MLP)
7. Advanced Ensembles (Meta Stacking & Calibrated Soft Voting)

NOTE: LightGBM is strictly EXCLUDED as it represents the base-paper classifier.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier, 
    GradientBoostingClassifier, 
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

def get_candidate_models(random_state: int = 42, scale_pos_weight: float = 5.19) -> Dict[str, BaseEstimator]:
    """
    Instantiates candidate estimators with well-configured baseline parameters.
    LightGBM is strictly excluded.
    """
    models = {
        # 1. Gradient Boosted Trees (Excluding LightGBM)
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.035,
            max_depth=6,
            min_child_weight=2,
            subsample=0.80,
            colsample_bytree=0.80,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0
        ),
        "CatBoost": CatBoostClassifier(
            iterations=350,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.035,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=random_state
        ),
        
        # 2. Ensemble Decision Trees
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "BalancedRandomForest": BalancedRandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1
        ),

        # 3. Glassbox Additive Model
        "ExplainableBoostingMachine": ExplainableBoostingClassifier(
            max_bins=256,
            outer_bags=12,
            inner_bags=6,
            learning_rate=0.035,
            random_state=random_state
        ),

        # 4. Regularized Linear Models
        "LogisticRegression_ElasticNet": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.5,
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state
        ),
        "LogisticRegression_L2": LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ),

        # 5. Kernel Methods
        "SupportVectorClassifier": SVC(
            C=1.5,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),

        # 6. Deep Neural Architecture
        "MultiLayerPerceptron": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.002,
            max_iter=450,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=random_state
        )
    }
    
    return models

def build_stacking_ensemble(
    estimators: List[Tuple[str, BaseEstimator]], 
    meta_learner: BaseEstimator = None,
    cv: int = 5,
    random_state: int = 42
) -> StackingClassifier:
    """
    Constructs a cross-validated Stacking Classifier with calibrated probability features.
    """
    if meta_learner is None:
        meta_learner = LogisticRegression(C=0.5, penalty='l2', max_iter=1000, random_state=random_state)
        
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1
    )

def build_voting_ensemble(
    estimators: List[Tuple[str, BaseEstimator]], 
    weights: List[float] = None
) -> VotingClassifier:
    """
    Constructs a soft-voting probability ensemble.
    """
    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
        n_jobs=-1
    )
