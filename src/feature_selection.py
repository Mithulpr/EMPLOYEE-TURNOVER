"""
Feature Selection and Preprocessing Module (Version 2).
Encodes categoricals, standardizes numericals, eliminates redundant collinearity,
and performs multi-method consensus feature selection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

class HRPreprocessor(BaseEstimator, TransformerMixin):
    """
    Handles categorical encoding, missing value imputation, and numerical scaling.
    Learns all statistics strictly on the training set.
    """
    def __init__(self, scale_numerics: bool = True):
        self.scale_numerics = scale_numerics
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self.encoded_feature_names: List[str] = []
        self.ohe: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.num_impute_values: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_df = X.copy()
        self.cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = [c for c in X_df.columns if c not in self.cat_cols]

        # Learn median imputation for numericals
        for c in self.num_cols:
            self.num_impute_values[c] = float(X_df[c].median()) if not X_df[c].dropna().empty else 0.0

        # OneHotEncoder for categoricals
        if self.cat_cols:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            cat_df = X_df[self.cat_cols].fillna('Missing').astype(str)
            self.ohe.fit(cat_df)
            ohe_names = self.ohe.get_feature_names_out(self.cat_cols).tolist()
        else:
            ohe_names = []

        # Scaler for numerics
        if self.scale_numerics and self.num_cols:
            self.scaler = StandardScaler()
            num_df = X_df[self.num_cols].fillna(self.num_impute_values)
            self.scaler.fit(num_df)

        self.encoded_feature_names = self.num_cols + ohe_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        
        # Numerics
        if self.num_cols:
            num_data = X_df[self.num_cols].fillna(self.num_impute_values).values
            if self.scaler is not None:
                num_data = self.scaler.transform(num_data)
        else:
            num_data = np.empty((len(X_df), 0))

        # Categoricals
        if self.cat_cols and self.ohe is not None:
            cat_df = X_df[self.cat_cols].fillna('Missing').astype(str)
            cat_data = self.ohe.transform(cat_df)
            ohe_names = self.ohe.get_feature_names_out(self.cat_cols).tolist()
        else:
            cat_data = np.empty((len(X_df), 0))
            ohe_names = []

        combined = np.hstack([num_data, cat_data])
        cols = self.num_cols + ohe_names
        return pd.DataFrame(combined, columns=cols, index=X_df.index)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.encoded_feature_names


class MultiMethodFeatureSelector:
    """
    Executes collinearity pruning and multi-criteria feature selection:
    1. Collinearity filter (prunes features with Pearson |r| > correlation_threshold)
    2. Mutual Information ranking
    3. L1-Penalized LASSO sparsity
    4. Tree importance (ExtraTrees + RandomForest)
    5. Consensus selection across criteria
    """
    def __init__(
        self, 
        top_k: int = 40, 
        consensus_threshold: int = 2, 
        correlation_threshold: float = 0.92,
        random_state: int = 42
    ):
        self.top_k = top_k
        self.consensus_threshold = consensus_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.selected_features_dict: Dict[str, List[str]] = {}
        self.feature_scores_df: Optional[pd.DataFrame] = None
        self.consensus_features: List[str] = []
        self.uncorrelated_features: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_df = X.copy()
        
        # 1. Collinearity Filtering: identify and prune highly correlated redundant pairs
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        self.uncorrelated_features = [c for c in X_df.columns if c not in to_drop]
        
        X_filtered = X_df[self.uncorrelated_features]
        feature_names = self.uncorrelated_features
        actual_k = min(self.top_k, len(feature_names))

        # 2. Mutual Information
        mi_scores = mutual_info_classif(X_filtered, y, random_state=self.random_state)
        mi_ranking = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
        mi_selected = mi_ranking.head(actual_k).index.tolist()
        self.selected_features_dict['Mutual_Information'] = mi_selected

        # 3. L1-Regularized LASSO Logistic Regression
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.25, random_state=self.random_state, max_iter=1000)
        lasso.fit(X_filtered, y)
        lasso_coef = np.abs(lasso.coef_[0])
        lasso_ranking = pd.Series(lasso_coef, index=feature_names).sort_values(ascending=False)
        lasso_selected = lasso_ranking[lasso_ranking > 1e-4].head(actual_k).index.tolist()
        if len(lasso_selected) < 8:
            lasso_selected = lasso_ranking.head(actual_k).index.tolist()
        self.selected_features_dict['L1_LASSO'] = lasso_selected

        # 4. Tree Importance (ExtraTrees)
        et = ExtraTreesClassifier(n_estimators=150, random_state=self.random_state, class_weight='balanced')
        et.fit(X_filtered, y)
        et_importances = pd.Series(et.feature_importances_, index=feature_names).sort_values(ascending=False)
        tree_selected = et_importances.head(actual_k).index.tolist()
        self.selected_features_dict['Tree_Importance'] = tree_selected

        # 5. Consensus Feature Selection
        votes = pd.Series(0, index=feature_names)
        for method, selected in self.selected_features_dict.items():
            votes[selected] += 1

        consensus = votes[votes >= self.consensus_threshold].sort_values(ascending=False).index.tolist()
        if len(consensus) < 20:
            consensus = votes.sort_values(ascending=False).head(min(30, len(feature_names))).index.tolist()
        
        self.consensus_features = consensus
        self.selected_features_dict['Consensus'] = consensus

        # Summary dataframe
        self.feature_scores_df = pd.DataFrame({
            'Mutual_Info': mi_ranking,
            'L1_Coeff': lasso_ranking,
            'Tree_Importance': et_importances,
            'Consensus_Votes': votes
        }).sort_values(by='Consensus_Votes', ascending=False)

        return self

    def transform(self, X: pd.DataFrame, method: str = 'Consensus') -> pd.DataFrame:
        if method not in self.selected_features_dict:
            raise ValueError(f"Method {method} not recognized. Available: {list(self.selected_features_dict.keys())}")
        cols = self.selected_features_dict[method]
        return X[[c for c in cols if c in X.columns]]
