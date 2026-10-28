"""
HR Domain Feature Engineering Module (Version 2).
Constructs career momentum, promotion velocity, compensation equity, supervisory stability,
burnout hazard, and logarithmic representations for employee turnover prediction.
Strictly adheres to scikit-learn Transformer API (fit/transform) to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Optional

class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant feature engineering transformer for HR employee turnover data.
    Learns statistical aggregations strictly on the training partition and applies them to test/unseen data.
    """
    def __init__(self, include_interaction_ratios: bool = True, impute_missing: bool = True):
        self.include_interaction_ratios = include_interaction_ratios
        self.impute_missing = impute_missing
        self.travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
        self.role_income_medians: Dict[str, float] = {}
        self.dept_income_medians: Dict[str, float] = {}
        self.role_income_stds: Dict[str, float] = {}
        self.imputation_values: Dict[str, float] = {}
        self.feature_names_out: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_df = X.copy()
        
        # Learn group-level median & std incomes strictly from training partition
        if 'JobRole' in X_df.columns and 'MonthlyIncome' in X_df.columns:
            self.role_income_medians = X_df.groupby('JobRole')['MonthlyIncome'].median().to_dict()
            self.role_income_stds = X_df.groupby('JobRole')['MonthlyIncome'].std().fillna(1.0).to_dict()
        if 'Department' in X_df.columns and 'MonthlyIncome' in X_df.columns:
            self.dept_income_medians = X_df.groupby('Department')['MonthlyIncome'].median().to_dict()
            
        # Learn imputation medians for numeric columns
        for col in X_df.select_dtypes(include=[np.number]).columns:
            self.imputation_values[col] = float(X_df[col].median())
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # 1. Impute missing values learned from fit (e.g. YearsWithCurrManager)
        if self.impute_missing:
            for col, val in self.imputation_values.items():
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna(val)
                    
        # 2. Travel Intensity Score
        if 'BusinessTravel' in df.columns:
            df['BusinessTravelScore'] = df['BusinessTravel'].map(self.travel_map).fillna(1).astype(float)
        else:
            df['BusinessTravelScore'] = 1.0

        if self.include_interaction_ratios:
            # 3. Tenure & Career Velocity Ratios
            if 'YearsAtCompany' in df.columns and 'Age' in df.columns:
                df['TenureToAgeRatio'] = df['YearsAtCompany'] / np.maximum(1.0, df['Age'] - 18.0 + 1.0)
            
            if 'TotalWorkingYears' in df.columns and 'Age' in df.columns:
                df['TotalExpToAgeRatio'] = df['TotalWorkingYears'] / np.maximum(1.0, df['Age'] - 18.0 + 1.0)
            
            if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
                df['CompanyLoyaltyRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1.0)
                df['PriorCompanyExperience'] = np.maximum(0.0, df['TotalWorkingYears'] - df['YearsAtCompany'])
            
            if 'NumCompaniesWorked' in df.columns and 'TotalWorkingYears' in df.columns:
                df['JobHoppingRate'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1.0)
                df['AvgYearsPerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1.0)
            
            if 'JobLevel' in df.columns and 'TotalWorkingYears' in df.columns:
                df['CareerPaceIndex'] = df['JobLevel'] / (df['TotalWorkingYears'] + 1.0)
            
            # 4. Career Stagnation & Manager Alignment
            if 'YearsSinceLastPromotion' in df.columns and 'YearsAtCompany' in df.columns:
                df['PromotionStagnationRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1.0)
                df['PromotionLagScore'] = np.maximum(0.0, df['YearsAtCompany'] - df['YearsSinceLastPromotion'])
            
            if 'YearsWithCurrManager' in df.columns and 'YearsAtCompany' in df.columns:
                df['ManagerBondRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1.0)
                df['ManagerTenureMismatch'] = np.abs(df['YearsAtCompany'] - df['YearsWithCurrManager'])
                df['SupervisoryChurnFlag'] = ((df['YearsAtCompany'] >= 3) & (df['YearsWithCurrManager'] <= 1.0)).astype(float)
                
            # 5. Compensation Equity & Relative Standing
            if 'MonthlyIncome' in df.columns:
                if 'TotalWorkingYears' in df.columns:
                    df['IncomePerWorkingYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1.0)
                if 'Age' in df.columns:
                    df['IncomePerAge'] = df['MonthlyIncome'] / (df['Age'] + 1.0)
                if 'JobLevel' in df.columns:
                    df['IncomeToJobLevelRatio'] = df['MonthlyIncome'] / (df['JobLevel'] * 1000.0 + 1.0)
                    df['CompPenetrationIndex'] = df['MonthlyIncome'] / (df['JobLevel'] * 2500.0 + 1000.0)
                if 'JobRole' in df.columns and self.role_income_medians:
                    role_median = df['JobRole'].map(self.role_income_medians).fillna(df['MonthlyIncome'].median())
                    df['RoleIncomeDisparity'] = df['MonthlyIncome'] / (role_median + 1.0)
                if 'Department' in df.columns and self.dept_income_medians:
                    dept_median = df['Department'].map(self.dept_income_medians).fillna(df['MonthlyIncome'].median())
                    df['DeptIncomeDisparity'] = df['MonthlyIncome'] / (dept_median + 1.0)
                    
            if 'PercentSalaryHike' in df.columns and 'PerformanceRating' in df.columns:
                df['HikeToPerformanceRatio'] = df['PercentSalaryHike'] / (df['PerformanceRating'] + 1e-5)
                
            if 'StockOptionLevel' in df.columns:
                df['HasStockOptions'] = (df['StockOptionLevel'] > 0).astype(float)
                df['HighStockOptions'] = (df['StockOptionLevel'] >= 2).astype(float)
                if 'PercentSalaryHike' in df.columns:
                    df['EquityIncentiveScore'] = (df['StockOptionLevel'] + 1.0) * (df['PercentSalaryHike'] / 10.0)
                
            # 6. Workload, Commute, and Burnout Indices
            if 'DistanceFromHome' in df.columns:
                df['CommuteBurdenIndex'] = df['DistanceFromHome'] * (df['BusinessTravelScore'] + 1.0)
                df['CommuteBurdenExponential'] = df['DistanceFromHome'] * ((1.0 + df['BusinessTravelScore']) ** 1.5)
                
            sat_cols = [c for c in ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'JobInvolvement'] if c in df.columns]
            if len(sat_cols) >= 3:
                df['CompositeSatisfaction'] = df[sat_cols].sum(axis=1)
                df['SatisfactionMin'] = df[sat_cols].min(axis=1)
                df['SatisfactionRange'] = df[sat_cols].max(axis=1) - df['SatisfactionMin']
                
            if 'WorkLifeBalance' in df.columns and 'JobSatisfaction' in df.columns:
                df['WorkLifeDeficit'] = (5.0 - df['WorkLifeBalance']) * (5.0 - df['JobSatisfaction'])
                if 'EnvironmentSatisfaction' in df.columns:
                    df['HolisticBurnoutHazard'] = (5.0 - df['JobSatisfaction']) * (5.0 - df['WorkLifeBalance']) * (5.0 - df['EnvironmentSatisfaction'])
                
            # 7. Logarithmic non-linear feature transforms
            if 'MonthlyIncome' in df.columns:
                df['LogMonthlyIncome'] = np.log1p(np.maximum(0.0, df['MonthlyIncome']))
            if 'DistanceFromHome' in df.columns:
                df['LogDistanceFromHome'] = np.log1p(np.maximum(0.0, df['DistanceFromHome']))
            if 'TotalWorkingYears' in df.columns:
                df['LogTotalWorkingYears'] = np.log1p(np.maximum(0.0, df['TotalWorkingYears']))

            # 8. Behavioral Risk Archetype Flags
            if 'JobSatisfaction' in df.columns and 'EnvironmentSatisfaction' in df.columns and 'WorkLifeBalance' in df.columns:
                df['SevereDissatisfactionFlag'] = ((df['JobSatisfaction'] == 1) | 
                                                   (df['EnvironmentSatisfaction'] == 1) | 
                                                   (df['WorkLifeBalance'] == 1)).astype(float)
                
            if 'Age' in df.columns and 'JobLevel' in df.columns and 'YearsAtCompany' in df.columns:
                df['EarlyCareerFlightRisk'] = ((df['Age'] < 30) & (df['JobLevel'] <= 2) & (df['YearsAtCompany'] <= 2)).astype(float)
                
            if 'YearsSinceLastPromotion' in df.columns and 'JobSatisfaction' in df.columns:
                df['StagnantLowSatisfaction'] = ((df['YearsSinceLastPromotion'] >= 4) & (df['JobSatisfaction'] <= 2)).astype(float)
                
            if 'StockOptionLevel' in df.columns and 'JobInvolvement' in df.columns and 'WorkLifeBalance' in df.columns:
                df['HighRetentionAnchor'] = ((df['StockOptionLevel'] >= 1) & (df['JobInvolvement'] >= 3) & (df['WorkLifeBalance'] >= 3)).astype(float)

        self.feature_names_out = df.columns.tolist()
        return df

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_out
