"""
Unified Data Quality and Input Validation Module
=================================================
Validates employee records against:
1. Expected 30-feature schema
2. Missing values and invalid data types
3. Numerical feature ranges
4. Valid categorical levels
5. Career timeline invariants (TotalWorkingYears >= YearsAtCompany, etc.)
6. Generates actionable validation reports
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional


CORE_REQUIRED_COLS = [
    "Age", "BusinessTravel", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked",
    "PercentSalaryHike", "PerformanceRating",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsSinceLastPromotion", "YearsWithCurrManager"
]

IMPUTABLE_SECONDARY_COLS = [
    "DailyRate", "HourlyRate", "MonthlyRate", "OverTime",
    "RelationshipSatisfaction", "YearsInCurrentRole"
]

VALID_CATEGORIES = {
    "BusinessTravel": {"Non-Travel", "Travel_Rarely", "Travel_Frequently"},
    "Department": {"Sales", "Research & Development", "Human Resources"},
    "EducationField": {"Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"},
    "Gender": {"Male", "Female"},
    "JobRole": {
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    },
    "MaritalStatus": {"Single", "Married", "Divorced"},
    "OverTime": {"Yes", "No"}
}

NUMERIC_RANGES = {
    "Age": (18, 65),
    "DailyRate": (100, 1500),
    "DistanceFromHome": (1, 35),
    "Education": (1, 5),
    "EnvironmentSatisfaction": (1, 4),
    "HourlyRate": (30, 100),
    "JobInvolvement": (1, 4),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4),
    "MonthlyIncome": (1000, 25000),
    "MonthlyRate": (2000, 30000),
    "NumCompaniesWorked": (0, 10),
    "PercentSalaryHike": (10, 30),
    "PerformanceRating": (1, 4),
    "RelationshipSatisfaction": (1, 4),
    "StockOptionLevel": (0, 3),
    "TotalWorkingYears": (0, 45),
    "TrainingTimesLastYear": (0, 6),
    "WorkLifeBalance": (1, 4),
    "YearsAtCompany": (0, 40),
    "YearsInCurrentRole": (0, 25),
    "YearsSinceLastPromotion": (0, 20),
    "YearsWithCurrManager": (0, 25)
}


class UnifiedDataValidator:
    """
    Validates single records and batch DataFrames for data quality and integrity.
    """
    def __init__(self):
        self.core_columns = CORE_REQUIRED_COLS
        self.secondary_columns = IMPUTABLE_SECONDARY_COLS
        self.all_columns = CORE_REQUIRED_COLS + IMPUTABLE_SECONDARY_COLS

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validates a single employee profile dictionary.
        Returns (is_valid, list_of_violations).
        """
        violations = []

        # 1. Core missing columns
        for col in self.core_columns:
            if col not in record:
                violations.append({
                    "column": col,
                    "value": None,
                    "type": "MissingColumn",
                    "severity": "ERROR",
                    "message": f"Core feature '{col}' is missing from input profile."
                })

        # 2. Secondary missing columns (warning only, will be imputed)
        for col in self.secondary_columns:
            if col not in record or record[col] is None or pd.isna(record[col]):
                violations.append({
                    "column": col,
                    "value": None,
                    "type": "ImputableSecondaryMissing",
                    "severity": "WARNING",
                    "message": f"Secondary feature '{col}' is missing and will be auto-imputed using domain median/mode."
                })

        # 3. Null or Empty values in Core columns
        for col in self.core_columns:
            if col in record:
                val = record[col]
                if val is None or (isinstance(val, str) and val.strip() == "") or pd.isna(val):
                    violations.append({
                        "column": col,
                        "value": val,
                        "type": "MissingValue",
                        "severity": "ERROR",
                        "message": f"Core feature '{col}' contains null, empty or missing value."
                    })

        # 3. Categorical levels check
        for cat_col, valid_set in VALID_CATEGORIES.items():
            if cat_col in record and record[cat_col] is not None and not pd.isna(record[cat_col]):
                val_str = str(record[cat_col]).strip()
                if val_str not in valid_set:
                    violations.append({
                        "column": cat_col,
                        "value": val_str,
                        "type": "InvalidCategory",
                        "severity": "ERROR",
                        "message": f"Value '{val_str}' is not in valid categorical options: {sorted(list(valid_set))}."
                    })

        # 4. Numerical range check
        for num_col, (min_v, max_v) in NUMERIC_RANGES.items():
            if num_col in record and record[num_col] is not None and not pd.isna(record[num_col]):
                try:
                    num_val = float(record[num_col])
                    if num_val < min_v or num_val > max_v:
                        violations.append({
                            "column": num_col,
                            "value": num_val,
                            "type": "RangeViolation",
                            "severity": "WARNING",
                            "message": f"Value {num_val} is outside expected range [{min_v}, {max_v}]."
                        })
                except (ValueError, TypeError):
                    violations.append({
                        "column": num_col,
                        "value": record[num_col],
                        "type": "DataTypeError",
                        "severity": "ERROR",
                        "message": f"Feature '{num_col}' must be numeric, received: {record[num_col]}."
                    })

        # 5. Timeline Invariant checks
        try:
            twy = float(record.get("TotalWorkingYears", 0))
            yac = float(record.get("YearsAtCompany", 0))
            yicr = float(record.get("YearsInCurrentRole", 0))
            yslp = float(record.get("YearsSinceLastPromotion", 0))
            ywcm = float(record.get("YearsWithCurrManager", 0))
            age = float(record.get("Age", 0))

            if yac > twy:
                violations.append({
                    "column": "YearsAtCompany",
                    "value": f"YearsAtCompany={yac} > TotalWorkingYears={twy}",
                    "type": "TimelineInvariantViolation",
                    "severity": "ERROR",
                    "message": "Logical Invariant Broken: Years at company cannot exceed total working years."
                })
            if yicr > yac:
                violations.append({
                    "column": "YearsInCurrentRole",
                    "value": f"YearsInCurrentRole={yicr} > YearsAtCompany={yac}",
                    "type": "TimelineInvariantViolation",
                    "severity": "ERROR",
                    "message": "Logical Invariant Broken: Years in current role cannot exceed years at company."
                })
            if yslp > yac:
                violations.append({
                    "column": "YearsSinceLastPromotion",
                    "value": f"YearsSinceLastPromotion={yslp} > YearsAtCompany={yac}",
                    "type": "TimelineInvariantViolation",
                    "severity": "ERROR",
                    "message": "Logical Invariant Broken: Years since last promotion cannot exceed years at company."
                })
            if ywcm > yac:
                violations.append({
                    "column": "YearsWithCurrManager",
                    "value": f"YearsWithCurrManager={ywcm} > YearsAtCompany={yac}",
                    "type": "TimelineInvariantViolation",
                    "severity": "ERROR",
                    "message": "Logical Invariant Broken: Years with current manager cannot exceed years at company."
                })
            if age - twy < 15:
                violations.append({
                    "column": "Age",
                    "value": f"Age={age}, TotalWorkingYears={twy}",
                    "type": "TimelineInvariantViolation",
                    "severity": "WARNING",
                    "message": f"Implausible starting age (Age - TotalWorkingYears = {age - twy:.1f} < 15)."
                })
        except Exception:
            pass

        has_fatal_errors = any(v["severity"] == "ERROR" for v in violations)
        return not has_fatal_errors, violations

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates an entire batch DataFrame.
        Returns a dictionary with validation summary and row-by-row violation details.
        """
        total_records = len(df)
        row_violations = []
        valid_indices = []
        invalid_indices = []

        # Check core required columns present in dataframe
        missing_cols = [c for c in self.core_columns if c not in df.columns]
        if missing_cols:
            return {
                "is_valid": False,
                "total_records": total_records,
                "valid_records_count": 0,
                "invalid_records_count": total_records,
                "schema_error": f"Uploaded dataset is missing required columns: {missing_cols}",
                "row_violations": [],
                "valid_indices": [],
                "invalid_indices": list(range(total_records))
            }

        for idx, row in df.iterrows():
            rec_dict = row.to_dict()
            is_valid, viols = self.validate_record(rec_dict)
            if is_valid:
                valid_indices.append(idx)
            else:
                invalid_indices.append(idx)

            if viols:
                row_violations.append({
                    "row_index": int(idx),
                    "violations": viols
                })

        return {
            "is_valid": len(invalid_indices) == 0,
            "total_records": total_records,
            "valid_records_count": len(valid_indices),
            "invalid_records_count": len(invalid_indices),
            "schema_error": None,
            "row_violations": row_violations[:50],  # Return up to 50 detailed violations
            "total_violation_count": len(row_violations),
            "valid_indices": valid_indices,
            "invalid_indices": invalid_indices
        }
