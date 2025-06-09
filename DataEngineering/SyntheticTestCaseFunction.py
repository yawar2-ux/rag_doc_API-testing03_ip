import warnings
import requests
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)

class GenericCSVTestGenerator:
    def __init__(self, ollama_url: str = None, model: str = "llama3.2:latest"):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_BASE_URL")
        self.model = model
        self.current_test_id = 1
        self.potential_primary_keys = []
        self.numeric_bounds = (-1000000, 1000000)
        self.date_bounds = ('1900-01-01', '2100-12-31')
        self.string_length_bounds = (1, 100)
        self.max_categories = 50
        self.z_score_threshold = 3.0

    def identify_primary_keys(self, df: pd.DataFrame) -> List[str]:
        primary_keys = []
        for column in df.columns:
            unique_count = df[column].nunique()
            total_count = len(df[column])
            if unique_count == total_count and not df[column].isnull().any():
                primary_keys.append(column)
        return primary_keys

    def infer_column_type(self, series: pd.Series) -> str:
        datetime_sample = pd.to_datetime(series.dropna().head(), errors='coerce')
        if datetime_sample.notna().all():
            return "datetime"
        numeric_sample = pd.to_numeric(series.dropna().head(), errors='coerce')
        if numeric_sample.notna().all():
            if series.dropna().apply(lambda x: float(x) % 1 == 0).all():
                return "integer"
            return "float"
        bool_values = {'true', 'false', '1', '0', 'yes', 'no'}
        if series.dropna().astype(str).str.lower().isin(bool_values).all():
            return "boolean"
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.1:
            return "categorical"
        return "string"

    def add_validation(self, validations: List[Dict], test_type: str, description: str,
                        column: str, validation: str, result: str, status: str) -> None:
        validations.append({
            "Test ID": self.current_test_id,
            "Test Type": test_type,
            "Description": description,
            "Column": column,
            "Validation": validation,
            "Result": result,
            "Status": "Pass" if status == "Pass" else "Fail"
        })
        self.current_test_id += 1

    def generate_column_validations(self, df: pd.DataFrame, column: str) -> List[Dict]:
        series = df[column]
        data_type = self.infer_column_type(series)
        validations = []

        null_count = series.isnull().sum()
        self.add_validation(
            validations,
            "Missing Data",
            f"Check for null values in {column}",
            column,
            "Values should not be null",
            f"Found {null_count} null values",
            "Fail" if null_count > 0 else "Pass"
        )

        if column not in self.potential_primary_keys:
            duplicate_count = series.duplicated().sum()
            self.add_validation(
                validations,
                "Duplicates",
                f"Check for duplicate values in {column}",
                column,
                "Values should be unique",
                f"Found {duplicate_count} duplicates",
                "Fail" if duplicate_count > 0 else "Pass"
            )

        if data_type in ["integer", "float"]:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) > 0:
                lower_bound, upper_bound = self.numeric_bounds
                out_of_bounds = numeric_series[
                    (numeric_series < lower_bound) |
                    (numeric_series > upper_bound)
                ]
                self.add_validation(
                    validations,
                    "Boundary Check",
                    f"Check value boundaries in {column}",
                    column,
                    f"Values should be between {lower_bound} and {upper_bound}",
                    f"Found {len(out_of_bounds)} values outside bounds",
                    "Fail" if len(out_of_bounds) > 0 else "Pass"
                )

                if column not in self.potential_primary_keys and len(numeric_series) > 1:
                    mean = numeric_series.mean()
                    std = numeric_series.std()
                    if std > 0:
                        z_scores = np.abs((numeric_series - mean) / std)
                        outliers = numeric_series[z_scores > self.z_score_threshold]
                        self.add_validation(
                            validations,
                            "Outliers",
                            f"Check for outliers in {column}",
                            column,
                            f"Values should have Z-score <= {self.z_score_threshold}",
                            f"Found {len(outliers)} outliers",
                            "Fail" if len(outliers) > 0 else "Pass"
                        )

        elif data_type == "datetime":
            datetime_series = pd.to_datetime(series, errors='coerce')
            invalid_count = datetime_series.isna().sum()
            self.add_validation(
                validations,
                "Date Format",
                f"Check date format in {column}",
                column,
                "All dates must be in valid format",
                f"Found {invalid_count} invalid dates",
                "Fail" if invalid_count > 0 else "Pass"
            )

            if not datetime_series.isna().all():
                min_date = pd.Timestamp(self.date_bounds[0])
                max_date = pd.Timestamp(self.date_bounds[1])
                invalid_dates = datetime_series[
                    (datetime_series < min_date) |
                    (datetime_series > max_date)
                ].count()
                self.add_validation(
                    validations,
                    "Date Range",
                    f"Check date range in {column}",
                    column,
                    f"Dates should be between {min_date.date()} and {max_date.date()}",
                    f"Found {invalid_dates} dates outside range",
                    "Fail" if invalid_dates > 0 else "Pass"
                )

        elif data_type == "string":
            min_length, max_length = self.string_length_bounds
            lengths = series.str.len()
            invalid_lengths = series[
                (lengths < min_length) |
                (lengths > max_length)
            ].count()
            self.add_validation(
                validations,
                "String Length",
                f"Check string length in {column}",
                column,
                f"Length should be between {min_length} and {max_length}",
                f"Found {invalid_lengths} invalid lengths",
                "Fail" if invalid_lengths > 0 else "Pass"
            )

        return validations

    def process_csv(self, input_path: str, output_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(input_path, dtype=str)
            self.potential_primary_keys = self.identify_primary_keys(df)
            all_validations = []
            for column in df.columns:
                column_validations = self.generate_column_validations(df, column)
                all_validations.extend(column_validations)
            results_df = pd.DataFrame(all_validations)
            results_df.to_csv(output_path, index=False)
            return results_df
        except Exception as e:
            raise Exception(f"Error processing CSV: {str(e)}")
