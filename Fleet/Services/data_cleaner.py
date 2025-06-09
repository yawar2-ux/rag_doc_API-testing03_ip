#!/usr/bin/env python3
"""
Dataset cleaning and preprocessing utilities.
Contains the DataCleaner class and related helper functions.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
import warnings
from pydantic import BaseModel
from typing import Optional, List

warnings.filterwarnings('ignore')

class CleaningConfig(BaseModel):
    """Configuration for dataset cleaning parameters"""
    target_columns: Optional[List[str]] = None
    high_cardinality_threshold: int = 10
    k_best_features: int = 15
    standardize: bool = True
    remove_duplicates: bool = True
    remove_datetime: bool = True
    remove_constant: bool = True
    remove_high_cardinality: bool = True

class DataCleaner:
    def __init__(self):
        self.categorical_encoders = {}
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.standard_scaler = None
        self.kbest_selector = None
        self.feature_stats = {
            'removed_columns': {},
            'missing_stats': {},
            'numerical_stats': {},
            'categorical_stats': {},
            'selected_features': {}
        }
        
    def detect_datetime_columns(self, df):
        """Detect columns containing date/time information"""
        datetime_columns = []
        
        for col in df.columns:
            # Skip if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Try parsing as datetime
            try:
                pd.to_datetime(df[col])
                datetime_columns.append(col)
            except:
                continue
                
        return datetime_columns
    
    def detect_constant_columns(self, df):
        """Detect columns with constant values"""
        return [col for col in df.columns if df[col].nunique() == 1]
    
    def detect_high_cardinality_categorical(self, df, threshold=10, target_cols=None):
        """Detect categorical columns with cardinality > threshold, excluding target columns"""
        high_card_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            # Skip target columns
            if target_cols and col in target_cols:
                continue
                
            # Count unique values
            unique_count = df[col].nunique()
            
            # Add to high cardinality list if exceeds threshold
            if unique_count > threshold:
                high_card_cols.append(col)
        
        return high_card_cols
    
    def handle_missing_values(self, df, target_cols):
        """Handle missing values separately for numerical and categorical columns"""
        df_processed = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Remove target columns from processing lists
        numerical_cols = [col for col in numerical_cols if col not in target_cols]
        categorical_cols = [col for col in categorical_cols if col not in target_cols]
        
        # Initialize imputers if not already done
        if self.numerical_imputer is None:
            self.numerical_imputer = SimpleImputer(strategy='median')
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Store missing value statistics
        self.feature_stats['missing_stats'] = {
            'numerical': {col: int(df_processed[col].isnull().sum()) for col in numerical_cols},
            'categorical': {col: int(df_processed[col].isnull().sum()) for col in categorical_cols}
        }
        
        # Impute missing values
        if len(numerical_cols) > 0:
            df_processed[numerical_cols] = self.numerical_imputer.fit_transform(df_processed[numerical_cols])
        if len(categorical_cols) > 0:
            df_processed[categorical_cols] = self.categorical_imputer.fit_transform(df_processed[categorical_cols])
        
        return df_processed
    
    def encode_categorical(self, df, target_cols):
        """Encode categorical variables based on cardinality"""
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target columns from processing list if they are categorical
        categorical_cols = [col for col in categorical_cols if col not in target_cols]
        
        # Store encoding statistics
        self.feature_stats['categorical_stats'] = {}
        
        for col in categorical_cols:
            cardinality = df_processed[col].nunique()
            self.feature_stats['categorical_stats'][col] = {
                'cardinality': int(cardinality),
                'encoding_type': 'label' if cardinality == 2 else 'onehot'
            }
            
            if cardinality == 2:
                # Binary categorical: use label encoding
                if col not in self.categorical_encoders:
                    self.categorical_encoders[col] = LabelEncoder()
                df_processed[col] = self.categorical_encoders[col].fit_transform(df_processed[col])
            else:
                # Multi-category: use one-hot encoding
                dummies = pd.get_dummies(df_processed[col], prefix=col)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed.drop(col, axis=1, inplace=True)
        
        return df_processed

    def normalize_and_standardize(self, df, target_cols):
        """Apply standardization and keep original column names"""
        df_processed = df.copy()
        numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        
        # Remove target columns from processing list
        numerical_cols = [col for col in numerical_cols if col not in target_cols]
        
        if len(numerical_cols) > 0:
            # Initialize scalers if not already done
            if self.standard_scaler is None:
                self.standard_scaler = StandardScaler()
            
            # Store scaling statistics
            self.feature_stats['numerical_stats'] = {
                col: {
                    'mean': float(df_processed[col].mean()),
                    'std': float(df_processed[col].std()),
                    'min': float(df_processed[col].min()),
                    'max': float(df_processed[col].max())
                } for col in numerical_cols
            }
            
            # Create a copy of the numerical data
            X = df_processed[numerical_cols].values
            
            # Apply standardization
            X_standardized = self.standard_scaler.fit_transform(X)
            
            # Replace the original values with the standardized values
            df_processed[numerical_cols] = X_standardized
        
        return df_processed
    
    def select_kbest_features(self, df, target_col, k=15):
        """Select top K best features using mutual information"""
        # Only select from numerical columns
        feature_cols = df.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in feature_cols if col != target_col]
        
        if len(feature_cols) == 0:
            return df
            
        X = df[feature_cols]
        y = df[target_col]
        
        # Adjust k if there are fewer features than requested
        k = min(k, len(feature_cols))
        
        # Initialize selector if not already done
        if self.kbest_selector is None:
            self.kbest_selector = SelectKBest(score_func=mutual_info_regression, k=k)
        
        # Fit and transform
        X_selected = self.kbest_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.kbest_selector.get_support()].tolist()
        
        # Store feature selection statistics
        self.feature_stats['selected_features'] = {
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'scores': {
                feature: float(score) 
                for feature, score in zip(feature_cols, self.kbest_selector.scores_)
            }
        }
        
        # Create new dataframe with selected features
        df_selected = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        
        # Add back the target column
        df_selected[target_col] = df[target_col]
        
        return df_selected
    
    def clean_dataset(self, df, config: CleaningConfig):
        """Main function to clean and transform the dataset"""
        original_shape = df.shape
        self.feature_stats['original_shape'] = original_shape
        
        # Use provided target columns or default ones
        target_cols = config.target_columns or []
        
        # Store target data if target columns exist
        target_data = {}
        for col in target_cols:
            if col in df.columns:
                target_data[col] = df[col].copy()
        
        # Remove duplicates
        if config.remove_duplicates:
            df.drop_duplicates(inplace=True)
            self.feature_stats['shape_after_duplicates'] = df.shape
        
        # Detect and remove datetime columns (excluding target columns)
        if config.remove_datetime:
            datetime_cols = self.detect_datetime_columns(df)
            datetime_cols = [col for col in datetime_cols if col not in target_cols]
            df.drop(datetime_cols, axis=1, inplace=True)
            self.feature_stats['removed_columns']['datetime'] = datetime_cols
        
        # Remove constant columns (excluding target columns)
        if config.remove_constant:
            constant_cols = self.detect_constant_columns(df)
            constant_cols = [col for col in constant_cols if col not in target_cols]
            df.drop(constant_cols, axis=1, inplace=True)
            self.feature_stats['removed_columns']['constant'] = constant_cols
        
        # Remove high cardinality categorical columns (excluding target columns)
        if config.remove_high_cardinality:
            high_card_cols = self.detect_high_cardinality_categorical(
                df, threshold=config.high_cardinality_threshold, target_cols=target_cols
            )
            df.drop(high_card_cols, axis=1, inplace=True)
            self.feature_stats['removed_columns']['high_cardinality'] = high_card_cols
        
        # Handle missing values (excluding target columns)
        df = self.handle_missing_values(df, target_cols)
        
        # Encode categorical variables (excluding target columns)
        df = self.encode_categorical(df, target_cols)
        
        # Apply standardization (excluding target columns)
        if config.standardize:
            df = self.normalize_and_standardize(df, target_cols)
        
        # Restore target columns
        for col, data in target_data.items():
            df[col] = data
        
        # Select K-best features if there's a regression target
        regression_target = None
        for col in target_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                regression_target = col
                break
        
        if regression_target and config.k_best_features > 0:
            # Temporarily remove other target columns for feature selection
            other_targets = {col: df[col].copy() for col in target_cols if col != regression_target and col in df.columns}
            df_temp = df.drop([col for col in other_targets.keys()], axis=1)
            
            # Apply feature selection
            df = self.select_kbest_features(df_temp, regression_target, config.k_best_features)
            
            # Restore other target columns
            for col, data in other_targets.items():
                df[col] = data
        
        self.feature_stats['final_shape'] = df.shape
        return df