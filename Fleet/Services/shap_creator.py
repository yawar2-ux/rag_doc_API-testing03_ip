#!/usr/bin/env python3
"""
SHAP (SHapley Additive exPlanations) analysis utilities.
Contains the SHAPAnalyzer class using Decision Tree and beeswarm plots only.
"""

import os
import uuid
import json
import pickle
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from io import BytesIO

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import base64
import warnings
from pydantic import BaseModel

warnings.filterwarnings('ignore')

class SHAPConfig(BaseModel):
    """Configuration for SHAP analysis parameters"""
    target_column: Optional[str] = None
    max_display: int = 20
    sample_size: Optional[int] = None  # For large datasets, sample for faster processing

def safe_float_for_json(value):
    """Convert value to JSON-safe float."""
    if pd.isna(value) or np.isinf(value) or abs(value) > 1e308:
        return 0.0
    return float(np.clip(value, -1e308, 1e308))

class SHAPAnalyzer:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.label_encoder = None
        self.class_mapping = {}
    
    def prepare_data(self, df, target_column, sample_size=None):
        """Prepare data for SHAP analysis"""
        if target_column and target_column in df.columns:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
        else:
            # If no target specified, use all columns as features
            X = df
            y = None
        
        # Remove any remaining categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X = X.drop(categorical_cols, axis=1)
        
        # Sample data if requested for faster processing
        if sample_size and len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[sample_indices]
            if y is not None:
                y = y.iloc[sample_indices]
        
        self.feature_names = X.columns.tolist()
        return X, y
    
    def train_decision_tree_model(self, X, y):
        """Train a Decision Tree model"""
        if y is None:
            raise ValueError("Cannot train model without target variable")
        
        # Encode target if categorical
        if y.dtype == 'object':
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self.class_mapping = {
                encoded: original for original, encoded in 
                zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))
            }
        else:
            y_encoded = y
        
        # Determine if classification or regression
        if len(np.unique(y_encoded)) <= 10:  # Assuming classification if <= 10 unique values
            self.model = DecisionTreeClassifier(
                random_state=42,
                max_depth=10,  # Limit depth to prevent overfitting
                min_samples_split=10,
                min_samples_leaf=5
            )
        else:
            self.model = DecisionTreeRegressor(
                random_state=42,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5
            )
        
        self.model.fit(X, y_encoded)
    
    def generate_shap_explanations(self, X, config: SHAPConfig):
        """Generate SHAP explanations with safe float conversion."""
        # Create explainer - use Tree explainer for decision trees
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X)
        
        results = {
            "analysis_info": {
                "num_samples": int(len(X)),
                "num_features": int(len(X.columns)),
                "feature_names": self.feature_names,
                "model_type": type(self.model).__name__,
                "class_mapping": self.class_mapping
            },
            "visualizations": {},
            "feature_importance": {}
        }
        
        # Generate visualizations and importance with safe handling
        try:
            results["visualizations"]["beeswarm"] = self._create_beeswarm_plot(X, config.max_display)
            results["feature_importance"] = self._calculate_feature_importance()
        except Exception as e:
            results["visualizations"]["beeswarm"] = ""
            results["feature_importance"] = {"scores": {}, "rankings": {}}
        
        return results
    
    def _create_beeswarm_plot(self, X, max_display):
        """Create beeswarm plot for Decision Tree SHAP values"""
        plt.figure(figsize=(14, 10))
        
        # Handle multiclass case
        if isinstance(self.shap_values, list) and len(self.shap_values) > 1:
            # For multiclass, use the first class or aggregate
            shap_values_to_plot = self.shap_values[0]  # Use first class
            title = 'Feature Importance (SHAP Beeswarm Plot - First Class)'
        else:
            # For binary classification or regression
            shap_values_to_plot = self.shap_values
            title = 'Feature Importance (SHAP Beeswarm Plot)'
        
        # Create SHAP beeswarm plot
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_values_to_plot,
                base_values=np.full(len(X), self.explainer.expected_value if not isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value[0]),
                data=X.values,
                feature_names=self.feature_names
            ),
            max_display=max_display,
            show=False
        )
        
        plt.title(title, fontsize=14)
        plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
        plt.tight_layout()
        
        return self._plot_to_base64()
    
    def _calculate_feature_importance(self):
        """Calculate feature importance scores from SHAP values with safe conversion."""
        if isinstance(self.shap_values, list):
            # Multiclass: aggregate across classes
            importance_scores = np.mean([np.abs(shap_vals).mean(axis=0) for shap_vals in self.shap_values], axis=0)
        else:
            # Binary classification or regression
            importance_scores = np.abs(self.shap_values).mean(axis=0)
        
        # Create feature importance dictionary with safe conversions
        feature_importance = {
            "scores": {
                feature: safe_float_for_json(score) 
                for feature, score in zip(self.feature_names, importance_scores)
            },
            "rankings": {
                feature: int(rank) 
                for rank, feature in enumerate(
                    sorted(self.feature_names, 
                          key=lambda x: importance_scores[self.feature_names.index(x)], 
                          reverse=True), 1
                )
            }
        }
        
        return feature_importance
    
    def _plot_to_base64(self):
        """Convert current matplotlib plot to base64 string"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()