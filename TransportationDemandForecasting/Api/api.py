#!/usr/bin/env python3
"""
FastAPI router for transportation ridership data pipeline:
1. Upload CSV 
2. Generate YData profiling report (auto-processes uploaded CSV)
3. Clean data (auto-processes uploaded CSV)
4. Train model (auto-processes cleaned CSV with 'ridership' target)
5. Make predictions (auto-processes trained model with test CSV)
"""

from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import os
from pydantic import BaseModel
import chromadb
import uuid
import json
import pickle
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import base64
import io
import warnings
import groq
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Path as FastAPIPath, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ydata_profiling import ProfileReport
from chromadb.utils import embedding_functions
from typing import List, Optional

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Directory to store files
TEMP_DIR = Path("pipeline_files")
TEMP_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Initialize Groq client
groq_client = None

# Pydantic models for chat
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    reference_data: List[dict]  # Structured data for easy tabulation
    raw_context: List[str]  # Original context for debugging

# Default collection prefix for chat
CHAT_COLLECTION_PREFIX = "chat_data_"

def init_groq_client():
    global groq_client
    api_key = "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"
    groq_client = groq.Groq(api_key=api_key)

def get_groq_client():
    if groq_client is None:
        init_groq_client()
    return groq_client

# Initialize embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
class ModelRegistrationRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    description: Optional[str] = None
    stage: str = "None"  # None, Staging, Production, Archived
    
class DriftDetectionRequest(BaseModel):
    drift_threshold: float = 0.05  # P-value threshold for drift detection
    include_feature_drift: bool = True
    include_target_drift: bool = True

# ADD THIS CONFIGURATION (around line 76 after other configurations)
# MLflow configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"  # Default MLflow UI URL
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# Fast token estimation (4 chars ≈ 1 token)
def estimate_tokens(text: str) -> int:
    """Fast token estimation without heavy computation"""
    return len(text) // 4

def truncate_context_fast(context_docs: List[str], max_chars: int = 32000) -> str:
    """Fast context truncation using character limits"""
    context = ""
    current_chars = 0
    
    for doc in context_docs:
        doc_length = len(doc)
        if current_chars + doc_length > max_chars:
            # Add partial document if there's significant space
            remaining_chars = max_chars - current_chars
            if remaining_chars > 500:  # Only add if meaningful space left
                context += doc[:remaining_chars] + "...\n\n"
            break
        
        context += doc + "\n\n"
        current_chars += doc_length
    
    return context.strip()

def parse_reference_data(context_docs: List[str]) -> List[dict]:
    """Parse context documents into structured data for frontend tables"""
    structured_data = []
    
    for doc in context_docs:
        # Split document into rows
        rows = doc.split('\n')
        
        for row in rows:
            if not row.strip() or row.startswith('Row '):
                continue
                
            # Parse key-value pairs from each row
            row_data = {}
            pairs = row.split('; ')
            
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    row_data[key.strip()] = value.strip()
            
            if row_data:  # Only add if we found data
                structured_data.append(row_data)
    
    return structured_data[:20]  # Limit to 20 rows for frontend performance

def process_csv_data(df: pd.DataFrame, chunk_size: int = 200) -> List[dict]:
    """Process CSV data efficiently into chunks for vector storage"""
    documents = []
    
    # Process in chunks for better performance with large CSVs
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i + chunk_size]
        
        # Convert chunk to text
        chunk_text = ""
        for idx, row in chunk_df.iterrows():
            row_text = ""
            for col, val in row.items():
                if pd.notna(val):
                    row_text += f"{col}: {val}; "
            chunk_text += f"Row {idx}: {row_text.strip()}\n"
        
        documents.append({
            "id": str(uuid.uuid4()),
            "text": chunk_text.strip(),
            "metadata": {
                "chunk_start": i,
                "chunk_end": min(i + chunk_size, len(df)),
                "source": "csv_upload"
            }
        })
    
    return documents

# Create router instead of app
TransportationDemandForecasting_router = APIRouter()

def convert_numpy_types(obj):
    """Convert numpy data types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
# Import inference module
class RidershipInference:
    """
    Inference class for ridership prediction using trained XGBoost models.
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.is_loaded = False
        
        # Expected columns from cleaned data
        self.expected_columns = [
            'datetime', 'zone', 'location', 'prev_week_same_period',
            'prev_same_period_ridership', 'prev_day_same_period', 
            'ridership_rolling_7', 'employment_density', 'population_density',
            'transit_accessibility', 'commercial_floor_area', 'transit_frequency',
            'walk_score', 'competing_modes', 'parking_cost', 'income_level',
            'vehicle_ownership_rate', 'time_period', 'ridership'
        ]
    
    def load_model(self, model_path: str):
        """Load trained model from pickle file."""
        try:
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
            
            self.model = model_dict['model']
            self.feature_columns = model_dict['feature_columns']
            self.label_encoders = model_dict.get('label_encoders', {})
            self.is_loaded = True
            
            return {
                "status": "success",
                "message": "Model loaded successfully",
                "model_type": str(type(self.model).__name__),
                "feature_count": int(len(self.feature_columns)),
                "features": self.feature_columns
            }
            
        except Exception as e:
            self.is_loaded = False
            return {
                "status": "error",
                "message": f"Failed to load model: {str(e)}"
            }
    
    def validate_input_data(self, df: pd.DataFrame):
        """Validate input DataFrame for prediction."""
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check if ridership column exists (for evaluation purposes)
        has_target = 'ridership' in df.columns
        
        # Check for missing required feature columns
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Missing required features: {missing_features}")
        
        # Check for extra columns that might indicate data issues
        extra_columns = [col for col in df.columns if col not in self.expected_columns]
        if extra_columns:
            validation_results["warnings"].append(f"Unexpected columns found (will be ignored): {extra_columns}")
        
        # Check for missing values in feature columns
        if validation_results["is_valid"]:
            feature_missing = df[self.feature_columns].isnull().sum()
            missing_features_with_nulls = feature_missing[feature_missing > 0]
            if not missing_features_with_nulls.empty:
                validation_results["warnings"].append(f"Missing values found in: {missing_features_with_nulls.to_dict()}")
        
        # Check data shape
        validation_results["data_shape"] = [int(df.shape[0]), int(df.shape[1])]
        validation_results["has_target_column"] = has_target
        
        return validation_results
    
    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess input data for prediction."""
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Handle categorical columns with label encoding if encoders exist
        for col, encoder in self.label_encoders.items():
            if col in df_processed.columns:
                try:
                    # Handle unseen categories by setting them to 0 (or most frequent category)
                    df_processed[col] = df_processed[col].astype(str)
                    
                    # Check for unseen categories
                    unique_values = set(df_processed[col].unique())
                    known_values = set(encoder.classes_)
                    unseen_values = unique_values - known_values
                    
                    if unseen_values:
                        # Replace unseen values with the most frequent category
                        most_frequent = encoder.classes_[0]  # First class is usually most frequent
                        df_processed[col] = df_processed[col].replace(list(unseen_values), most_frequent)
                    
                    df_processed[col] = encoder.transform(df_processed[col])
                    
                except Exception as e:
                    # If encoding fails, set to 0
                    df_processed[col] = 0
        
        # Handle any remaining categorical columns that don't have encoders
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' and col != 'datetime':
                # Simple numeric encoding for unseen categorical columns
                df_processed[col] = pd.Categorical(df_processed[col]).codes
        
        # Handle missing values (forward fill, then backward fill, then 0)
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df_processed
    
    def predict(self, df: pd.DataFrame):
        """Make ridership predictions on input data."""
        if not self.is_loaded:
            return {
                "status": "error",
                "message": "Model not loaded. Please load a trained model first."
            }
        
        # Validate input data
        validation = self.validate_input_data(df)
        if not validation["is_valid"]:
            return {
                "status": "error",
                "message": "Data validation failed",
                "validation_errors": validation["errors"]
            }
        
        try:
            # Preprocess data
            df_processed = self.preprocess_data(df)
            
            # Extract features for prediction
            X = df_processed[self.feature_columns]
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Prepare results
            results = {
                "status": "success",
                "message": "Predictions completed successfully",
                "predictions": [float(pred) for pred in predictions],
                "prediction_count": int(len(predictions)),
                "prediction_statistics": {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions)),
                    "median": float(np.median(predictions))
                },
                "validation_info": {
                    "warnings": validation["warnings"],
                    "data_shape": validation["data_shape"],
                    "has_target_column": validation["has_target_column"]
                }
            }
            
            # If actual ridership values are available, calculate evaluation metrics
            if 'ridership' in df.columns:
                actual_values = df['ridership'].values
                mae = mean_absolute_error(actual_values, predictions)
                rmse = np.sqrt(mean_squared_error(actual_values, predictions))
                r2 = r2_score(actual_values, predictions)
                
                results["evaluation_metrics"] = {
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "mape": float(np.mean(np.abs((actual_values - predictions) / actual_values)) * 100)
                }
                
                results["actual_vs_predicted"] = [
                    {"actual": float(actual), "predicted": float(pred), "error": float(abs(actual - pred))}
                    for actual, pred in zip(actual_values, predictions)
                ]
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Prediction failed: {str(e)}"
            }

class DataDriftAnalyzer:
    """
    Data drift detection analyzer for transportation ridership data.
    Compares new data with reference (training) data to detect distribution shifts.
    """
    
    def __init__(self):
        self.reference_data = None
        self.reference_target = None
        self.feature_columns = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.drift_threshold = 0.05
        
    def set_reference_data(self, df: pd.DataFrame, target_col: str = 'ridership'):
        """Set reference data (usually training data) for drift comparison."""
        self.reference_data = df.copy()
        self.reference_target = df[target_col] if target_col in df.columns else None
        
        # Identify feature columns (exclude target and datetime columns)
        exclude_cols = [target_col, 'datetime', 'date', 'time']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Identify categorical and numerical columns
        for col in self.feature_columns:
            if df[col].dtype == 'object' or df[col].nunique() <= 10:
                self.categorical_columns.append(col)
            else:
                self.numerical_columns.append(col)
        
        return {
            "status": "success",
            "message": "Reference data set successfully",
            "reference_shape": [int(df.shape[0]), int(df.shape[1])],
            "feature_columns": len(self.feature_columns),
            "categorical_features": len(self.categorical_columns),
            "numerical_features": len(self.numerical_columns)
        }
    
    def detect_numerical_drift(self, ref_data: pd.Series, new_data: pd.Series, feature_name: str):
        """Detect drift in numerical features using Kolmogorov-Smirnov test."""
        try:
            # Remove NaN values
            ref_clean = ref_data.dropna()
            new_clean = new_data.dropna()
            
            if len(ref_clean) == 0 or len(new_clean) == 0:
                return {
                    "feature": feature_name,
                    "drift_detected": False,
                    "p_value": 1.0,
                    "test_statistic": 0.0,
                    "test_type": "KS",
                    "warning": "Insufficient data for testing"
                }
            
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(ref_clean, new_clean)
            
            # Calculate additional statistics
            ref_mean, ref_std = ref_clean.mean(), ref_clean.std()
            new_mean, new_std = new_clean.mean(), new_clean.std()
            
            mean_shift = abs(new_mean - ref_mean) / (ref_std + 1e-8)
            std_ratio = new_std / (ref_std + 1e-8)
            
            # Convert numpy types to Python native types
            result = {
                "feature": str(feature_name),
                "drift_detected": bool(p_value < self.drift_threshold),
                "p_value": float(p_value),
                "test_statistic": float(ks_statistic),
                "test_type": "Kolmogorov-Smirnov",
                "statistics": {
                    "reference_mean": float(ref_mean),
                    "reference_std": float(ref_std),
                    "new_mean": float(new_mean),
                    "new_std": float(new_std),
                    "mean_shift": float(mean_shift),
                    "std_ratio": float(std_ratio)
                },
                "drift_severity": "High" if float(p_value) < 0.01 else "Medium" if float(p_value) < 0.05 else "Low"
            }
            
            return result
            
        except Exception as e:
            return {
                "feature": str(feature_name),
                "drift_detected": False,
                "p_value": 1.0,
                "test_statistic": 0.0,
                "test_type": "KS",
                "error": str(e)
            }
    
    def detect_categorical_drift(self, ref_data: pd.Series, new_data: pd.Series, feature_name: str):
        """Detect drift in categorical features using Chi-square test."""
        try:
            # Get value counts for both datasets
            ref_counts = ref_data.value_counts()
            new_counts = new_data.value_counts()
            
            # Get all unique categories
            all_categories = set(ref_counts.index) | set(new_counts.index)
            
            if len(all_categories) <= 1:
                return {
                    "feature": str(feature_name),
                    "drift_detected": False,
                    "p_value": 1.0,
                    "test_statistic": 0.0,
                    "test_type": "Chi-square",
                    "warning": "Insufficient categories for testing"
                }
            
            # Create frequency tables
            ref_freq = []
            new_freq = []
            
            for category in all_categories:
                ref_freq.append(ref_counts.get(category, 0))
                new_freq.append(new_counts.get(category, 0))
            
            # Perform Chi-square test
            chi2_statistic, p_value = stats.chisquare(new_freq, ref_freq)
            
            # Calculate distribution similarity
            ref_prop = np.array(ref_freq) / sum(ref_freq)
            new_prop = np.array(new_freq) / sum(new_freq)
            
            # Jensen-Shannon divergence for additional measure
            try:
                js_divergence = stats.entropy((ref_prop + new_prop) / 2) - (stats.entropy(ref_prop) + stats.entropy(new_prop)) / 2
            except:
                js_divergence = 0.0
            
            # Convert numpy types to Python native types
            result = {
                "feature": str(feature_name),
                "drift_detected": bool(p_value < self.drift_threshold),
                "p_value": float(p_value),
                "test_statistic": float(chi2_statistic),
                "test_type": "Chi-square",
                "statistics": {
                    "reference_categories": int(len(ref_counts)),
                    "new_categories": int(len(new_counts)),
                    "common_categories": int(len(set(ref_counts.index) & set(new_counts.index))),
                    "js_divergence": float(js_divergence)
                },
                "drift_severity": "High" if float(p_value) < 0.01 else "Medium" if float(p_value) < 0.05 else "Low"
            }
            
            return result
            
        except Exception as e:
            return {
                "feature": str(feature_name),
                "drift_detected": False,
                "p_value": 1.0,
                "test_statistic": 0.0,
                "test_type": "Chi-square",
                "error": str(e)
            }
    
    def analyze_data_quality(self, new_data: pd.DataFrame):
        """Analyze data quality metrics for drift detection."""
        ref_data = self.reference_data
        
        quality_metrics = {
            "reference_stats": {
                "total_rows": int(len(ref_data)),
                "missing_values": int(ref_data.isnull().sum().sum()),
                "missing_percentage": float(ref_data.isnull().sum().sum() / (len(ref_data) * len(ref_data.columns)) * 100)
            },
            "new_data_stats": {
                "total_rows": int(len(new_data)),
                "missing_values": int(new_data.isnull().sum().sum()),
                "missing_percentage": float(new_data.isnull().sum().sum() / (len(new_data) * len(new_data.columns)) * 100)
            },
            "drift_indicators": {}
        }
        
        # Compare missing value patterns
        for col in self.feature_columns:
            if col in new_data.columns:
                ref_missing = ref_data[col].isnull().sum() / len(ref_data) * 100
                new_missing = new_data[col].isnull().sum() / len(new_data) * 100
                missing_diff = abs(new_missing - ref_missing)
                
                quality_metrics["drift_indicators"][col] = {
                    "reference_missing_pct": float(ref_missing),
                    "new_missing_pct": float(new_missing),
                    "missing_drift": float(missing_diff),
                    "missing_drift_severity": "High" if missing_diff > 10 else "Medium" if missing_diff > 5 else "Low"
                }
        
        return quality_metrics
    
    def detect_drift(self, new_data: pd.DataFrame, target_col: str = 'ridership'):
        """Main drift detection function."""
        if self.reference_data is None:
            return {
                "status": "error",
                "message": "Reference data not set. Please set reference data first."
            }
        
        # Validate new data
        missing_features = [col for col in self.feature_columns if col not in new_data.columns]
        if missing_features:
            return {
                "status": "error",
                "message": f"Missing features in new data: {missing_features}"
            }
        
        drift_results = {
            "status": "success",
            "message": "Drift detection completed",
            "timestamp": datetime.now().isoformat(),
            "drift_threshold": self.drift_threshold,
            "summary": {
                "total_features_tested": 0,
                "features_with_drift": 0,
                "drift_percentage": 0.0,
                "overall_drift_detected": False
            },
            "feature_drift_results": [],
            "target_drift_result": None,
            "data_quality_analysis": None
        }
        
        # Feature drift detection
        features_with_drift = 0
        
        for feature in self.feature_columns:
            if feature in new_data.columns:
                if feature in self.numerical_columns:
                    result = self.detect_numerical_drift(
                        self.reference_data[feature], 
                        new_data[feature], 
                        feature
                    )
                else:
                    result = self.detect_categorical_drift(
                        self.reference_data[feature], 
                        new_data[feature], 
                        feature
                    )
                
                drift_results["feature_drift_results"].append(result)
                
                if result["drift_detected"]:
                    features_with_drift += 1
        
        # Target drift detection (if target column exists in new data)
        if target_col in new_data.columns and self.reference_target is not None:
            target_drift = self.detect_numerical_drift(
                self.reference_target, 
                new_data[target_col], 
                target_col
            )
            drift_results["target_drift_result"] = target_drift
        
        # Data quality analysis
        drift_results["data_quality_analysis"] = self.analyze_data_quality(new_data)
        
        # Summary statistics
        total_features = len(drift_results["feature_drift_results"])
        drift_results["summary"]["total_features_tested"] = int(total_features)
        drift_results["summary"]["features_with_drift"] = int(features_with_drift)
        drift_results["summary"]["drift_percentage"] = float((features_with_drift / total_features * 100) if total_features > 0 else 0.0)
        drift_results["summary"]["overall_drift_detected"] = bool(features_with_drift > 0 or (
            drift_results["target_drift_result"] and drift_results["target_drift_result"]["drift_detected"]
        ))

        # Drift severity assessment
        high_drift_features = [r for r in drift_results["feature_drift_results"] if r.get("drift_severity") == "High"]
        medium_drift_features = [r for r in drift_results["feature_drift_results"] if r.get("drift_severity") == "Medium"]

        if len(high_drift_features) > 0:
            drift_results["summary"]["overall_severity"] = "High"
        elif len(medium_drift_features) > 0:
            drift_results["summary"]["overall_severity"] = "Medium"
        else:
            drift_results["summary"]["overall_severity"] = "Low"

        # Convert all numpy types to Python native types before returning
        drift_results = convert_numpy_types(drift_results)

        return drift_results
    

class SHAPAnalyzer:
    """
    SHAP analysis class for ridership prediction using Decision Tree.
    Creates bee swarm plots and feature importance explanations.
    """
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.feature_columns = None
        self.X_sample = None
        self.is_trained = False
    
    def train_decision_tree(self, df: pd.DataFrame, target_col: str = 'ridership', test_size: float = 0.2):
        """Train Decision Tree model on cleaned data."""
        try:
            # Prepare features and target
            exclude_cols = [target_col, 'datetime', 'date', 'time']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols]
            y = df[target_col]
            
            self.feature_columns = feature_cols
            
            # Train-test split (time-aware if datetime available)
            if 'datetime' in df.columns:
                df_sorted = df.sort_values('datetime')
                split_idx = int(len(df_sorted) * (1 - test_size))
                
                train_df = df_sorted.iloc[:split_idx]
                test_df = df_sorted.iloc[split_idx:]
                
                X_train = train_df[feature_cols]
                X_test = test_df[feature_cols]
                y_train = train_df[target_col]
                y_test = test_df[target_col]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            # Train Decision Tree model
            self.model = DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Make predictions for evaluation
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            train_metrics = {
                "mae": float(mean_absolute_error(y_train, y_train_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                "r2": float(r2_score(y_train, y_train_pred))
            }
            
            test_metrics = {
                "mae": float(mean_absolute_error(y_test, y_test_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                "r2": float(r2_score(y_test, y_test_pred))
            }
            
            # Store sample data for SHAP analysis (use smaller sample for performance)
            sample_size = min(1000, len(X_train))
            self.X_sample = X_train.sample(n=sample_size, random_state=42)
            
            self.is_trained = True
            
            return {
                "status": "success",
                "message": "Decision Tree model trained successfully",
                "model_info": {
                    "model_type": "Decision Tree Regressor",
                    "training_samples": int(X_train.shape[0]),
                    "testing_samples": int(X_test.shape[0]),
                    "total_features": len(feature_cols),
                    "sample_size_for_shap": sample_size
                },
                "performance": {
                    "training_metrics": train_metrics,
                    "testing_metrics": test_metrics,
                    "performance_grade": "Excellent" if test_metrics["r2"] > 0.8 else 
                                      "Good" if test_metrics["r2"] > 0.6 else 
                                      "Fair" if test_metrics["r2"] > 0.4 else "Needs Improvement"
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to train Decision Tree model: {str(e)}"
            }
    
    def compute_shap_values(self, background_samples: int = 100):
        """Compute SHAP values using TreeExplainer."""
        if not self.is_trained:
            return {
                "status": "error",
                "message": "Model not trained. Please train the model first."
            }
        
        try:
            # Create background dataset (smaller sample for performance)
            background_size = min(background_samples, len(self.X_sample))
            background_data = self.X_sample.sample(n=background_size, random_state=42)
            
            # Create SHAP explainer (TreeExplainer is faster for tree models)
            self.explainer = shap.TreeExplainer(self.model, background_data)
            
            # Compute SHAP values for sample data
            self.shap_values = self.explainer(self.X_sample)
            
            return {
                "status": "success",
                "message": "SHAP values computed successfully",
                "shap_info": {
                    "explainer_type": "TreeExplainer",
                    "background_samples": background_size,
                    "explained_samples": len(self.X_sample),
                    "feature_count": len(self.feature_columns)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to compute SHAP values: {str(e)}"
            }
    
    def create_beeswarm_plot(self, max_display: int = 15, figsize: tuple = (12, 8)):
        """Create SHAP bee swarm plot and return as base64 string."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        try:
            # Create the plot
            plt.figure(figsize=figsize)
            
            # Create bee swarm plot
            shap.plots.beeswarm(self.shap_values, max_display=max_display, show=False)
            
            # Customize the plot
            plt.title('SHAP Bee Swarm Plot - Feature Impact on Ridership Prediction', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            
            # Add subtitle with explanation
            plt.figtext(0.5, 0.02, 
                       'Each dot represents one prediction. Color indicates feature value (red=high, blue=low). '
                       'Position shows impact direction and magnitude.',
                       ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            plt.close()  # Ensure plot is closed even on error
            raise Exception(f"Failed to create bee swarm plot: {str(e)}")
    
    def get_feature_importance_data(self, top_n: int = 15):
        """Get feature importance data from SHAP values."""
        if self.shap_values is None:
            return {
                "status": "error",
                "message": "SHAP values not computed"
            }
        
        try:
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)
            
            # Get top N features
            top_features = importance_df.head(top_n)
            
            return {
                "status": "success",
                "feature_importance": [
                    {
                        "feature": str(row.feature),
                        "mean_abs_shap": float(row.mean_abs_shap),
                        "rank": idx + 1,
                        "percentage": float(row.mean_abs_shap / mean_abs_shap.sum() * 100)
                    }
                    for idx, row in top_features.iterrows()
                ],
                "total_features": len(self.feature_columns)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get feature importance: {str(e)}"
            }
    
    def create_comprehensive_analysis(self, max_display: int = 15):
        """Create comprehensive SHAP analysis with bee swarm plot."""
        try:
            # Compute SHAP values if not already done
            if self.shap_values is None:
                shap_result = self.compute_shap_values()
                if shap_result["status"] != "success":
                    return shap_result
            
            # Create bee swarm plot
            beeswarm_plot = self.create_beeswarm_plot(max_display=max_display)
            
            # Get feature importance data
            feature_importance = self.get_feature_importance_data(top_n=max_display)
            
            return {
                "status": "success",
                "message": "Comprehensive SHAP analysis completed",
                "visualizations": {
                    "beeswarm_plot": f"data:image/png;base64,{beeswarm_plot}"
                },
                "feature_analysis": feature_importance,
                "analysis_info": {
                    "explained_samples": len(self.X_sample),
                    "features_analyzed": len(self.feature_columns),
                    "max_features_displayed": max_display
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create comprehensive analysis: {str(e)}"
            }

class PartialDependencyAnalyzer:
    """
    Partial Dependency Plot analyzer for ridership prediction.
    Creates PDP plots to understand feature effects on model predictions.
    """
    
    def __init__(self, model_type: str = 'decision_tree'):
        self.model = None
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_trained = False
        self.model_type = model_type
        self.available_features = []
        
    def train_model(self, df: pd.DataFrame, target_col: str = 'ridership', 
                   test_size: float = 0.2, model_type: str = None):
        """Train model for partial dependency analysis."""
        if model_type:
            self.model_type = model_type
            
        try:
            # Prepare features and target
            exclude_cols = [target_col, 'datetime', 'date', 'time']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols]
            y = df[target_col]
            
            self.feature_columns = feature_cols
            self.available_features = feature_cols.copy();
            
            # Train-test split (time-aware if datetime available)
            if 'datetime' in df.columns:
                df_sorted = df.sort_values('datetime')
                split_idx = int(len(df_sorted) * (1 - test_size))
                
                train_df = df_sorted.iloc[:split_idx]
                test_df = df_sorted.iloc[split_idx:]
                
                self.X_train = train_df[feature_cols]
                self.X_test = test_df[feature_cols]
                self.y_train = train_df[target_col]
                self.y_test = test_df[target_col]
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            # Train model based on type
            if self.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            else:  # decision_tree
                self.model = DecisionTreeRegressor(
                    max_depth=15,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                )
            
            self.model.fit(self.X_train, self.y_train)
            
            # Make predictions for evaluation
            y_train_pred = self.model.predict(self.X_train)
            y_test_pred = self.model.predict(self.X_test)
            
            # Calculate metrics
            train_metrics = {
                "mae": float(mean_absolute_error(self.y_train, y_train_pred)),
                "rmse": float(np.sqrt(mean_squared_error(self.y_train, y_train_pred))),
                "r2": float(r2_score(self.y_train, y_train_pred))
            }
            
            test_metrics = {
                "mae": float(mean_absolute_error(self.y_test, y_test_pred)),
                "rmse": float(np.sqrt(mean_squared_error(self.y_test, y_test_pred))),
                "r2": float(r2_score(self.y_test, y_test_pred))
            }
            
            self.is_trained = True
            
            return {
                "status": "success",
                "message": f"{self.model_type.replace('_', ' ').title()} model trained successfully",
                "model_info": {
                    "model_type": self.model_type.replace('_', ' ').title(),
                    "training_samples": int(self.X_train.shape[0]),
                    "testing_samples": int(self.X_test.shape[0]),
                    "total_features": len(feature_cols),
                    "available_features": self.available_features
                },
                "performance": {
                    "training_metrics": train_metrics,
                    "testing_metrics": test_metrics,
                    "performance_grade": "Excellent" if test_metrics["r2"] > 0.8 else 
                                      "Good" if test_metrics["r2"] > 0.6 else 
                                      "Fair" if test_metrics["r2"] > 0.4 else "Needs Improvement"
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to train model: {str(e)}"
            }
    
    def validate_features(self, features):
        """Validate requested features for PDP analysis."""
        if not self.is_trained:
            return {
                "status": "error",
                "message": "Model not trained yet"
            }
        
        # Check if features exist
        missing_features = [f for f in features if f not in self.available_features]
        valid_features = [f for f in features if f in self.available_features]
        
        if missing_features:
            return {
                "status": "warning",
                "message": f"Some features not found: {missing_features}",
                "valid_features": valid_features,
                "missing_features": missing_features,
                "available_features": self.available_features
            }
        
        return {
            "status": "success",
            "message": "All features are valid",
            "valid_features": valid_features,
            "missing_features": [],
            "available_features": self.available_features
        }
    
    def create_single_feature_pdp(self, feature: str, grid_resolution: int = 100, figsize=(10, 6)):
        """Create partial dependency plot for a single feature."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if feature not in self.available_features:
            raise ValueError(f"Feature '{feature}' not found in available features")
        
        try:
            # Get feature index
            feature_idx = self.feature_columns.index(feature)
            
            # Compute partial dependence
            pd_result = partial_dependence(
                self.model, 
                self.X_train, 
                features=[feature_idx],
                grid_resolution=grid_resolution,
                kind='average'
            )
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot partial dependence
            ax.plot(pd_result['grid_values'][0], pd_result['average'][0], 
                   linewidth=3, color='#1f77b4')
            
            # Customize plot
            ax.set_xlabel(f'{feature}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Partial Dependence\n(Change in Predicted Ridership)', fontsize=12, fontweight='bold')
            ax.set_title(f'Partial Dependency Plot: {feature}', fontsize=14, fontweight='bold', pad=20)
            
            # Add grid and styling
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add subtitle with explanation
            plt.figtext(0.5, 0.02, 
                       f'Shows how {feature} affects ridership predictions on average, holding other features constant.',
                       ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            plt.close()
            raise Exception(f"Failed to create PDP for {feature}: {str(e)}")
    
    def create_multiple_pdp(self, features, grid_resolution: int = 100, figsize=(15, 10)):
        """Create multiple PDP plots in a grid layout."""
        if len(features) > 9:
            raise ValueError("Maximum 9 features allowed for multiple PDP plot")
        
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Validate features
        validation = self.validate_features(features)
        if validation["status"] == "error":
            raise ValueError(validation["message"])
        
        valid_features = validation["valid_features"]
        
        try:
            # Calculate subplot grid
            n_features = len(valid_features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            # Create subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_features == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # Flatten axes array for easier indexing
            axes_flat = axes.flatten() if n_features > 1 else axes
            
            for i, feature in enumerate(valid_features):
                ax = axes_flat[i]
                
                # Get feature index
                feature_idx = self.feature_columns.index(feature)
                
                # Compute partial dependence
                pd_result = partial_dependence(
                    self.model,
                    self.X_train,
                    features=[feature_idx],
                    grid_resolution=grid_resolution,
                    kind='average'
                )
                
                # Plot partial dependence
                ax.plot(pd_result['grid_values'][0], pd_result['average'][0], 
                       linewidth=2.5, color='#1f77b4')
                
                # Customize subplot
                ax.set_xlabel(feature, fontsize=10, fontweight='bold')
                ax.set_ylabel('Partial Dependence', fontsize=10, fontweight='bold')
                ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            # Hide unused subplots
            for i in range(n_features, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            # Add main title
            fig.suptitle('Partial Dependency Plots - Feature Effects on Ridership Prediction', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Add subtitle
            fig.text(0.5, 0.02, 
                    'Each plot shows how individual features affect ridership predictions on average.',
                    ha='center', fontsize=11, style='italic')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.08)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            plt.close()
            raise Exception(f"Failed to create multiple PDP plots: {str(e)}")
    
    def analyze_feature_effects(self, features, grid_resolution: int = 100):
        """Analyze feature effects using partial dependency."""
        if not self.is_trained:
            return {
                "status": "error",
                "message": "Model not trained yet"
            }
        
        # Validate features
        validation = self.validate_features(features)
        if validation["status"] == "error":
            return validation
        
        valid_features = validation["valid_features"]
        
        try:
            feature_effects = []
            
            for feature in valid_features:
                feature_idx = self.feature_columns.index(feature)
                
                # Compute partial dependence
                pd_result = partial_dependence(
                    self.model,
                    self.X_train,
                    features=[feature_idx],
                    grid_resolution=grid_resolution,
                    kind='average'
                )
                
                # Analyze effect
                pd_values = pd_result['average'][0]
                effect_range = float(np.max(pd_values) - np.min(pd_values))
                effect_std = float(np.std(pd_values))
                
                # Determine trend
                correlation = np.corrcoef(pd_result['grid_values'][0], pd_values)[0, 1]
                if abs(correlation) > 0.7:
                    trend = "positive" if correlation > 0 else "negative"
                else:
                    trend = "non-linear"
                
                feature_effects.append({
                    "feature": feature,
                    "effect_range": effect_range,
                    "effect_std": effect_std,
                    "trend": trend,
                    "correlation": float(correlation),
                    "min_effect": float(np.min(pd_values)),
                    "max_effect": float(np.max(pd_values)),
                    "mean_effect": float(np.mean(pd_values))
                })
            
            # Sort by effect range (importance)
            feature_effects.sort(key=lambda x: x["effect_range"], reverse=True)
            
            return {
                "status": "success",
                "message": "Feature effects analyzed successfully",
                "feature_effects": feature_effects,
                "analysis_summary": {
                    "most_impactful_feature": feature_effects[0]["feature"] if feature_effects else None,
                    "largest_effect_range": feature_effects[0]["effect_range"] if feature_effects else 0,
                    "features_analyzed": len(valid_features)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to analyze feature effects: {str(e)}"
            }

class SimpleTimeSeriesCleaner:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_lag_features(self, df, target_col='ridership', lags=[1, 7]):
        """Create simple lag features"""
        if 'zone' in df.columns and 'datetime' in df.columns:
            df = df.sort_values(['zone', 'datetime'])
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df.groupby('zone')[target_col].shift(lag)
            df[f'{target_col}_rolling_7'] = df.groupby('zone')[target_col].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
        return df
    
    def create_time_features(self, df):
        """Create basic time features"""
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Create cyclical features if time columns exist
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def handle_missing_and_encode(self, df, target_col='ridership'):
        """Handle missing values and encode categoricals"""
        # Forward fill missing values (time series appropriate)
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'datetime':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Fill any remaining missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def scale_features(self, df, target_col='ridership'):
        """Scale numerical features except target"""
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != target_col]
        
        if feature_cols:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df
    
    def select_top_features(self, df, target_col='ridership', k=15):
        """Select top K correlated features"""
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != target_col]
        
        if feature_cols and target_col in df.columns:
            # Calculate correlations and select top K
            correlations = abs(df[feature_cols].corrwith(df[target_col]))
            top_features = correlations.nlargest(min(k, len(feature_cols))).index.tolist()
            
            # Keep essential columns
            keep_cols = []
            for col in ['datetime', 'zone', 'location']:
                if col in df.columns:
                    keep_cols.append(col)
            keep_cols.extend(top_features)
            keep_cols.append(target_col)
            
            return df[keep_cols]
        
        return df
    
    def clean_data(self, df):
        """Main cleaning function"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # 1. Create time features
        df = self.create_time_features(df)
        
        # 2. Create lag features
        df = self.create_lag_features(df)
        
        # 3. Handle missing values and encode
        df = self.handle_missing_and_encode(df)
        
        # 4. Scale features
        df = self.scale_features(df)
        
        # 5. Select top features
        df = self.select_top_features(df, k=15)
        
        return df

class RidershipPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def prepare_data(self, df, target_col='ridership', test_size=0.2):
        """Prepare data for model training"""
        # Create feature columns list
        exclude_cols = [target_col, 'datetime', 'date', 'time']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        self.feature_columns = feature_cols
        
        # Time-aware split if datetime available
        if 'datetime' in df.columns:
            df_sorted = df.sort_values('datetime')
            split_idx = int(len(df_sorted) * (1 - test_size))
            
            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]
            
            X_train = train_df[feature_cols]
            X_test = test_df[feature_cols]
            y_train = train_df[target_col]
            y_test = test_df[target_col]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        return X_train, X_test, y_train, y_test
    
    def generate_prediction_plot(self, y_test, y_pred):
        """Generate predicted vs actual values scatter plot"""
        plt.figure(figsize=(10, 6))
        
        plt.scatter(y_test, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        max_val = max(max(y_test), max(y_pred))
        min_val = min(min(y_test), min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                label='Perfect Prediction', linewidth=2)
        
        # Calculate metrics for display
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        plt.xlabel('Actual Ridership')
        plt.ylabel('Predicted Ridership')
        plt.title('Ridership Prediction Performance')
        
        # Add metrics text box
        textstr = f'R² Score: {r2:.3f}\nRMSE: {rmse:.1f}\nMAE: {mae:.1f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()
    
    def train_and_evaluate(self, df):
        """Train XGBoost model and evaluate performance"""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            "mae": float(mean_absolute_error(y_train, y_train_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "r2": float(r2_score(y_train, y_train_pred))
        }
        
        test_metrics = {
            "mae": float(mean_absolute_error(y_test, y_test_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "r2": float(r2_score(y_test, y_test_pred))
        }
        
        # Generate visualization
        prediction_plot = self.generate_prediction_plot(y_test, y_test_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create results
        results = {
            "model_info": {
                "model_type": "XGBoost Regressor",
                "training_samples": int(X_train.shape[0]),
                "testing_samples": int(X_test.shape[0]),
                "total_features": int(len(self.feature_columns))
            },
            "performance": {
                "training_metrics": train_metrics,
                "testing_metrics": test_metrics,
                "performance_grade": "Excellent" if test_metrics["r2"] > 0.8 else 
                                  "Good" if test_metrics["r2"] > 0.6 else 
                                  "Fair" if test_metrics["r2"] > 0.4 else "Needs Improvement"
            },
            "top_features": [
                {"feature": str(row.feature), "importance": float(row.importance)}
                for _, row in feature_importance.head(10).iterrows()
            ],
            "visualization": f"data:image/png;base64,{prediction_plot}"
        }
        
        return results

@TransportationDemandForecasting_router.get("/")
async def root():
    """Root endpoint with pipeline information."""
    return {
        "message": "Transportation Data Pipeline API",
        "pipeline_steps": [
            "1. Upload CSV file",
            "2. Generate YData profile (auto-processes uploaded CSV)",
            "3. Clean data (auto-processes uploaded CSV)", 
            "4. Train model (auto-processes cleaned CSV)",
            "5. Make predictions (auto-processes trained model with test CSV)",
            "6. SHAP analysis (auto-processes cleaned CSV with Decision Tree)",
            "7. Partial Dependency Plots (auto-processes cleaned CSV with feature selection)",
            "8. Chat with CSV data (natural language queries on raw CSV)",
            "9. Register and deploy model in MLflow (auto-processes trained model)",
            "10. Detect data drift and monitor in MLflow (compares new data with training data)"
        ],
        "endpoints": {
            "upload": "/upload-csv/",
            "profile": "/profile-data/{pipeline_id}",
            "clean": "/clean-data/{pipeline_id}",
            "train": "/train-model/{pipeline_id}",
            "predict": "/predict-ridership/{pipeline_id}",
            "shap": "/analyze-shap/{pipeline_id}",
            "pdp": "/create-pdp/{pipeline_id}",
            "chat": "/chat/{pipeline_id}",
            "register_model": "/register-model/{pipeline_id}",
            "detect_drift": "/detect-drift/{pipeline_id}",
            "mlflow_status": "/mlflow-status"
        }
    }

@TransportationDemandForecasting_router.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Step 1: Upload CSV file and save it for processing.
    Returns pipeline_id for subsequent operations.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Generate unique ID for this pipeline run
    unique_id = str(uuid.uuid4())
    raw_csv_path = TEMP_DIR / f"raw_data_{unique_id}.csv"
    
    try:
        # Save uploaded raw CSV
        with open(raw_csv_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read and validate CSV
        df = pd.read_csv(raw_csv_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        return {
            "message": "CSV uploaded successfully",
            "pipeline_id": unique_id,
            "filename": file.filename,
            "data_shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": list(df.columns),
            "next_steps": {
                "profile": f"/profile-data/{unique_id}",
                "clean": f"/clean-data/{unique_id}",
                "train": f"/train-model/{unique_id}",
                "predict": f"/predict-ridership/{unique_id}",
                "shap": f"/analyze-shap/{unique_id}",
                "pdp": f"/create-pdp/{unique_id}",
                "chat": f"/chat/{unique_id}"
            }
        }
        
    except Exception as e:
        if raw_csv_path.exists():
            raw_csv_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@TransportationDemandForecasting_router.post("/profile-data/{pipeline_id}")
async def profile_data(pipeline_id: str = FastAPIPath(...)):
    """
    Step 2: Generate YData profiling report from uploaded CSV.
    Automatically processes the uploaded CSV using pipeline_id.
    """
    raw_csv_path = TEMP_DIR / f"raw_data_{pipeline_id}.csv"
    profile_html_path = TEMP_DIR / f"profile_report_{pipeline_id}.html"
    
    if not raw_csv_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Raw data not found for pipeline_id: {pipeline_id}. Please upload CSV first."
        )
    
    try:
        # Load raw data
        df = pd.read_csv(raw_csv_path)
        
        # Generate YData profile
        profile = ProfileReport(
            df, 
            title=f"Data Profile - Pipeline {pipeline_id}", 
            minimal=True,
            interactions=None,
            correlations=None
        )
        profile.to_file(profile_html_path)
        
        return {
            "message": "Data profiling completed successfully",
            "pipeline_id": pipeline_id,
            "data_shape": [int(df.shape[0]), int(df.shape[1])],
            "profile_generated": True,
            "missing_values": int(df.isnull().sum().sum()),
            "numeric_columns": int(len(df.select_dtypes(include=[np.number]).columns)),
            "categorical_columns": int(len(df.select_dtypes(include=['object']).columns)),
            "download_url": f"/download-profile/{pipeline_id}"
        }
        
    except Exception as e:
        if profile_html_path.exists():
            profile_html_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error generating profile: {str(e)}")

@TransportationDemandForecasting_router.post("/clean-data/{pipeline_id}")
async def clean_data(pipeline_id: str = FastAPIPath(...)):
    """
    Step 3: Clean the uploaded raw CSV automatically.
    Automatically processes the uploaded CSV using pipeline_id.
    """
    raw_csv_path = TEMP_DIR / f"raw_data_{pipeline_id}.csv"
    cleaned_csv_path = TEMP_DIR / f"cleaned_data_{pipeline_id}.csv"
    
    if not raw_csv_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Raw data not found for pipeline_id: {pipeline_id}. Please upload CSV first."
        )
    
    try:
        # Load raw data
        df = pd.read_csv(raw_csv_path)
        
        # Initialize cleaner and clean data
        cleaner = SimpleTimeSeriesCleaner()
        df_cleaned = cleaner.clean_data(df)
        
        # Save cleaned data
        df_cleaned.to_csv(cleaned_csv_path, index=False)
        
        # Save the label encoders for later use in training
        encoders_path = TEMP_DIR / f"encoders_{pipeline_id}.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(cleaner.label_encoders, f)
        
        return {
            "message": "Data cleaned successfully",
            "pipeline_id": pipeline_id,
            "original_shape": [int(df.shape[0]), int(df.shape[1])],
            "cleaned_shape": [int(df_cleaned.shape[0]), int(df_cleaned.shape[1])],
            "features_created": list(df_cleaned.columns),
            "cleaning_steps": [
                "Removed duplicates",
                "Created time features (cyclical encoding)",
                "Created lag features",
                "Handled missing values",
                "Encoded categorical variables",
                "Scaled numerical features",
                "Selected top correlated features"
            ],
            "download_url": f"/download-cleaned/{pipeline_id}"
        }
        
    except Exception as e:
        if cleaned_csv_path.exists():
            cleaned_csv_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error cleaning data: {str(e)}")

@TransportationDemandForecasting_router.post("/train-model/{pipeline_id}")
async def train_model(pipeline_id: str = FastAPIPath(...)):
    """
    Step 4: Train XGBoost model on cleaned CSV with 'ridership' as target.
    Automatically processes the cleaned CSV using pipeline_id.
    """
    cleaned_csv_path = TEMP_DIR / f"cleaned_data_{pipeline_id}.csv"
    model_path = TEMP_DIR / f"model_{pipeline_id}.pkl"
    
    if not cleaned_csv_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Cleaned data not found for pipeline_id: {pipeline_id}. Please clean data first."
        )
    
    try:
        # Load cleaned data
        df = pd.read_csv(cleaned_csv_path)
        
        # Check if ridership column exists
        if 'ridership' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column 'ridership' not found. Available columns: {list(df.columns)}"
            )
        
        # Initialize predictor and train model
        predictor = RidershipPredictor()
        results = predictor.train_and_evaluate(df)
        
        # Load the saved label encoders from cleaning step
        encoders_path = TEMP_DIR / f"encoders_{pipeline_id}.pkl"
        label_encoders = {}
        if encoders_path.exists():
            with open(encoders_path, 'rb') as f:
                label_encoders = pickle.load(f)
        
        # Save trained model with proper structure for inference
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': predictor.model,
                'feature_columns': predictor.feature_columns,
                'label_encoders': label_encoders
            }, f)
        
        # Add pipeline info to results
        results["pipeline_id"] = pipeline_id
        results["timestamp"] = datetime.now().isoformat()
        results["target_column"] = "ridership"
        results["download_url"] = f"/download-model/{pipeline_id}"
        
        return JSONResponse(content={
            "message": "Model trained successfully",
            "pipeline_id": pipeline_id,
            "model_results": results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@TransportationDemandForecasting_router.post("/predict-ridership/{pipeline_id}")
async def predict_ridership(
    pipeline_id: str = FastAPIPath(...),
    test_file: UploadFile = File(..., description="CSV file with test data for prediction")
):
    """
    Step 5: Make predictions using trained model on test CSV.
    Automatically loads the trained model using pipeline_id.
    
    Expected CSV columns:
    datetime, zone, location, prev_week_same_period, prev_same_period_ridership,
    prev_day_same_period, ridership_rolling_7, employment_density, population_density,
    transit_accessibility, commercial_floor_area, transit_frequency, walk_score,
    competing_modes, parking_cost, income_level, vehicle_ownership_rate, time_period
    
    Optional: ridership (for evaluation metrics)
    """
    if not test_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Test file must be CSV")
    
    model_path = TEMP_DIR / f"model_{pipeline_id}.pkl"
    
    if not model_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Trained model not found for pipeline_id: {pipeline_id}. Please train model first."
        )
    
    # Generate unique filename for test data
    test_unique_id = str(uuid.uuid4())
    temp_test_path = TEMP_DIR / f"test_data_{test_unique_id}.csv"
    
    try:
        # Save uploaded test file
        with open(temp_test_path, "wb") as buffer:
            content = await test_file.read()
            buffer.write(content)
        
        # Load test data
        try:
            test_df = pd.read_csv(temp_test_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading test CSV file: {str(e)}")
        
        if test_df.empty:
            raise HTTPException(status_code=400, detail="Test CSV file is empty")
        
        # Initialize inference engine
        inference = RidershipInference()
        
        # Load trained model
        load_result = inference.load_model(str(model_path))
        if load_result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Failed to load model: {load_result['message']}")
        
        # Make predictions
        prediction_result = inference.predict(test_df)
        
        if prediction_result["status"] != "success":
            raise HTTPException(status_code=400, detail=f"Prediction failed: {prediction_result['message']}")
        
        # Clean up temporary test file
        temp_test_path.unlink()
        
        # Prepare comprehensive response
        response = {
            "message": "Predictions completed successfully",
            "pipeline_id": pipeline_id,
            "test_filename": test_file.filename,
            "model_info": load_result,
            "prediction_results": prediction_result,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        if temp_test_path.exists():
            temp_test_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")
@TransportationDemandForecasting_router.post("/register-model/{pipeline_id}")
async def register_and_deploy_model(
    pipeline_id: str = FastAPIPath(...),
    model_name: str = Form(..., description="Name for the registered model"),
    description: str = Form("", description="Model description"),
    stage: str = Form("None", description="Model stage: None, Staging, Production, Archived"),
    tags: str = Form("", description="Comma-separated tags for the model")
):
    """
    Step 9: Register and deploy trained model in MLflow.
    Automatically loads the trained model using pipeline_id and registers it in MLflow.
    
    Args:
        pipeline_id: Pipeline ID from training step
        model_name: Name for the registered model in MLflow
        description: Description of the model
        stage: Model stage (None, Staging, Production, Archived)
        tags: Comma-separated tags for the model
    
    Returns:
        Model registration details and MLflow tracking information.
    """
    model_path = TEMP_DIR / f"model_{pipeline_id}.pkl"
    cleaned_csv_path = TEMP_DIR / f"cleaned_data_{pipeline_id}.csv"
    
    if not model_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Trained model not found for pipeline_id: {pipeline_id}. Please train model first."
        )
    
    try:
        # Load the model to get metadata
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        model = model_dict['model']
        feature_columns = model_dict['feature_columns']
        label_encoders = model_dict.get('label_encoders', {})
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        model_tags = {f"tag_{i+1}": tag for i, tag in enumerate(tag_list)}
        
        # Add default tags
        model_tags.update({
            "pipeline_id": pipeline_id,
            "model_type": str(type(model).__name__),
            "framework": "xgboost" if "XGB" in str(type(model).__name__) else "sklearn",
            "created_by": "transportation_pipeline",
            "features_count": str(len(feature_columns))
        })
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"ridership_model_{pipeline_id}"):
            
            # Log model parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Log feature information
            mlflow.log_param("feature_count", len(feature_columns))
            mlflow.log_param("features", str(feature_columns))
            mlflow.log_param("has_label_encoders", len(label_encoders) > 0)
            
            # Load performance metrics if available
            if cleaned_csv_path.exists():
                try:
                    df = pd.read_csv(cleaned_csv_path)
                    if 'ridership' in df.columns:
                        # Quick model evaluation
                        X = df[feature_columns]
                        y = df['ridership']
                        
                        # Simple train-test split for metrics
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Get predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate and log metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)
                        
                        mlflow.log_metric("mae", mae)
                        mlflow.log_metric("rmse", rmse)
                        mlflow.log_metric("r2_score", r2)
                        
                        # Log dataset info
                        mlflow.log_param("training_samples", len(X_train))
                       
                        mlflow.log_param("test_samples", len(X_test))
                
                except Exception as e:
                    logger.warning(f"Could not calculate metrics: {e}")
            
            # Log the model based on type
            if "XGB" in str(type(model).__name__):
                model_info = mlflow.xgboost.log_model(
                    model, 
                    "model",
                    registered_model_name=model_name
                )
            else:
                model_info = mlflow.sklearn.log_model(
                    model, 
                    "model",
                    registered_model_name=model_name
                )
            
            # Log the complete model dictionary as artifact
            mlflow.log_dict(
                {
                    "feature_columns": feature_columns,
                    "label_encoders": {k: list(v.classes_) if hasattr(v, 'classes_') else str(v) 
                                     for k, v in label_encoders.items()},
                    "model_type": str(type(model).__name__)
                },
                "model_metadata.json"
            )
            
            # Save and log the pickle file
            mlflow.log_artifact(str(model_path), "model_files")
            
            # Get run info
            run = mlflow.active_run()
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
        
        # Initialize MLflow client for model management
        client = MlflowClient()
        
        # Get the latest version of the registered model
        try:
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                latest_version = latest_versions[0]
                model_version = latest_version.version
                
                # Update model version description and tags
                if description:
                    client.update_model_version(
                        name=model_name,
                        version=model_version,
                        description=description
                    )
                
                # Set model tags
                for key, value in model_tags.items():
                    client.set_model_version_tag(model_name, model_version, key, value)
                
                # Transition model to specified stage if not "None"
                if stage and stage != "None":
                    client.transition_model_version_stage(
                        name=model_name,
                        version=model_version,
                        stage=stage
                    )
                
                # Get model version details
                model_version_details = client.get_model_version(model_name, model_version)
                
                response = {
                    "message": "Model registered and deployed successfully in MLflow",
                    "pipeline_id": pipeline_id,
                    "mlflow_info": {
                        "tracking_uri": MLFLOW_TRACKING_URI,
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "model_name": model_name,
                        "model_version": model_version,
                        "model_stage": model_version_details.current_stage,
                        "model_uri": model_info.model_uri,
                        "artifact_uri": run.info.artifact_uri
                    },
                    "model_details": {
                        "model_type": str(type(model).__name__),
                        "feature_count": len(feature_columns),
                        "features": feature_columns,
                        "has_label_encoders": len(label_encoders) > 0,
                        "description": description,
                        "tags": model_tags
                    },
                    "deployment_info": {
                        "stage": stage,
                        "deployment_ready": True,
                        "model_loading_code": f"""
# Load model from MLflow
import mlflow.pyfunc
model = mlflow.pyfunc.load_model('models:/{model_name}/{model_version}')

# Make predictions
predictions = model.predict(your_data)
""",
                        "curl_command": f"""
# Test model via MLflow REST API
curl -X POST http://127.0.0.1:5000/invocations \\
  -H 'Content-Type: application/json' \\
  -d '{{"inputs": [[your_feature_values]]}}'
"""
                    },
                    "links": {
                        "mlflow_ui": f"{MLFLOW_TRACKING_URI}",
                        "experiment_url": f"{MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}",
                        "run_url": f"{MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{run_id}",
                        "model_url": f"{MLFLOW_TRACKING_URI}/#/models/{model_name}"
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                return JSONResponse(content=response)
            
            else:
                raise HTTPException(status_code=500, detail="Model registration completed but version not found")
                
        except Exception as e:
            logger.error(f"Error in model registration post-processing: {e}")
            raise HTTPException(status_code=500, detail=f"Model registered but post-processing failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering model in MLflow: {e}")
        raise HTTPException(status_code=500, detail=f"Error registering model: {str(e)}")
    
@TransportationDemandForecasting_router.post("/detect-drift/{pipeline_id}")
async def detect_data_drift(
    pipeline_id: str = FastAPIPath(...),
    new_data_file: UploadFile = File(..., description="New CSV data to compare against training data"),
    drift_threshold: float = Form(0.05, description="P-value threshold for drift detection (default: 0.05)"),
    record_in_mlflow: bool = Form(True, description="Whether to record drift results in MLflow")
):
    """
    Step 10: Detect data drift by comparing new data with training data.
    Records drift analysis results in MLflow for monitoring.
    
    Args:
        pipeline_id: Pipeline ID from training step
        new_data_file: New CSV data to compare against reference data
        drift_threshold: Statistical significance threshold for drift detection
        record_in_mlflow: Whether to log drift results in MLflow
    
    Returns:
        Comprehensive drift analysis with statistical tests and MLflow tracking.
    """
    if not new_data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="New data file must be CSV")
    
    cleaned_csv_path = TEMP_DIR / f"cleaned_data_{pipeline_id}.csv"
    
    if not cleaned_csv_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Reference data not found for pipeline_id: {pipeline_id}. Please clean data first."
        )
    
    # Generate unique filename for new data
    drift_unique_id = str(uuid.uuid4())
    temp_new_data_path = TEMP_DIR / f"drift_data_{drift_unique_id}.csv"
    
    try:
        # Save uploaded new data file
        with open(temp_new_data_path, "wb") as buffer:
            content = await new_data_file.read()
            buffer.write(content)
        
        # Load reference data (cleaned training data)
        try:
            reference_df = pd.read_csv(cleaned_csv_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading reference data: {str(e)}")
        
        # Load new data
        try:
            new_df = pd.read_csv(temp_new_data_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading new data CSV: {str(e)}")
        
        if new_df.empty:
            raise HTTPException(status_code=400, detail="New data CSV is empty")
        
        # Initialize drift analyzer
        drift_analyzer = DataDriftAnalyzer()
        drift_analyzer.drift_threshold = drift_threshold
        
        # Set reference data
        ref_setup = drift_analyzer.set_reference_data(reference_df, target_col='ridership')
        if ref_setup["status"] != "success":
            raise HTTPException(status_code=500, detail="Failed to set reference data")
        
        # Detect drift
        drift_results = drift_analyzer.detect_drift(new_df, target_col='ridership')
        
        if drift_results["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Drift detection failed: {drift_results['message']}")
        
        # Record in MLflow if requested
        mlflow_info = None
        if record_in_mlflow:
            try:
                # Start MLflow run for drift monitoring
                with mlflow.start_run(run_name=f"drift_detection_{pipeline_id}_{drift_unique_id}"):
                    
                    # Log drift parameters
                    mlflow.log_param("pipeline_id", pipeline_id)
                    mlflow.log_param("drift_threshold", drift_threshold)
                    mlflow.log_param("reference_data_size", len(reference_df))
                    mlflow.log_param("new_data_size", len(new_df))
                    mlflow.log_param("new_data_filename", new_data_file.filename)
                    
                    # Log drift metrics
                    mlflow.log_metric("total_features_tested", drift_results["summary"]["total_features_tested"])
                    mlflow.log_metric("features_with_drift", drift_results["summary"]["features_with_drift"])
                    mlflow.log_metric("drift_percentage", drift_results["summary"]["drift_percentage"])
                    mlflow.log_metric("overall_drift_detected", 1 if drift_results["summary"]["overall_drift_detected"] else 0)
                    
                    # Log individual feature drift p-values
                    for feature_result in drift_results["feature_drift_results"]:
                        mlflow.log_metric(f"drift_pvalue_{feature_result['feature']}", feature_result["p_value"])
                        mlflow.log_metric(f"drift_detected_{feature_result['feature']}", 1 if feature_result["drift_detected"] else 0)
                    
                    # Log target drift if available
                    if drift_results["target_drift_result"]:
                        mlflow.log_metric("target_drift_pvalue", drift_results["target_drift_result"]["p_value"])
                        mlflow.log_metric("target_drift_detected", 1 if drift_results["target_drift_result"]["drift_detected"] else 0)
                    
                    # Log data quality metrics
                    if drift_results["data_quality_analysis"]:
                        quality = drift_results["data_quality_analysis"]
                        mlflow.log_metric("reference_missing_pct", quality["reference_stats"]["missing_percentage"])
                        mlflow.log_metric("new_data_missing_pct", quality["new_data_stats"]["missing_percentage"])
                    
                    # Log drift results as JSON artifact
                    mlflow.log_dict(drift_results, "drift_analysis_results.json")
                    
                    # Log new data as artifact
                    mlflow.log_artifact(str(temp_new_data_path), "drift_monitoring")
                    
                    # Add tags for easy filtering
                    mlflow.set_tags({
                        "type": "drift_detection",
                        "pipeline_id": pipeline_id,
                        "drift_detected": str(drift_results["summary"]["overall_drift_detected"]),
                        "drift_severity": drift_results["summary"].get("overall_severity", "Unknown"),
                        "monitoring_type": "data_drift"
                    })
                    
                    # Get run info
                    run = mlflow.active_run()
                    run_id = run.info.run_id
                    experiment_id = run.info.experiment_id
                    
                    mlflow_info = {
                        "tracking_uri": MLFLOW_TRACKING_URI,
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "run_url": f"{MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{run_id}",
                        "logged": True
                    }
                    
                    logger.info(f"Drift detection results logged to MLflow run: {run_id}")
                    
            except Exception as e:
                logger.error(f"Failed to log drift results to MLflow: {e}")
                mlflow_info = {
                    "logged": False,
                    "error": str(e)
                }
        
        # Clean up temporary file
        temp_new_data_path.unlink()
        
        # Prepare comprehensive response
        response = {
            "message": "Data drift detection completed successfully",
            "pipeline_id": str(pipeline_id),
            "drift_analysis": drift_results,  # Already converted by convert_numpy_types
            "mlflow_tracking": mlflow_info,
            "data_comparison": {
                "reference_data_shape": [int(reference_df.shape[0]), int(reference_df.shape[1])],
                "new_data_shape": [int(new_df.shape[0]), int(new_df.shape[1])],
                "new_data_filename": str(new_data_file.filename)
            },
            "recommendations": {
                "action_required": bool(drift_results["summary"]["overall_drift_detected"]),
                "recommended_actions": []
            },
            "timestamp": datetime.now().isoformat()
        }
                
        # Add recommendations based on drift severity
        if drift_results["summary"]["overall_drift_detected"]:
            severity = drift_results["summary"].get("overall_severity", "Medium")
            if severity == "High":
                response["recommendations"]["recommended_actions"] = [
                    "Immediate attention required - significant data drift detected",
                    "Consider retraining the model with recent data",
                    "Review data collection processes",
                    "Implement continuous monitoring"
                ]
            elif severity == "Medium":
                response["recommendations"]["recommended_actions"] = [
                    "Monitor closely - moderate drift detected",
                    "Plan for model retraining in near future",
                    "Investigate root causes of drift"
                ]
            else:
                response["recommendations"]["recommended_actions"] = [
                    "Low-level drift detected - continue monitoring",
                    "No immediate action required"
                ]
        else:
            response["recommendations"]["recommended_actions"] = [
                "No significant drift detected",
                "Current model should perform well on new data",
                "Continue regular monitoring"
            ]

        # Final conversion to ensure everything is JSON serializable
        response = convert_numpy_types(response)

        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        if temp_new_data_path.exists():
            temp_new_data_path.unlink()
        
        logger.error(f"Error in drift detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting drift: {str(e)}")
    
@TransportationDemandForecasting_router.get("/mlflow-status")
async def check_mlflow_status():
    """Check MLflow server connectivity and list registered models."""
    try:
        client = MlflowClient()
        
        # Test connection by listing experiments
        experiments = client.search_experiments()
        
        # List registered models
        registered_models = client.search_registered_models()
        
        return {
            "status": "connected",
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "experiments_count": len(experiments),
            "registered_models_count": len(registered_models),
            "registered_models": [
                {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description
                }
                for model in registered_models
            ],
            "message": "MLflow server is accessible"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "error": str(e),
            "message": "Cannot connect to MLflow server. Make sure 'mlflow ui' is running."
        }    

@TransportationDemandForecasting_router.post("/analyze-shap/{pipeline_id}")
async def analyze_shap(pipeline_id: str = FastAPIPath(...)):
    """
    Step 6: Generate SHAP bee swarm plot analysis.
    Automatically processes the cleaned CSV using pipeline_id with Decision Tree model.
    
    Returns SHAP bee swarm plot as base64 image showing feature importance
    and impact on ridership predictions.
    """
    cleaned_csv_path = TEMP_DIR / f"cleaned_data_{pipeline_id}.csv"
    
    if not cleaned_csv_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Cleaned data not found for pipeline_id: {pipeline_id}. Please clean data first."
        )
    
    try:
        # Load cleaned data
        df = pd.read_csv(cleaned_csv_path)
        
        # Check if ridership column exists
        if 'ridership' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column 'ridership' not found. Available columns: {list(df.columns)}"
            )
        
        # Initialize SHAP analyzer
        analyzer = SHAPAnalyzer()
        
        # Train Decision Tree model for SHAP analysis
        train_result = analyzer.train_decision_tree(df, target_col='ridership')
        if train_result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Failed to train model: {train_result['message']}")
        
        # Perform comprehensive SHAP analysis
        shap_result = analyzer.create_comprehensive_analysis(max_display=15)
        if shap_result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"SHAP analysis failed: {shap_result['message']}")
        
        # Prepare comprehensive response
        response = {
            "message": "SHAP analysis completed successfully",
            "pipeline_id": pipeline_id,
            "model_training": train_result,
            "shap_analysis": shap_result,
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": {
                "model_type": "Decision Tree Regressor (for SHAP)",
                "samples_analyzed": shap_result["analysis_info"]["explained_samples"],
                "features_analyzed": shap_result["analysis_info"]["features_analyzed"],
                "top_feature": shap_result["feature_analysis"]["feature_importance"][0]["feature"] if shap_result["feature_analysis"]["feature_importance"] else "N/A"
            }
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SHAP analysis: {str(e)}")

@TransportationDemandForecasting_router.post("/create-pdp/{pipeline_id}")
async def create_pdp(
    pipeline_id: str = FastAPIPath(...),
    features: str = Form(..., description="Comma-separated list of features to analyze"),
    model_type: str = Form("decision_tree", description="Model type: 'decision_tree' or 'random_forest'"),
    plot_type: str = Form("auto", description="Plot type: 'single', 'multiple', or 'auto'"),
    grid_resolution: int = Form(100, description="Grid resolution for PDP computation")
):
    """
    Step 7: Create Partial Dependency Plots (PDP) for specified features.
    Automatically processes the cleaned CSV using pipeline_id.
    
    Args:
        pipeline_id: Pipeline ID from previous steps
        features: Comma-separated list of feature names to analyze
        model_type: Type of model to use ('decision_tree' or 'random_forest')
        plot_type: Type of plot ('single', 'multiple', or 'auto')
        grid_resolution: Number of points in the PDP grid
    
    Returns:
        PDP plots as base64 images with feature effect analysis.
    
    Example features from cleaned data:
    prev_same_period_ridership,ridership_rolling_7,employment_density,transit_accessibility
    """
    cleaned_csv_path = TEMP_DIR / f"cleaned_data_{pipeline_id}.csv"
    
    if not cleaned_csv_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Cleaned data not found for pipeline_id: {pipeline_id}. Please clean data first."
        )
    
    try:
        # Parse features
        feature_list = [f.strip() for f in features.split(',') if f.strip()]
        
        if not feature_list:
            raise HTTPException(status_code=400, detail="No features provided")
        
        if len(feature_list) > 9:
            raise HTTPException(status_code=400, detail="Maximum 9 features allowed")
        
        # Validate model type
        if model_type not in ['decision_tree', 'random_forest']:
            raise HTTPException(status_code=400, detail="Model type must be 'decision_tree' or 'random_forest'")
        
        # Load cleaned data
        df = pd.read_csv(cleaned_csv_path)
        
        # Check if ridership column exists
        if 'ridership' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column 'ridership' not found. Available columns: {list(df.columns)}"
            )
        
        # Initialize PDP analyzer
        analyzer = PartialDependencyAnalyzer(model_type=model_type)
        
        # Train model for PDP analysis
        train_result = analyzer.train_model(df, target_col='ridership', model_type=model_type)
        if train_result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Failed to train model: {train_result['message']}")
        
        # Validate requested features
        validation = analyzer.validate_features(feature_list)
        if validation["status"] == "error":
            raise HTTPException(status_code=400, detail=validation["message"])
        
        # Get valid features
        valid_features = validation["valid_features"]
        if not valid_features:
            available_features = validation.get("available_features", [])
            raise HTTPException(
                status_code=400, 
                detail=f"None of the requested features found. Available features: {available_features}"
            )
        
        # Determine plot type if auto
        if plot_type == "auto":
            if len(valid_features) == 1:
                plot_type = "single"
            else:
                plot_type = "multiple"
        
        # Create PDP plots
        if plot_type == "single" and len(valid_features) == 1:
            plot_base64 = analyzer.create_single_feature_pdp(
                valid_features[0], 
                grid_resolution=grid_resolution
            )
        elif plot_type == "multiple" or len(valid_features) > 1:
            plot_base64 = analyzer.create_multiple_pdp(
                valid_features[:9],  # Limit to 9 features
                grid_resolution=grid_resolution
            )
        else:
            plot_base64 = analyzer.create_single_feature_pdp(
                valid_features[0], 
                grid_resolution=grid_resolution
            )
        
        # Analyze feature effects
        effects_analysis = analyzer.analyze_feature_effects(valid_features, grid_resolution=grid_resolution)
        
        # Prepare comprehensive response
        response = {
            "message": "Partial Dependency Plot analysis completed successfully",
            "pipeline_id": pipeline_id,
            "model_training": train_result,
            "pdp_analysis": {
                "status": "success",
                "plot_data": f"data:image/png;base64,{plot_base64}",
                "plot_info": {
                    "plot_type": plot_type,
                    "features_requested": feature_list,
                    "features_plotted": valid_features,
                    "model_used": model_type,
                    "grid_resolution": grid_resolution
                },
                "feature_effects": effects_analysis,
                "validation_info": {
                    "valid_features": valid_features,
                    "missing_features": validation.get("missing_features", []),
                    "warnings": [validation["message"]] if validation["status"] == "warning" else []
                }
            },
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": {
                "model_type": f"{model_type.replace('_', ' ').title()} (for PDP)",
                "features_analyzed": len(valid_features),
                "most_impactful_feature": effects_analysis.get("analysis_summary", {}).get("most_impactful_feature", "N/A"),
                "plot_description": f"{plot_type.title()} PDP plot with {len(valid_features)} feature(s)"
            }
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating PDP analysis: {str(e)}")

@TransportationDemandForecasting_router.post("/chat/{pipeline_id}")
async def chat_with_csv(
    pipeline_id: str = FastAPIPath(...),
    query: str = Form(..., description="Natural language query about the CSV data")
):
    """
    Step 8: Chat with CSV data using natural language.
    Automatically processes the raw CSV using pipeline_id and enables AI-powered querying.
    
    Args:
        pipeline_id: Pipeline ID from upload step
        query: Natural language question about the data
    
    Returns:
        AI-generated answer with reference data and context.
    
    Example queries:
    - "What is the average ridership by zone?"
    - "Show me the highest ridership periods"
    - "Which factors correlate with high ridership?"
    - "Summarize the employment density data"
    """
    raw_csv_path = TEMP_DIR / f"raw_data_{pipeline_id}.csv"
    
    if not raw_csv_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Raw data not found for pipeline_id: {pipeline_id}. Please upload CSV first."
        )
    
    try:
        # Load raw CSV data
        df = pd.read_csv(raw_csv_path)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV data is empty")
        
        # Create collection name for this pipeline
        collection_name = f"{CHAT_COLLECTION_PREFIX}{pipeline_id}"
        
        # Check if collection already exists, if not create it
        existing_collections = [col.name for col in chroma_client.list_collections()]
        
        if collection_name not in existing_collections:
            logger.info(f"Creating new ChromaDB collection for pipeline {pipeline_id}")
            
            # Process CSV data into documents
            documents = process_csv_data(df, chunk_size=200)
            
            # Create collection
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef
            )
            
            # Add documents to collection
            texts = [doc["text"] for doc in documents]
            ids = [doc["id"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Processed {len(documents)} documents into collection {collection_name}")
        else:
            # Get existing collection
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef
            )
            logger.info(f"Using existing collection {collection_name}")
        
        # Query for relevant documents
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        if not results['documents'] or not results['documents'][0]:
            raise HTTPException(status_code=404, detail="No relevant data found for your query.")
        
        # Fast context truncation using character limits
        relevant_docs = results['documents'][0]
        context = truncate_context_fast(relevant_docs, max_chars=30000)  # ~7500 tokens
        
        # Parse reference data for frontend tabulation
        structured_reference = parse_reference_data(relevant_docs)
        
        # Generate response using Groq
        client = get_groq_client()
        
        prompt = f"""You are a transport demand forecasting expert that helps analyze and predict ridership data. Based on the following context from a transportation CSV file, answer the user's question accurately and helpfully.

Focus on providing insights about:
- Ridership patterns and trends
- Factors affecting transportation demand
- Zone-based analysis
- Time-based patterns
- Correlations between variables

Context from CSV:
{context}

User Question: {query}

Please provide a clear, accurate answer based on the transportation data provided. If you need to make calculations or summarize data, please do so. Focus on actionable insights for transportation planning and demand forecasting. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        answer = completion.choices[0].message.content
        
        # Prepare comprehensive response
        response = {
            "message": "Chat query processed successfully",
            "pipeline_id": pipeline_id,
            "query": query,
            "chat_response": {
                "answer": answer,
                "reference_data": structured_reference,  # Structured for easy tabulation
                "raw_context": relevant_docs[:3]  # Raw context for debugging
            },
            "data_info": {
                "csv_shape": [int(df.shape[0]), int(df.shape[1])],
                "columns": list(df.columns),
                "collection_name": collection_name,
                "documents_searched": len(relevant_docs)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat query: {str(e)}")

@TransportationDemandForecasting_router.get("/chat-status/{pipeline_id}")
async def get_chat_status(pipeline_id: str = FastAPIPath(...)):
    """Check if chat data is available for a pipeline."""
    try:
        collection_name = f"{CHAT_COLLECTION_PREFIX}{pipeline_id}"
        existing_collections = [col.name for col in chroma_client.list_collections()]
        
        has_chat_data = collection_name in existing_collections
        
        if has_chat_data:
            collection = chroma_client.get_collection(name=collection_name)
            doc_count = collection.count()
        else:
            doc_count = 0
        
        return {
            "pipeline_id": pipeline_id,
            "has_chat_data": has_chat_data,
            "collection_name": collection_name if has_chat_data else None,
            "document_count": doc_count,
            "message": "Chat ready" if has_chat_data else "Upload CSV first to enable chat"
        }
    except Exception as e:
        logger.error(f"Error checking chat status: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking chat status: {str(e)}")

@TransportationDemandForecasting_router.delete("/clear-chat/{pipeline_id}")
async def clear_chat_data(pipeline_id: str = FastAPIPath(...)):
    """Clear chat data for a specific pipeline."""
    try:
        collection_name = f"{CHAT_COLLECTION_PREFIX}{pipeline_id}"
        existing_collections = [col.name for col in chroma_client.list_collections()]
        
        if collection_name in existing_collections:
            chroma_client.delete_collection(name=collection_name)
            return {
                "message": f"Chat data cleared for pipeline {pipeline_id}",
                "collection_cleared": collection_name
            }
        else:
            return {
                "message": f"No chat data found for pipeline {pipeline_id}",
                "collection_cleared": None
            }
    except Exception as e:
        logger.error(f"Error clearing chat data: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing chat data: {str(e)}")

@TransportationDemandForecasting_router.get("/download-profile/{pipeline_id}")
async def download_profile(pipeline_id: str):
    """Download the YData profiling report."""
    profile_path = TEMP_DIR / f"profile_report_{pipeline_id}.html"
    
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Profile report not found")
    
    return FileResponse(
        path=profile_path,
        filename=f"data_profile_{pipeline_id}.html",
        media_type="text/html"
    )

@TransportationDemandForecasting_router.get("/download-cleaned/{pipeline_id}")
async def download_cleaned(pipeline_id: str):
    """Download the cleaned CSV data."""
    cleaned_path = TEMP_DIR / f"cleaned_data_{pipeline_id}.csv"
    
    if not cleaned_path.exists():
        raise HTTPException(status_code=404, detail="Cleaned data not found")
    
    return FileResponse(
        path=cleaned_path,
        filename=f"cleaned_data_{pipeline_id}.csv",
        media_type="text/csv"
    )

@TransportationDemandForecasting_router.get("/download-model/{pipeline_id}")
async def download_model(pipeline_id: str):
    """Download the trained model."""
    model_path = TEMP_DIR / f"model_{pipeline_id}.pkl"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Trained model not found")
    
    return FileResponse(
        path=model_path,
        filename=f"ridership_model_{pipeline_id}.pkl",
        media_type="application/octet-stream"
    )
