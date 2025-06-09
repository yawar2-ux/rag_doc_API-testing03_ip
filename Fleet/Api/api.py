#enter your all endpoints here
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd  # Add this import
import uuid
import json
import pickle
from datetime import datetime
import logging
from pydantic import BaseModel
from typing import List

# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ydata-profiling requires 'pkg_resources', which is part of 'setuptools'.
# Ensure 'setuptools' is installed in your environment (e.g., pip install setuptools).
from ydata_profiling import ProfileReport # Moved import to top level

# Import service modules
from Fleet.Services.csv_chatbot import csv_chatbot
from Fleet.Services.y_data_profiler import lifespan_cleanup
from Fleet.Services.data_cleaner import DataCleaner, CleaningConfig
from Fleet.Services.model_trainer_and_predict import FleetMaintenanceModel, TrainingConfig
from Fleet.Services.shap_creator import SHAPAnalyzer, SHAPConfig
from Fleet.Services.cfs import FleetMaintenanceExplainer, CounterfactualConfig
from Fleet.Services.pdp_creator import PDPAnalyzer, PDPConfig

# MLflow configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"  # Default MLflow UI URL
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

warnings.filterwarnings('ignore')

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    reference_data: List[dict]
    raw_context: List[str]

class UploadResponse(BaseModel):
    message: str
    documents_processed: int
    rows: int
    columns: int
    filename: str

# Main temporary directory - all subdirectories will be created under this
TEMP_BASE_DIR = Path("temp_fleet_api")
TEMP_BASE_DIR.mkdir(exist_ok=True)

# Subdirectories for different services
TEMP_DIR = TEMP_BASE_DIR / "reports"
TEMP_CLEANING_DIR = TEMP_BASE_DIR / "cleaning"
TEMP_MODELS_DIR = TEMP_BASE_DIR / "models"
TEMP_SHAP_DIR = TEMP_BASE_DIR / "shap"
TEMP_CFS_DIR = TEMP_BASE_DIR / "counterfactuals"
TEMP_PDP_DIR = TEMP_BASE_DIR / "pdp"

# Create all subdirectories
for temp_dir in [TEMP_DIR, TEMP_CLEANING_DIR, TEMP_MODELS_DIR, TEMP_SHAP_DIR, TEMP_CFS_DIR, TEMP_PDP_DIR]:
    temp_dir.mkdir(exist_ok=True)

# Utility function for numpy type conversion with enhanced float handling
def convert_numpy_types(obj):
    """Convert numpy data types to Python native types for JSON serialization with comprehensive float handling."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle inf and nan values more robustly
        if np.isinf(obj) or np.isnan(obj) or abs(obj) > 1e308:
            return None
        else:
            # Ensure the float is within JSON-safe range
            return float(np.clip(obj, -1e308, 1e308))
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, (float, np.float64, np.float32)):
        # Handle Python float inf/nan values and very large numbers
        if np.isinf(obj) or np.isnan(obj) or abs(obj) > 1e308:
            return None
        else:
            return float(np.clip(obj, -1e308, 1e308))
    elif isinstance(obj, (int, np.int64, np.int32)):
        # Handle very large integers
        try:
            return int(np.clip(obj, -2**53, 2**53))
        except (OverflowError, ValueError):
            return None
    else:
        return obj

def validate_json_serializable(obj):
    """Validate that an object can be JSON serialized and fix common issues."""
    try:
        # First convert numpy types
        cleaned_obj = convert_numpy_types(obj)
        # Test serialization
        json.dumps(cleaned_obj)
        return cleaned_obj
    except (TypeError, ValueError, OverflowError) as e:
        logger.warning(f"JSON serialization issue: {e}. Attempting to fix...")
        # Fallback: recursively clean problematic values
        return deep_clean_for_json(cleaned_obj)

def deep_clean_for_json(obj):
    """Deep clean object for JSON compatibility."""
    if isinstance(obj, dict):
        return {str(key): deep_clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [deep_clean_for_json(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        if np.isinf(obj) or np.isnan(obj) or abs(obj) > 1e308:
            return 0.0
        return float(np.clip(obj, -1e308, 1e308))
    elif isinstance(obj, (int, np.integer)):
        try:
            return int(np.clip(obj, -2**53, 2**53))
        except (OverflowError, ValueError):
            return 0
    elif obj is None:
        return None
    else:
        try:
            # Try to convert to string as last resort
            return str(obj)
        except:
            return "undefined"

# Data Drift Analyzer for Fleet Management
class FleetDataDriftAnalyzer:
    """Data drift detection analyzer for fleet maintenance data."""
    
    def __init__(self):
        self.reference_data = None
        self.reference_classification_target = None
        self.reference_regression_target = None
        self.feature_columns = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.drift_threshold = 0.05
        
    def set_reference_data(self, df: pd.DataFrame, 
                          classification_target: str = 'component_at_risk',
                          regression_target: str = 'days_till_breakdown'):
        """Set reference data (usually training data) for drift comparison."""
        self.reference_data = df.copy()
        self.reference_classification_target = df[classification_target] if classification_target in df.columns else None
        self.reference_regression_target = df[regression_target] if regression_target in df.columns else None
        
        # Identify feature columns
        exclude_cols = [classification_target, regression_target, 'vehicle_id', 'reading_date']
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
        """Detect drift in numerical features using Kolmogorov-Smirnov test with robust calculations."""
        try:
            ref_clean = ref_data.dropna()
            new_clean = new_data.dropna()
            
            if len(ref_clean) == 0 or len(new_clean) == 0:
                return {
                    "feature": str(feature_name),
                    "drift_detected": False,
                    "p_value": 1.0,
                    "test_statistic": 0.0,
                    "test_type": "KS",
                    "warning": "Insufficient data for testing"
                }
            
            ks_statistic, p_value = stats.ks_2samp(ref_clean, new_clean)
            
            # Robust statistics calculation
            ref_mean = float(ref_clean.mean()) if not np.isnan(ref_clean.mean()) else 0.0
            ref_std = float(ref_clean.std()) if not np.isnan(ref_clean.std()) else 1.0
            new_mean = float(new_clean.mean()) if not np.isnan(new_clean.mean()) else 0.0
            new_std = float(new_clean.std()) if not np.isnan(new_clean.std()) else 1.0
            
            # Ensure minimum std to avoid division by near-zero
            ref_std_safe = max(abs(ref_std), 1e-6)
            mean_shift = abs(new_mean - ref_mean) / ref_std_safe
            std_ratio = abs(new_std) / ref_std_safe
            
            # Clip all values to safe ranges
            ks_statistic = float(np.clip(ks_statistic, 0.0, 1.0))
            p_value = float(np.clip(p_value, 0.0, 1.0))
            mean_shift = float(np.clip(mean_shift, 0.0, 1000.0))
            std_ratio = float(np.clip(std_ratio, 0.001, 1000.0))
            
            result = {
                "feature": str(feature_name),
                "drift_detected": bool(p_value < self.drift_threshold),
                "p_value": p_value,
                "test_statistic": ks_statistic,
                "test_type": "Kolmogorov-Smirnov",
                "statistics": {
                    "reference_mean": ref_mean,
                    "reference_std": ref_std,
                    "new_mean": new_mean,
                    "new_std": new_std,
                    "mean_shift": mean_shift,
                    "std_ratio": std_ratio
                },
                "drift_severity": "High" if p_value < 0.01 else "Medium" if p_value < 0.05 else "Low"
            }
            
            return validate_json_serializable(result)
            
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
        """Detect drift in categorical features using Chi-square test with robust calculations."""
        try:
            ref_counts = ref_data.value_counts()
            new_counts = new_data.value_counts()
            
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
            
            ref_freq = []
            new_freq = []
            
            for category in all_categories:
                ref_freq.append(max(ref_counts.get(category, 0), 1))
                new_freq.append(max(new_counts.get(category, 0), 1))
            
            try:
                chi2_statistic, p_value = stats.chisquare(new_freq, ref_freq)
                
                # Robust handling of chi-square results
                if np.isinf(chi2_statistic) or np.isnan(chi2_statistic):
                    chi2_statistic = 0.0
                if np.isinf(p_value) or np.isnan(p_value):
                    p_value = 1.0
                    
                chi2_statistic = float(np.clip(chi2_statistic, 0.0, 1e6))
                p_value = float(np.clip(p_value, 0.0, 1.0))
                
            except Exception:
                chi2_statistic = 0.0
                p_value = 1.0
            
            result = {
                "feature": str(feature_name),
                "drift_detected": bool(p_value < self.drift_threshold),
                "p_value": p_value,
                "test_statistic": chi2_statistic,
                "test_type": "Chi-square",
                "statistics": {
                    "reference_categories": int(len(ref_counts)),
                    "new_categories": int(len(new_counts)),
                    "common_categories": int(len(set(ref_counts.index) & set(new_counts.index)))
                },
                "drift_severity": "High" if p_value < 0.01 else "Medium" if p_value < 0.05 else "Low"
            }
            
            return validate_json_serializable(result)
            
        except Exception as e:
            return {
                "feature": str(feature_name),
                "drift_detected": False,
                "p_value": 1.0,
                "test_statistic": 0.0,
                "test_type": "Chi-square",
                "error": str(e)
            }
    
    def detect_drift(self, new_data: pd.DataFrame, 
                    classification_target: str = 'component_at_risk',
                    regression_target: str = 'days_till_breakdown'):
        """Main drift detection function with enhanced error handling."""
        if self.reference_data is None:
            return {
                "status": "error",
                "message": "Reference data not set. Please set reference data first."
            }
        
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
            "classification_target_drift": None,
            "regression_target_drift": None,
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
        
        # Classification target drift detection
        if classification_target in new_data.columns and self.reference_classification_target is not None:
            target_drift = self.detect_categorical_drift(
                self.reference_classification_target, 
                new_data[classification_target], 
                classification_target
            )
            drift_results["classification_target_drift"] = target_drift
        
        # Regression target drift detection
        if regression_target in new_data.columns and self.reference_regression_target is not None:
            target_drift = self.detect_numerical_drift(
                self.reference_regression_target, 
                new_data[regression_target], 
                regression_target
            )
            drift_results["regression_target_drift"] = target_drift
        
        # Summary statistics
        total_features = len(drift_results["feature_drift_results"])
        drift_results["summary"]["total_features_tested"] = int(total_features)
        drift_results["summary"]["features_with_drift"] = int(features_with_drift)
        drift_results["summary"]["drift_percentage"] = float((features_with_drift / total_features * 100) if total_features > 0 else 0.0)
        
        # Overall drift detection
        target_drift_detected = (
            (drift_results["classification_target_drift"] and drift_results["classification_target_drift"]["drift_detected"]) or
            (drift_results["regression_target_drift"] and drift_results["regression_target_drift"]["drift_detected"])
        )
        
        drift_results["summary"]["overall_drift_detected"] = bool(features_with_drift > 0 or target_drift_detected)
        
        # Drift severity assessment
        high_drift_features = [r for r in drift_results["feature_drift_results"] if r.get("drift_severity") == "High"]
        medium_drift_features = [r for r in drift_results["feature_drift_results"] if r.get("drift_severity") == "Medium"]

        if len(high_drift_features) > 0:
            drift_results["summary"]["overall_severity"] = "High"
        elif len(medium_drift_features) > 0:
            drift_results["summary"]["overall_severity"] = "Medium"
        else:
            drift_results["summary"]["overall_severity"] = "Low"

        # Convert all results to JSON-safe format
        drift_results = validate_json_serializable(drift_results)

        return drift_results

# Create router
Fleet_router = APIRouter()

@Fleet_router.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Fleet Management API with MLflow Integration",
        "version": "2.1.0",
        "endpoints": {
            "upload": "/profile-csv/",
            "model_registry": "/register-model/",
            "drift_detection": "/detect-drift/",
            "mlflow_status": "/mlflow-status",
            "docs": "/docs"
        }
    }

@Fleet_router.post("/profile-csv/")
async def create_profile_report(
    file: UploadFile = File(..., description="CSV file to profile"),
    title: Optional[str] = Form(None, description="Custom title for the report"),
    minimal: bool = Form(False, description="Generate minimal report for faster processing")
):
    """
    Upload a CSV file and get a downloadable YData Profiling report.
    
    - **file**: CSV file to analyze
    - **title**: Optional custom title for the report
    - **minimal**: Generate minimal report for large datasets (faster but less detailed)
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_DIR / f"temp_{unique_id}.csv"
    output_html_path = TEMP_DIR / f"profile_report_{unique_id}.html"
    
    try:
        # Save uploaded file temporarily
        with open(temp_csv_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read CSV with pandas
        try:
            df = pd.read_csv(temp_csv_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Validate that CSV has data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Set report title
        report_title = title if title else f"Profiling Report - {file.filename}"
        
        # Configure profiling based on minimal flag
        if minimal:
            # Minimal configuration for faster processing
            profile = ProfileReport(
                df, 
                title=report_title,
                minimal=True,
                interactions=None,
                correlations=None,
                missing_diagrams=None
            )
        else:
            # Full profiling report
            profile = ProfileReport(df, title=report_title)
        
        # Generate HTML report
        profile.to_file(output_html_path)
        
        # Clean up temporary CSV file
        temp_csv_path.unlink()
        
        # Return the HTML file as download
        return FileResponse(
            path=output_html_path,
            filename=f"ydata_profile_report_{file.filename.replace('.csv', '')}.html",
            media_type="text/html",
            headers={"Content-Disposition": "attachment"}
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up files in case of error
        if temp_csv_path.exists():
            temp_csv_path.unlink()
        if output_html_path.exists():
            output_html_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error generating profile report: {str(e)}")

@Fleet_router.post("/clean-dataset/")
async def clean_dataset(
    file: UploadFile = File(..., description="CSV file to clean"),
    target_columns: Optional[str] = Form(None, description="Comma-separated target column names"),
    high_cardinality_threshold: int = Form(10, description="Threshold for high cardinality categorical columns"),
    k_best_features: int = Form(15, description="Number of best features to select (0 to disable)"),
    standardize: bool = Form(True, description="Apply standardization to numerical features"),
    remove_duplicates: bool = Form(True, description="Remove duplicate rows"),
    remove_datetime: bool = Form(True, description="Remove datetime columns"),
    remove_constant: bool = Form(True, description="Remove constant columns"),
    remove_high_cardinality: bool = Form(True, description="Remove high cardinality categorical columns")
):
    """Upload a CSV file and get back a cleaned dataset with statistics."""
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_CLEANING_DIR / f"temp_{unique_id}.csv"
    output_csv_path = TEMP_CLEANING_DIR / f"cleaned_{unique_id}.csv"
    stats_path = TEMP_CLEANING_DIR / f"stats_{unique_id}.json"
    
    try:
        # Save uploaded file temporarily
        with open(temp_csv_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read CSV with pandas
        try:
            df = pd.read_csv(temp_csv_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Validate that CSV has data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Parse target columns
        target_cols = []
        if target_columns:
            target_cols = [col.strip() for col in target_columns.split(',')]
            # Validate target columns exist
            missing_targets = [col for col in target_cols if col not in df.columns]
            if missing_targets:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Target columns not found in dataset: {missing_targets}"
                )
        
        # Create cleaning configuration
        config = CleaningConfig(
            target_columns=target_cols,
            high_cardinality_threshold=high_cardinality_threshold,
            k_best_features=k_best_features,
            standardize=standardize,
            remove_duplicates=remove_duplicates,
            remove_datetime=remove_datetime,
            remove_constant=remove_constant,
            remove_high_cardinality=remove_high_cardinality
        )
        
        # Initialize cleaner and process dataset
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_dataset(df, config)
        
        # Save cleaned dataset
        cleaned_df.to_csv(output_csv_path, index=False)
        
        # Save statistics
        with open(stats_path, 'w') as f:
            json.dump(cleaner.feature_stats, f, indent=2, default=str)
        
        # Clean up temporary input file
        temp_csv_path.unlink()
        
        # Return the cleaned CSV file
        return FileResponse(
            path=output_csv_path,
            filename=f"cleaned_{file.filename}",
            media_type="text/csv",
            headers={"Content-Disposition": "attachment"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        for path in [temp_csv_path, output_csv_path, stats_path]:
            if path.exists():
                path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error cleaning dataset: {str(e)}")

@Fleet_router.get("/download-stats/{file_id}")
async def download_stats(file_id: str):
    """Download the statistics JSON file for a previously cleaned dataset."""
    stats_path = TEMP_CLEANING_DIR / f"stats_{file_id}.json"
    
    if not stats_path.exists():
        raise HTTPException(status_code=404, detail="Statistics file not found")
    
    return FileResponse(
        path=stats_path,
        filename=f"cleaning_stats_{file_id}.json",
        media_type="application/json"
    )

@Fleet_router.post("/train-model/")
async def train_model(
    file: UploadFile = File(..., description="CSV file to train models on"),
    classification_target: Optional[str] = Form(None, description="Name of classification target column"),
    regression_target: Optional[str] = Form(None, description="Name of regression target column"),
    exclude_columns: Optional[str] = Form(None, description="Comma-separated list of columns to exclude"),
    test_size: float = Form(0.2, description="Proportion of data to use for testing"),
    random_state: int = Form(42, description="Random state for reproducibility"),
    tune_hyperparameters: bool = Form(True, description="Whether to tune hyperparameters")
):
    """
    Train machine learning models on uploaded CSV data with enhanced error handling.
    
    Returns comprehensive evaluation results including metrics and visualizations.
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_MODELS_DIR / f"temp_{unique_id}.csv"
    model_path = TEMP_MODELS_DIR / f"model_{unique_id}.pkl"
    results_path = TEMP_MODELS_DIR / f"results_{unique_id}.json"
    
    try:
        # Save uploaded file temporarily
        with open(temp_csv_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read CSV with pandas
        try:
            df = pd.read_csv(temp_csv_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Validate that CSV has data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Parse exclude columns
        exclude_cols = []
        if exclude_columns:
            exclude_cols = [col.strip() for col in exclude_columns.split(',')]
        
        # Create training configuration
        config = TrainingConfig(
            classification_target=classification_target,
            regression_target=regression_target,
            exclude_columns=exclude_cols,
            test_size=test_size,
            random_state=random_state,
            tune_hyperparameters=tune_hyperparameters,
            log_to_mlflow=True  # Enable MLflow logging
        )
        
        # Initialize and train model
        model = FleetMaintenanceModel()
        results = model.train_and_evaluate(df, config)
        
        # Validate and clean results before returning
        cleaned_results = validate_json_serializable(results)
        
        # Save results with validated data
        with open(results_path, 'w') as f:
            json.dump(cleaned_results, f, indent=2)
        
        # Add model ID to results for downloading
        cleaned_results["model_id"] = unique_id
        
        # Clean up temporary input file
        temp_csv_path.unlink()
        
        return JSONResponse(content=cleaned_results)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        for path in [temp_csv_path, model_path, results_path]:
            if path.exists():
                path.unlink()
        
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail=f"Error training models: {str(e)}")

@Fleet_router.post("/predict/")
async def predict(
    data_file: UploadFile = File(..., description="CSV file with data to predict on"),
    model_file: UploadFile = File(..., description="Trained model file (.pkl)")
):
    """Make predictions using a trained model on new data with enhanced validation."""
    
    # Validate file types
    if not data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Data file must be CSV")
    if not model_file.filename.endswith('.pkl'):
        raise HTTPException(status_code=400, detail="Model file must be .pkl")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_MODELS_DIR / f"predict_data_{unique_id}.csv"
    temp_model_path = TEMP_MODELS_DIR / f"predict_model_{unique_id}.pkl"
    
    try:
        # Save uploaded files temporarily
        with open(temp_csv_path, "wb") as buffer:
            content = await data_file.read()
            buffer.write(content)
        
        with open(temp_model_path, "wb") as buffer:
            content = await model_file.read()
            buffer.write(content)
        
        # Load data and model
        try:
            df = pd.read_csv(temp_csv_path)
            with open(temp_model_path, 'rb') as f:
                models_dict = pickle.load(f)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading files: {str(e)}")
        
        # Prepare features (same logic as training)
        exclude_cols = models_dict.get('exclude_columns', [])
        exclude_cols.extend([
            models_dict.get('classification_target', 'component_at_risk'),
            models_dict.get('regression_target', 'days_till_breakdown')
        ])
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove categorical columns
        cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in cat_cols]
        
        X = df[feature_cols]
        
        # Make predictions
        classification_model = models_dict['classification_model']
        label_encoder = models_dict['label_encoder']
        
        predictions = {}
        
        # Classification predictions
        y_pred = classification_model.predict(X)
        y_pred_proba = classification_model.predict_proba(X)
        
        # Decode predictions
        predictions['classification'] = {
            'predicted_classes': label_encoder.inverse_transform(y_pred).tolist(),
            'predicted_probabilities': y_pred_proba.tolist(),
            'class_names': label_encoder.classes_.tolist()
        }
        
        # Regression predictions if model exists
        if 'regression_model' in models_dict:
            regression_model = models_dict['regression_model']
            y_reg_pred = regression_model.predict(X)
            predictions['regression'] = {
                'predicted_values': y_reg_pred.tolist()
            }
        
        # Clean up temporary files
        temp_csv_path.unlink()
        temp_model_path.unlink()
        
        # Validate predictions before returning
        validated_predictions = validate_json_serializable(predictions)
        
        return JSONResponse(content=validated_predictions)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        for path in [temp_csv_path, temp_model_path]:
            if path.exists():
                path.unlink()
        
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

@Fleet_router.get("/download-model/{model_id}")
async def download_model(model_id: str):
    """Download a trained model file."""
    model_path = TEMP_MODELS_DIR / f"model_{model_id}.pkl"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        path=model_path,
        filename=f"trained_model_{model_id}.pkl",
        media_type="application/octet-stream"
    )

@Fleet_router.get("/download-results/{model_id}")
async def download_results(model_id: str):
    """Download the training results JSON file."""
    results_path = TEMP_MODELS_DIR / f"results_{model_id}.json"
    
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        path=results_path,
        filename=f"training_results_{model_id}.json",
        media_type="application/json"
    )

@Fleet_router.post("/analyze-data-shap/")
async def analyze_data_with_shap(
    data_file: UploadFile = File(..., description="CSV file with data to analyze"),
    target_column: str = Form(..., description="Name of target column"),
    max_display: int = Form(20, description="Maximum number of features to display in plot"),
    sample_size: Optional[int] = Form(None, description="Sample size for faster processing (None for all data)")
):
    """
    Analyze feature importance using SHAP with Decision Tree by training on the uploaded data.
    Only generates beeswarm plot.
    """
    
    # Validate file type
    if not data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Data file must be CSV")
    
    # Generate unique filename
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_SHAP_DIR / f"data_{unique_id}.csv"
    
    try:
        # Save uploaded file temporarily
        with open(temp_csv_path, "wb") as buffer:
            content = await data_file.read()
            buffer.write(content)
        
        # Load data
        try:
            df = pd.read_csv(temp_csv_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Validate target column exists
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in data")
        
        # Create configuration
        config = SHAPConfig(
            target_column=target_column,
            max_display=max_display,
            sample_size=sample_size
        )
        
        # Initialize analyzer
        analyzer = SHAPAnalyzer()
        
        # Prepare data
        X, y = analyzer.prepare_data(df, target_column, sample_size)
        
        if len(X) == 0:
            raise HTTPException(status_code=400, detail="No valid features found in the data")
        
        # Train Decision Tree model
        analyzer.train_decision_tree_model(X, y)
        
        # Generate SHAP explanations
        results = analyzer.generate_shap_explanations(X, config)
        
        # Clean up temporary file
        temp_csv_path.unlink()
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file in case of error
        if temp_csv_path.exists():
            temp_csv_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error analyzing data: {str(e)}")

@Fleet_router.post("/analyze-counterfactuals/")
async def analyze_counterfactuals(
    data_file: UploadFile = File(..., description="CSV file with fleet data"),
    model_file: UploadFile = File(..., description="Trained model file (.pkl)"),
    vehicle_index: int = Form(..., description="Index of vehicle instance to analyze"),
    target_component: str = Form("None", description="Target component risk (usually 'None' for no risk)"),
    num_counterfactuals: int = Form(3, description="Number of counterfactuals to generate"),
    include_synthetic: bool = Form(True, description="Include synthetic counterfactuals if needed")
):
    """
    Generate counterfactual explanations for a specific vehicle instance.
    
    Returns actionable maintenance recommendations to achieve target risk level.
    """
    
    # Validate file types
    if not data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Data file must be CSV")
    if not model_file.filename.endswith('.pkl'):
        raise HTTPException(status_code=400, detail="Model file must be .pkl")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_CFS_DIR / f"data_{unique_id}.csv"
    temp_model_path = TEMP_CFS_DIR / f"model_{unique_id}.pkl"
    
    try:
        # Save uploaded files temporarily
        with open(temp_csv_path, "wb") as buffer:
            content = await data_file.read()
            buffer.write(content)
        
        with open(temp_model_path, "wb") as buffer:
            content = await model_file.read()
            buffer.write(content)
        
        # Initialize explainer
        explainer = FleetMaintenanceExplainer()
        explainer.load_model_from_file(temp_model_path)
        explainer.load_data(temp_csv_path)
        
        # Validate vehicle index
        if vehicle_index < 0 or vehicle_index >= len(explainer.df):
            raise HTTPException(status_code=400, detail=f"Vehicle index {vehicle_index} out of range (0-{len(explainer.df)-1})")
        
        # Get the specific instance
        instance = explainer.df.iloc[vehicle_index:vehicle_index+1].copy()
        
        # Validate that instance has component at risk
        if 'component_at_risk' not in instance.columns:
            raise HTTPException(status_code=400, detail="Dataset must contain 'component_at_risk' column")
        
        # Create configuration
        config = CounterfactualConfig(
            target_component=target_component,
            num_counterfactuals=num_counterfactuals,
            include_synthetic=include_synthetic
        )
        
        # Generate counterfactuals
        counterfactuals = explainer.generate_counterfactuals_to_target(instance, config)
        
        # Analyze counterfactuals
        analysis = explainer.analyze_counterfactuals(instance, counterfactuals, config)
        
        # Clean up temporary files
        temp_csv_path.unlink()
        temp_model_path.unlink()
        
        return JSONResponse(content=analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        for path in [temp_csv_path, temp_model_path]:
            if path.exists():
                path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error analyzing counterfactuals: {str(e)}")

@Fleet_router.post("/get-components/")
async def get_components(
    data_file: UploadFile = File(..., description="CSV file with fleet data")
):
    """
    Get list of available component risk categories from the dataset.
    """
    
    if not data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Data file must be CSV")
    
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_CFS_DIR / f"data_{unique_id}.csv"
    
    try:
        # Save uploaded file temporarily
        with open(temp_csv_path, "wb") as buffer:
            content = await data_file.read()
            buffer.write(content)
        
        # Load data
        df = pd.read_csv(temp_csv_path)
        
        if 'component_at_risk' not in df.columns:
            raise HTTPException(status_code=400, detail="Dataset must contain 'component_at_risk' column")
        
        components = sorted(df['component_at_risk'].unique())
        component_counts = {str(k): int(v) for k, v in df['component_at_risk'].value_counts().to_dict().items()}
        
        # Clean up temporary file
        temp_csv_path.unlink()
        
        return JSONResponse(content={
            "components": [str(comp) for comp in components],
            "component_counts": component_counts,
            "total_instances": int(len(df))
        })
        
    except HTTPException:
        raise
    except Exception as e:
        if temp_csv_path.exists():
            temp_csv_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@Fleet_router.post("/get-instances/")
async def get_instances(
    data_file: UploadFile = File(..., description="CSV file with fleet data"),
    component: str = Form(..., description="Component at risk to filter by"),
    limit: int = Form(10, description="Maximum number of instances to return")
):
    """
    Get instances with specified component at risk for selection.
    """
    
    if not data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Data file must be CSV")
    
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_CFS_DIR / f"data_{unique_id}.csv"
    
    try:
        # Save uploaded file temporarily
        with open(temp_csv_path, "wb") as buffer:
            content = await data_file.read()
            buffer.write(content)
        
        # Initialize explainer
        explainer = FleetMaintenanceExplainer()
        explainer.load_data(temp_csv_path)
        
        # Get instances by component
        instances_df = explainer.get_instances_by_component(component, limit)
        
        if instances_df.empty:
            return JSONResponse(content={
                "instances": [],
                "message": f"No instances found with component '{component}'"
            })
        
        # Add index for selection
        instances_df = instances_df.reset_index()
        
        # Convert to dict and ensure JSON serializable types
        instances_list = []
        for _, row in instances_df.iterrows():
            instance_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    instance_dict[col] = None
                elif pd.api.types.is_integer_dtype(type(val)):
                    instance_dict[col] = int(val)
                elif pd.api.types.is_float_dtype(type(val)):
                    instance_dict[col] = float(val)
                else:
                    instance_dict[col] = str(val)
            instances_list.append(instance_dict)
        
        # Clean up temporary file
        temp_csv_path.unlink()
        
        return JSONResponse(content={
            "instances": instances_list,
            "count": int(len(instances_list))
        })
        
    except HTTPException:
        raise
    except Exception as e:
        if temp_csv_path.exists():
            temp_csv_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error retrieving instances: {str(e)}")

@Fleet_router.post("/analyze-feature/")
async def analyze_single_feature(
    data_file: UploadFile = File(..., description="CSV file with data"),
    model_file: UploadFile = File(..., description="Trained model file (.pkl)"),
    feature_name: str = Form(..., description="Name of feature to analyze"),
    num_points: int = Form(50, description="Number of points for PDP calculation"),
    include_classification: bool = Form(True, description="Include classification analysis"),
    include_regression: bool = Form(True, description="Include regression analysis")
):
    """Analyze a single feature with PDP plots."""
    
    # Validate file types
    if not data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Data file must be CSV")
    if not model_file.filename.endswith('.pkl'):
        raise HTTPException(status_code=400, detail="Model file must be .pkl")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_PDP_DIR / f"data_{unique_id}.csv"
    temp_model_path = TEMP_PDP_DIR / f"model_{unique_id}.pkl"
    
    try:
        # Save uploaded files
        with open(temp_csv_path, "wb") as buffer:
            content = await data_file.read()
            buffer.write(content)
        
        with open(temp_model_path, "wb") as buffer:
            content = await model_file.read()
            buffer.write(content)
        
        # Initialize analyzer
        analyzer = PDPAnalyzer()
        analyzer.load_data_and_model(str(temp_csv_path), str(temp_model_path))
        
        # Validate feature name
        if feature_name not in analyzer.numerical_features:
            available_features = analyzer.numerical_features[:10]  # Show first 10
            raise HTTPException(
                status_code=400, 
                detail=f"Feature '{feature_name}' not found in numerical features. Available: {available_features}"
            )
        
        # Create configuration
        config = PDPConfig(
            num_points=num_points,
            feature_list=[feature_name],
            include_classification=include_classification,
            include_regression=include_regression
        )
        
        # Analyze feature
        results = analyzer.analyze_multiple_features(config)
        
        # Extract single feature result
        feature_result = results['feature_analyses'][0] if results['feature_analyses'] else {}
        
        # Clean up temporary files
        temp_csv_path.unlink()
        temp_model_path.unlink()
        
        return JSONResponse(content={
            "feature_name": feature_name,
            "analysis": feature_result,
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        for path in [temp_csv_path, temp_model_path]:
            if path.exists():
                path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error analyzing feature: {str(e)}")

@Fleet_router.post("/analyze-features/")
async def analyze_multiple_features(
    data_file: UploadFile = File(..., description="CSV file with data"),
    model_file: UploadFile = File(..., description="Trained model file (.pkl)"),
    num_points: int = Form(50, description="Number of points for PDP calculation"),
    top_n_features: Optional[int] = Form(None, description="Number of top features to analyze by importance"),
    feature_names: Optional[str] = Form(None, description="Comma-separated feature names to analyze"),
    include_classification: bool = Form(True, description="Include classification analysis"),
    include_regression: bool = Form(True, description="Include regression analysis")
):
    """Analyze multiple features with PDP plots."""
    
    # Validate file types
    if not data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Data file must be CSV")
    if not model_file.filename.endswith('.pkl'):
        raise HTTPException(status_code=400, detail="Model file must be .pkl")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_PDP_DIR / f"data_{unique_id}.csv"
    temp_model_path = TEMP_PDP_DIR / f"model_{unique_id}.pkl"
    
    try:
        # Save uploaded files
        with open(temp_csv_path, "wb") as buffer:
            content = await data_file.read()
            buffer.write(content)
        
        with open(temp_model_path, "wb") as buffer:
            content = await model_file.read()
            buffer.write(content)
        
        # Initialize analyzer
        analyzer = PDPAnalyzer()
        analyzer.load_data_and_model(str(temp_csv_path), str(temp_model_path))
        
        # Parse feature names if provided
        feature_list = None
        if feature_names:
            feature_list = [f.strip() for f in feature_names.split(',')]
            # Validate features exist
            invalid_features = [f for f in feature_list if f not in analyzer.numerical_features]
            if invalid_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid features: {invalid_features}. Available: {analyzer.numerical_features[:10]}"
                )
        
        # Create configuration
        config = PDPConfig(
            num_points=num_points,
            top_n_features=top_n_features,
            feature_list=feature_list,
            include_classification=include_classification,
            include_regression=include_regression
        )
        
        # Analyze features
        results = analyzer.analyze_multiple_features(config)
        
        # Clean up temporary files
        temp_csv_path.unlink()
        temp_model_path.unlink()
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        for path in [temp_csv_path, temp_model_path]:
            if path.exists():
                path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error analyzing features: {str(e)}")

@Fleet_router.post("/get-features/")
async def get_feature_list(
    data_file: UploadFile = File(..., description="CSV file with data"),
    model_file: UploadFile = File(..., description="Trained model file (.pkl)")
):
    """Get list of available features for PDP analysis."""
    
    # Validate file types
    if not data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Data file must be CSV")
    if not model_file.filename.endswith('.pkl'):
        raise HTTPException(status_code=400, detail="Model file must be .pkl")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_PDP_DIR / f"data_{unique_id}.csv"
    temp_model_path = TEMP_PDP_DIR / f"model_{unique_id}.pkl"
    
    try:
        # Save uploaded files
        with open(temp_csv_path, "wb") as buffer:
            content = await data_file.read()
            buffer.write(content)
        
        with open(temp_model_path, "wb") as buffer:
            content = await model_file.read()
            buffer.write(content)
        
        # Initialize analyzer
        analyzer = PDPAnalyzer()
        analyzer.load_data_and_model(str(temp_csv_path), str(temp_model_path))
        
        # Get feature importance
        importance_data = analyzer.get_feature_importance()
        
        # Get basic statistics for numerical features
        df = pd.read_csv(temp_csv_path)
        feature_stats = {}
        for feature in analyzer.numerical_features:
            if feature in df.columns:
                feature_stats[feature] = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'missing_count': int(df[feature].isnull().sum())
                }
        
        # Clean up temporary files
        temp_csv_path.unlink()
        temp_model_path.unlink()
        
        return JSONResponse(content={
            "numerical_features": analyzer.numerical_features,
            "feature_statistics": feature_stats,
            "feature_importance": importance_data,
            "total_features": len(analyzer.numerical_features)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        for path in [temp_csv_path, temp_model_path]:
            if path.exists():
                path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error getting feature list: {str(e)}")

# NEW MLFLOW ENDPOINTS

@Fleet_router.post("/register-model/")
async def register_and_deploy_model(
    model_file: UploadFile = File(..., description="Trained model file (.pkl)"),
    model_name: str = Form(..., description="Name for the registered model"),
    description: str = Form("", description="Model description"),
    stage: str = Form("None", description="Model stage: None, Staging, Production, Archived"),
    tags: str = Form("", description="Comma-separated tags for the model")
):
    """
    Register and deploy trained Fleet Management model in MLflow.
    Handles both classification and regression models.
    """
    if not model_file.filename.endswith('.pkl'):
        raise HTTPException(status_code=400, detail="Model file must be .pkl")
    
    # Generate unique filename
    unique_id = str(uuid.uuid4())
    temp_model_path = TEMP_MODELS_DIR / f"register_model_{unique_id}.pkl"
    
    try:
        # Save uploaded model
        with open(temp_model_path, "wb") as buffer:
            content = await model_file.read()
            buffer.write(content)
        
        # Load the model to get metadata
        with open(temp_model_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        classification_model = model_dict.get('classification_model')
        regression_model = model_dict.get('regression_model')
        label_encoder = model_dict.get('label_encoder')
        
        if not classification_model:
            raise HTTPException(status_code=400, detail="No classification model found in file")
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        model_tags = {f"tag_{i+1}": tag for i, tag in enumerate(tag_list)}
        
        # Add default tags
        model_tags.update({
            "model_type": "fleet_maintenance",
            "framework": "xgboost",
            "has_classification": "true",
            "has_regression": str(regression_model is not None).lower(),
            "created_by": "fleet_management_pipeline"
        })
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"fleet_model_{unique_id}"):
            
            # Log model parameters
            if hasattr(classification_model, 'get_params'):
                mlflow.log_params({f"clf_{k}": v for k, v in classification_model.get_params().items()})
            
            if regression_model and hasattr(regression_model, 'get_params'):
                mlflow.log_params({f"reg_{k}": v for k, v in regression_model.get_params().items()})
            
            # Log model metadata
            mlflow.log_param("classification_target", model_dict.get('classification_target', 'component_at_risk'))
            mlflow.log_param("regression_target", model_dict.get('regression_target', 'days_till_breakdown'))
            mlflow.log_param("has_label_encoder", label_encoder is not None)
            
            if label_encoder:
                mlflow.log_param("target_classes", list(label_encoder.classes_))
            
            # Log classification model
            model_info = mlflow.xgboost.log_model(
                classification_model, 
                "classification_model",
                registered_model_name=f"{model_name}_classification"
            )
            
            # Log regression model if available
            regression_model_info = None
            if regression_model:
                regression_model_info = mlflow.xgboost.log_model(
                    regression_model, 
                    "regression_model",
                    registered_model_name=f"{model_name}_regression"
                )
            
            # Log the complete model dictionary as artifact
            mlflow.log_dict(
                {
                    "classification_target": model_dict.get('classification_target'),
                    "regression_target": model_dict.get('regression_target'),
                    "target_classes": list(label_encoder.classes_) if label_encoder else [],
                    "model_type": "fleet_maintenance_multi_model"
                },
                "model_metadata.json"
            )
            
            # Save and log the pickle file
            mlflow.log_artifact(str(temp_model_path), "model_files")
            
            # Get run info
            run = mlflow.active_run()
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
        
        # Initialize MLflow client for model management
        client = MlflowClient()
        
        # Get the latest version of the registered classification model
        classification_version = None
        regression_version = None
        
        try:
            latest_versions = client.get_latest_versions(f"{model_name}_classification", stages=["None"])
            if latest_versions:
                classification_version = latest_versions[0]
                
                # Update model version description and tags
                if description:
                    client.update_model_version(
                        name=f"{model_name}_classification",
                        version=classification_version.version,
                        description=f"{description} - Classification Model"
                    )
                
                # Set model tags
                for key, value in model_tags.items():
                    client.set_model_version_tag(f"{model_name}_classification", classification_version.version, key, value)
                
                # Transition model to specified stage if not "None"
                if stage and stage != "None":
                    client.transition_model_version_stage(
                        name=f"{model_name}_classification",
                        version=classification_version.version,
                        stage=stage
                    )
            
            # Handle regression model if it exists
            if regression_model:
                regression_latest_versions = client.get_latest_versions(f"{model_name}_regression", stages=["None"])
                if regression_latest_versions:
                    regression_version = regression_latest_versions[0]
                    
                    if description:
                        client.update_model_version(
                            name=f"{model_name}_regression",
                            version=regression_version.version,
                            description=f"{description} - Regression Model"
                        )
                    
                    for key, value in model_tags.items():
                        client.set_model_version_tag(f"{model_name}_regression", regression_version.version, key, value)
                    
                    if stage and stage != "None":
                        client.transition_model_version_stage(
                            name=f"{model_name}_regression",
                            version=regression_version.version,
                            stage=stage
                        )
        
        except Exception as model_management_error:
            logger.warning(f"Model management error: {model_management_error}")
            # Continue with response even if model management fails
        
        # Clean up temporary file
        if temp_model_path.exists():
            temp_model_path.unlink()
        
        response = {
            "message": "Fleet Management models registered successfully in MLflow",
            "mlflow_info": {
                "tracking_uri": MLFLOW_TRACKING_URI,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "classification_model": {
                    "name": f"{model_name}_classification",
                    "version": classification_version.version if classification_version else "N/A",
                    "uri": model_info.model_uri
                },
                "regression_model": {
                    "name": f"{model_name}_regression",
                    "version": regression_version.version if regression_version else "N/A",
                    "uri": regression_model_info.model_uri if regression_model_info else "N/A"
                }
            },
            "model_details": {
                "has_classification": True,
                "has_regression": regression_model is not None,
                "target_classes": list(label_encoder.classes_) if label_encoder else [],
                "description": description,
                "tags": model_tags
            },
            "deployment_info": {
                "stage": stage,
                "deployment_ready": True,
                "model_loading_code": f"""
# Load Fleet Management models from MLflow
import mlflow.pyfunc

# Load classification model
clf_model = mlflow.pyfunc.load_model('models:/{model_name}_classification/{classification_version.version if classification_version else 'latest'}')

# Load regression model (if available)
{'reg_model = mlflow.pyfunc.load_model("models:/' + model_name + '_regression/' + (regression_version.version if regression_version else 'latest') + '")' if regression_model else '# No regression model available'}

# Make predictions
classification_predictions = clf_model.predict(your_data)
{'regression_predictions = reg_model.predict(your_data)' if regression_model else ''}
"""
            },
            "links": {
                "mlflow_ui": f"{MLFLOW_TRACKING_URI}",
                "experiment_url": f"{MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}",
                "run_url": f"{MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{run_id}",
                "classification_model_url": f"{MLFLOW_TRACKING_URI}/#/models/{model_name}_classification",
                "regression_model_url": f"{MLFLOW_TRACKING_URI}/#/models/{model_name}_regression" if regression_model else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        if temp_model_path.exists():
            temp_model_path.unlink()
        raise
    except Exception as e:
        # Clean up files in case of error
        if temp_model_path.exists():
            temp_model_path.unlink()
        
        logger.error(f"Error registering Fleet Management model in MLflow: {e}")
        raise HTTPException(status_code=500, detail=f"Error registering model: {str(e)}")

@Fleet_router.post("/detect-drift/")
async def detect_data_drift(
    reference_data_file: UploadFile = File(..., description="Reference/training CSV data"),
    new_data_file: UploadFile = File(..., description="New CSV data to compare"),
    drift_threshold: float = Form(0.05, description="P-value threshold for drift detection"),
    classification_target: str = Form("component_at_risk", description="Classification target column"),
    regression_target: str = Form("days_till_breakdown", description="Regression target column"),
    record_in_mlflow: bool = Form(True, description="Whether to record drift results in MLflow")
):
    """
    Detect data drift by comparing new Fleet Management data with reference data.
    Records drift analysis results in MLflow for monitoring.
    """
    if not reference_data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Reference data file must be CSV")
    if not new_data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="New data file must be CSV")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_ref_path = TEMP_MODELS_DIR / f"drift_ref_{unique_id}.csv"
    temp_new_path = TEMP_MODELS_DIR / f"drift_new_{unique_id}.csv"
    
    try:
        # Save uploaded files
        with open(temp_ref_path, "wb") as buffer:
            content = await reference_data_file.read()
            buffer.write(content)
        
        with open(temp_new_path, "wb") as buffer:
            content = await new_data_file.read()
            buffer.write(content)
        
        # Load reference and new data
        try:
            reference_df = pd.read_csv(temp_ref_path)
            new_df = pd.read_csv(temp_new_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV files: {str(e)}")
        
        if reference_df.empty or new_df.empty:
            raise HTTPException(status_code=400, detail="CSV files cannot be empty")
        
        # Initialize drift analyzer
        drift_analyzer = FleetDataDriftAnalyzer()
        drift_analyzer.drift_threshold = drift_threshold
        
        # Set reference data
        ref_setup = drift_analyzer.set_reference_data(
            reference_df, 
            classification_target=classification_target,
            regression_target=regression_target
        )
        if ref_setup["status"] != "success":
            raise HTTPException(status_code=500, detail="Failed to set reference data")
        
        # Detect drift
        drift_results = drift_analyzer.detect_drift(
            new_df, 
            classification_target=classification_target,
            regression_target=regression_target
        )
        
        if drift_results["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Drift detection failed: {drift_results['message']}")
        
        # Record in MLflow if requested
        mlflow_info = None
        if record_in_mlflow:
            try:
                with mlflow.start_run(run_name=f"fleet_drift_detection_{unique_id}"):
                    
                    # Log drift parameters
                    mlflow.log_param("drift_threshold", drift_threshold)
                    mlflow.log_param("reference_data_size", len(reference_df))
                    mlflow.log_param("new_data_size", len(new_df))
                    mlflow.log_param("classification_target", classification_target)
                    mlflow.log_param("regression_target", regression_target)
                    
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
                    if drift_results["classification_target_drift"]:
                        mlflow.log_metric("classification_target_drift_pvalue", drift_results["classification_target_drift"]["p_value"])
                        mlflow.log_metric("classification_target_drift_detected", 1 if drift_results["classification_target_drift"]["drift_detected"] else 0)
                    
                    if drift_results["regression_target_drift"]:
                        mlflow.log_metric("regression_target_drift_pvalue", drift_results["regression_target_drift"]["p_value"])
                        mlflow.log_metric("regression_target_drift_detected", 1 if drift_results["regression_target_drift"]["drift_detected"] else 0)
                    
                    # Log drift results as JSON artifact
                    mlflow.log_dict(drift_results, "fleet_drift_analysis_results.json")
                    
                    # Log new data as artifact
                    mlflow.log_artifact(str(temp_new_path), "drift_monitoring")
                    
                    # Add tags for easy filtering
                    mlflow.set_tags({
                        "type": "fleet_drift_detection",
                        "drift_detected": str(drift_results["summary"]["overall_drift_detected"]),
                        "drift_severity": drift_results["summary"].get("overall_severity", "Unknown"),
                        "monitoring_type": "fleet_data_drift"
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
                    
            except Exception as e:
                logger.error(f"Failed to log drift results to MLflow: {e}")
                mlflow_info = {
                    "logged": False,
                    "error": str(e)
                }
        
        # Clean up temporary files
        temp_ref_path.unlink()
        temp_new_path.unlink()
        
        # Prepare comprehensive response
        response = {
            "message": "Fleet data drift detection completed successfully",
            "drift_analysis": drift_results,
            "mlflow_tracking": mlflow_info,
            "data_comparison": {
                "reference_data_shape": [int(reference_df.shape[0]), int(reference_df.shape[1])],
                "new_data_shape": [int(new_df.shape[0]), int(new_df.shape[1])],
                "classification_target": classification_target,
                "regression_target": regression_target
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
                    "Immediate attention required - significant fleet data drift detected",
                    "Consider retraining both classification and regression models",
                    "Review fleet data collection processes and sensor calibration",
                    "Implement continuous monitoring for fleet operations"
                ]
            elif severity == "Medium":
                response["recommendations"]["recommended_actions"] = [
                    "Monitor closely - moderate fleet data drift detected",
                    "Plan for model retraining in near future",
                    "Investigate root causes of operational changes"
                ]
            else:
                response["recommendations"]["recommended_actions"] = [
                    "Low-level drift detected - continue monitoring",
                    "No immediate action required for fleet operations"
                ]
        else:
            response["recommendations"]["recommended_actions"] = [
                "No significant drift detected in fleet data",
                "Current models should perform well on new fleet data",
                "Continue regular monitoring of fleet operations"
            ]
        
        # Final conversion to ensure JSON serialization
        response = convert_numpy_types(response)
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        for path in [temp_ref_path, temp_new_path]:
            if path.exists():
                path.unlink()
        
        logger.error(f"Error in fleet drift detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting drift: {str(e)}")

@Fleet_router.get("/mlflow-status")
async def check_mlflow_status():
    """Check MLflow server connectivity and list registered Fleet Management models."""
    try:
        client = MlflowClient()
        
        # Test connection by listing experiments
        experiments = client.search_experiments()
        
        # List registered models
        registered_models = client.search_registered_models()
        
        # Filter for fleet management models
        fleet_models = [model for model in registered_models 
                      if "fleet" in model.name.lower() or "classification" in model.name or "regression" in model.name]
        
        return {
            "status": "connected",
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "experiments_count": len(experiments),
            "registered_models_count": len(registered_models),
            "fleet_models_count": len(fleet_models),
            "fleet_models": [
                {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description
                }
                for model in fleet_models
            ],
            "message": "MLflow server is accessible - Fleet Management models ready"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "error": str(e),
            "message": "Cannot connect to MLflow server. Make sure 'mlflow ui' is running."
        }

# CSV CHAT ENDPOINTS

@Fleet_router.post("/csv-upload/", response_model=UploadResponse)
async def upload_csv_for_chat(
    file: UploadFile = File(..., description="CSV file to upload for chat analysis")
):
    """
    Upload CSV file for chat-based analysis using ChromaDB and Groq.
    Replaces the old multi-file upload functionality.
    """
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read file content
        contents = await file.read()
        
        # Upload to chatbot service
        result = csv_chatbot.upload_csv_data(contents, file.filename)
        
        return UploadResponse(
            message=result["message"],
            documents_processed=result["documents_processed"],
            rows=result["rows"],
            columns=result["columns"],
            filename=result["filename"]
        )
        
    except Exception as e:
        logger.error(f"Error uploading CSV for chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading CSV: {str(e)}")

@Fleet_router.post("/csv-query/", response_model=QueryResponse)
async def query_csv_chat(
    request: QueryRequest
):
    """
    Query the uploaded CSV data using natural language with Groq AI.
    """
    
    try:
        result = csv_chatbot.query_csv_data(request.query)
        
        return QueryResponse(
            answer=result["answer"],
            reference_data=result["reference_data"],
            raw_context=result["raw_context"]
        )
        
    except Exception as e:
        logger.error(f"Error querying CSV chat: {str(e)}")
        if "No CSV data found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Error querying data: {str(e)}")

@Fleet_router.get("/csv-status/")
async def get_csv_status():
    """Check if CSV data is available for chat"""
    try:
        has_data = csv_chatbot.has_data()
        return {
            "has_data": has_data,
            "message": "CSV data is available for chat" if has_data else "No CSV data uploaded yet"
        }
    except Exception as e:
        logger.error(f"Error checking CSV status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking status: {str(e)}")

@Fleet_router.delete("/csv-clear/")
async def clear_csv_data():
    """Clear all CSV data from the chat system"""
    try:
        result = csv_chatbot.clear_data()
        return result
    except Exception as e:
        logger.error(f"Error clearing CSV data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

@Fleet_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Fleet ML API with MLflow Integration"}