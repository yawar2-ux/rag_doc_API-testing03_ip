#!/usr/bin/env python3
"""
Complete FastAPI application for transportation ridership data pipeline:
1. Upload CSV 
2. Generate YData profiling report (auto-processes uploaded CSV)
3. Clean data (auto-processes uploaded CSV)
4. Train model (auto-processes cleaned CSV with 'ridership' target)
5. Make predictions (auto-processes trained model with test CSV)
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
import uuid
import warnings
import logging
import mlflow
import mlflow.sklearn
import chromadb
import groq
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import xgboost as xgb  # ADD MISSING IMPORT
import shap  # ADD MISSING IMPORT
import base64  # ADD MISSING IMPORT
import io  # ADD MISSING IMPORT
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from scipy.stats import ks_2samp # Added import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # Added import
from ydata_profiling import ProfileReport # Added import
from chromadb.utils import embedding_functions # Added import
from fastapi import APIRouter, File, UploadFile, HTTPException, Path as FastAPIPath, Form, FastAPI # Added FastAPI here
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional

# Import local services
from ..services.data_cleaner import SimpleTimeSeriesCleaner
from ..services.model_trainer import RidershipPredictor, TrainingConfig
from ..services.model_inference import RidershipInference, load_and_predict
from ..services.shap_analysis import SHAPAnalyzer
from ..services.partial_dependency import PartialDependencyAnalyzer

class RidershipPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def prepare_data(self, df, config: TrainingConfig):
        """Prepare data for model training using config"""
        # Create feature columns list
        exclude_cols = [config.target_column, 'datetime', 'date', 'time'] + config.exclude_columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[config.target_column]
        
        self.feature_columns = feature_cols
        
        # Time-aware split if datetime available and requested
        if config.time_aware_split and config.datetime_column in df.columns:
            df_sorted = df.sort_values(config.datetime_column)
            split_idx = int(len(df_sorted) * (1 - config.test_size))
            
            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]
            
            X_train = train_df[feature_cols]
            X_test = test_df[feature_cols]
            y_train = train_df[config.target_column]
            y_test = test_df[config.target_column]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=config.random_state
            )
        
        return X_train, X_test, y_train, y_test
    
    def generate_prediction_plot(self, y_test, y_pred):  # ADD MISSING METHOD
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
    
    def train_and_evaluate(self, df, config: TrainingConfig):  # FIX: Add config parameter
        """Train XGBoost model and evaluate performance with config"""
        # Prepare data using config
        X_train, X_test, y_train, y_test = self.prepare_data(df, config)
        
        # Train model with hyperparameter tuning if requested
        if config.tune_hyperparameters:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.random_state,
                n_jobs=-1
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=config.random_state,
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
        
        # Create results with corrected structure
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
            "feature_analysis": {
                "top_10_features": [
                    {"feature": str(row.feature), "importance": float(row.importance)}
                    for _, row in feature_importance.head(10).iterrows()
                ]
            },
            "visualization": f"data:image/png;base64,{prediction_plot}"
        }
        
        return results

    # ...existing code...

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Directory to store files
TEMP_DIR = Path("pipeline_files")
TEMP_DIR.mkdir(exist_ok=True)

MLFLOW_TRACKING_URI = "http://localhost:5000"  # Updated to match server port
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Clean up any existing temporary files and initialize services
    for temp_file in TEMP_DIR.glob("*"):
        try:
            temp_file.unlink()
        except:
            pass
    
    # Initialize Groq client
    try:
        init_groq_client()
        logger.info("Groq client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
    
    yield
    
    # Shutdown: Clean up temporary files
    for temp_file in TEMP_DIR.glob("*"):
        try:
            temp_file.unlink()
        except:
            pass

# Create APIRouter instead of FastAPI app
TransportationDemandForecasting_router = APIRouter()

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
            "8. Chat with CSV data (natural language queries on raw CSV)"
        ],
        "endpoints": {
            "upload": "/upload-csv/",
            "profile": "/profile-data/{pipeline_id}",
            "clean": "/clean-data/{pipeline_id}",
            "train": "/train-model/{pipeline_id}",
            "predict": "/predict-ridership/{pipeline_id}",
            "shap": "/analyze-shap/{pipeline_id}",
            "pdp": "/create-pdp/{pipeline_id}",
            "chat": "/chat/{pipeline_id}"
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
    Logs experiment to MLflow under 'transport'.
    """

    import mlflow
    from mlflow import sklearn

    cleaned_csv_path = TEMP_DIR / f"cleaned_data_{pipeline_id}.csv"
    model_path = TEMP_DIR / f"model_{pipeline_id}.pkl"
    train_data_path = TEMP_DIR / f"train_data.csv"
    feature_columns_path = TEMP_DIR / f"feature_columns.json"

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
        
        # Create training configuration
        config = TrainingConfig(
            target_column="ridership",
            exclude_columns=[],
            test_size=0.2,
            random_state=42,
            tune_hyperparameters=True,
            time_aware_split=True,
            datetime_column="datetime"
        )
        
        results = predictor.train_and_evaluate(df, config)

        # Save train data CSV
        df.to_csv(train_data_path, index=False)

        # Save feature columns JSON
        with open(feature_columns_path, "w") as f:
            json.dump(predictor.feature_columns, f)

        # Load saved label encoders
        encoders_path = TEMP_DIR / f"encoders_{pipeline_id}.pkl"
        label_encoders = {}
        if encoders_path.exists():
            with open(encoders_path, 'rb') as f:
                label_encoders = pickle.load(f)

        # Save trained model locally
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': predictor.model,
                'feature_columns': predictor.feature_columns,
                'label_encoders': label_encoders
            }, f)

        # === Start MLflow logging ===
        mlflow.set_experiment("transport")

        with mlflow.start_run(run_name=f"pipeline_{pipeline_id}") as run:
            mlflow.log_params({
                "model_type": "XGBoost Regressor",
                "pipeline_id": pipeline_id,
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1
            })

            # Log training and testing metrics
            for k, v in results["performance"]["training_metrics"].items():
                mlflow.log_metric(f"train_{k}", v)

            for k, v in results["performance"]["testing_metrics"].items():
                mlflow.log_metric(f"test_{k}", v)

            # Log feature importance - FIX: Use correct key from results
            if "feature_analysis" in results and "top_10_features" in results["feature_analysis"]:
                for feat in results["feature_analysis"]["top_10_features"]:
                    mlflow.log_metric(f"feature_importance_{feat['feature']}", feat["importance"])

            # Log the model
            mlflow.sklearn.log_model(
                predictor.model,
                artifact_path=f"ridership_model"
            )

            # Log artifacts
            mlflow.log_artifact(str(cleaned_csv_path), artifact_path="cleaned_data")
            mlflow.log_artifact(str(train_data_path), artifact_path="train_data")
            mlflow.log_artifact(str(feature_columns_path), artifact_path="feature_columns")
            if encoders_path.exists():
                mlflow.log_artifact(str(encoders_path), artifact_path="label_encoders")

            # Add MLflow info to results
            results["mlflow_run_id"] = run.info.run_id
            results["mlflow_experiment_id"] = run.info.experiment_id
            results["mlflow_url"] = f"http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

        # Add pipeline info to results
        results["pipeline_id"] = pipeline_id
        results["timestamp"] = datetime.now().isoformat()
        results["target_column"] = "ridership"
        results["download_url"] = f"/download-model/{pipeline_id}"

        return JSONResponse(content={
            "message": "Model trained and logged to MLflow successfully",
            "pipeline_id": pipeline_id,
            "model_results": results
        })

    except HTTPException:
        raise
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@TransportationDemandForecasting_router.post("/register-model/")
async def register_model(
    run_id: str = Form(..., description="MLflow run ID to register the model from"),
    model_name: str = Form("RidershipXGBoostModel", description="Name to register the model under")
):
    """
    Register a trained model from a given run ID in MLflow Model Registry.
    Optionally provide a custom model name.

    - **run_id**: MLflow run ID returned after training
    - **model_name**: Name to register the model under (default: RidershipXGBoostModel)
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    try:
        client = MlflowClient()
        
        # Construct the run-relative artifact URI
        model_uri = f"runs:/{run_id}/ridership_model"
        
        # Register the model
        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        return {
            "message": "Model registered successfully",
            "model_name": model_name,
            "version": result.version,
            "run_id": run_id,
            "registration_timestamp": datetime.now().isoformat(),
            "model_uri": model_uri,
            "mlflow_model_url": f"http://localhost:5000/#/models/{model_name}/versions/{result.version}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model registration failed: {str(e)}")

@TransportationDemandForecasting_router.post("/deploy-model/")
async def deploy_model(
    run_id: str = Form(..., description="MLflow run ID from the training run")
):
    """
    Deploy the trained model by:
    - Loading it from the specified run ID
    - Logging it to a new experiment: 'model deployment-transportation'
    - Registering and optionally transitioning it to Production
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    try:
        client = MlflowClient()

        # Create or set the experiment
        mlflow.set_experiment("model deployment-transportation")

        # Build the artifact path (must match training run's log path)
        model_uri = f"runs:/{run_id}/ridership_model"

        with mlflow.start_run(run_name=f"deployment_from_{run_id}") as run:
            # Register the model under a new version (or same name)
            registered_model_name = "DeployedRidershipModel"
            mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

            # Optional: Promote to Production
            # client.transition_model_version_stage(
            #     name=registered_model_name,
            #     version=mv.version,
            #     stage="Production",
            #     archive_existing_versions=True
            # )

            return {
                "message": "Model deployed successfully",
                "source_run_id": run_id,
                "deployment_run_id": run.info.run_id,
                # "deployment_experiment_id": run.info.experiment_id,
                # "registered_model_name": registered_model_name,
                # "model_version": mv.version,
                "mlflow_url": f"http://localhost:5000"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

@TransportationDemandForecasting_router.post("/predict-model/")
async def predict_model(
    run_id: str = Form(..., description="MLflow run ID containing the trained model"),
    test_file: UploadFile = File(..., description="Test CSV file for prediction")
):
    """
    Predict using a model from a given MLflow run.
    Loads model using: runs:/<run_id>/ridership_model
    Automatically encodes categorical columns.
    """
    import mlflow
    import pandas as pd
    import numpy as np
    import uuid
    import json
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from fastapi import HTTPException

    try:
        # Save uploaded test CSV
        temp_test_path = TEMP_DIR / f"test_data_{uuid.uuid4().hex}.csv"
        with open(temp_test_path, "wb") as f:
            content = await test_file.read()
            f.write(content)

        # Load test data
        df = pd.read_csv(temp_test_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded test file is empty")

        # Set tracking URI and load model
        mlflow.set_tracking_uri("http://localhost:5000")
        model_uri = f"runs:/{run_id}/ridership_model"
        model = mlflow.sklearn.load_model(model_uri)

        # Load feature columns from local or MLflow
        feature_columns_path = TEMP_DIR / f"feature_columns.json"
        if feature_columns_path.exists():
            with open(feature_columns_path, "r") as f:
                feature_columns = json.load(f)
        else:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            local_artifact_path = client.download_artifacts(
                run_id,
                f"feature_columns/feature_columns.json"
            )
            with open(local_artifact_path, "r") as f:
                feature_columns = json.load(f)

        # Validate columns
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_features}")

        # Encode any object-type columns using simple categorical encoding
        for col in feature_columns:
            if df[col].dtype == 'object':
                df[col] = pd.Categorical(df[col]).codes

        # Prepare features and predict
        X = df[feature_columns]
        predictions = model.predict(X)

        result = {
            "status": "success",
            "prediction_count": int(len(predictions)),
            "predictions": [float(p) for p in predictions],
            "prediction_stats": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
            }
        }

        # Optional: Evaluate if actual target column is present
        if "ridership" in df.columns:
            y_true = df["ridership"].values
            y_pred = predictions

            result["evaluation_metrics"] = {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
                "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        if temp_test_path.exists():
            temp_test_path.unlink()

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

@TransportationDemandForecasting_router.post("/evaluate/")
async def evaluate_drift_without_evidently(
    run_id: str = Form(...),
    test_file: UploadFile = File(...)
):
    try:
        # Save and read uploaded test file
        test_path = TEMP_DIR / f"test_data_{uuid.uuid4().hex}.csv"
        with open(test_path, "wb") as f:
            f.write(await test_file.read())
        test_df = pd.read_csv(test_path)

        # Load reference training data from MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.tracking.MlflowClient()
        ref_path = client.download_artifacts(run_id, "train_data/train_data.csv")
        ref_df = pd.read_csv(ref_path)

        ref_df.dropna(axis=1, how='all', inplace=True)
        test_df.dropna(axis=1, how='all', inplace=True)

        common_cols = list(set(ref_df.columns) & set(test_df.columns))

        drift_results = []
        for col in common_cols:
            try:
                if pd.api.types.is_numeric_dtype(ref_df[col]):
                    ks_result = ks_2samp(ref_df[col].dropna(), test_df[col].dropna())
                    drift_results.append({
                        "column": col,
                        "type": "numerical",
                        "ks_statistic": float(ks_result.statistic),
                        "p_value": float(ks_result.pvalue),
                        "drift_detected": bool(ks_result.pvalue < 0.05)
                    })
                else:
                    ref_col = ref_df[col].astype(str)
                    test_col = test_df[col].astype(str)
                    ref_counts = ref_col.value_counts(normalize=True)
                    test_counts = test_col.value_counts(normalize=True)
                    overlap = ref_counts.mul(test_counts, fill_value=0).sum()
                    drift_results.append({
                        "column": col,
                        "type": "categorical",
                        "category_overlap": float(overlap),
                        "drift_detected": bool(overlap < 0.7)
                    })
            except Exception:
                continue  # Skip error columns

        # ✅ Evaluation directly from uploaded CSV
        performance_metrics = {}
        try:
            if "ridership" in test_df.columns and "predicted" in test_df.columns:
                y_true = test_df["ridership"]
                y_pred = test_df["predicted"];

                y_true = np.rint(y_true).astype(int)
                y_pred = np.rint(y_pred).astype(int)

                performance_metrics = {
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
                }
            else:
                performance_metrics = {
                    "error": "Columns 'ridership' and/or 'predicted' not found in uploaded file"
                }
        except Exception as e:
            performance_metrics = {
                "error": f"Failed to evaluate metrics: {str(e)}"
            }

        return JSONResponse(content={
            "run_id": run_id,
            "drift_analysis": drift_results,
            # "model_performance": performance_metrics
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

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
        filename=f"ridership_model.pkl",
        media_type="application/octet-stream"
    )

@TransportationDemandForecasting_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "Transportation Data Pipeline API",
        "timestamp": datetime.now().isoformat()
    }