#!/usr/bin/env python3
"""
FastAPI application for transportation ridership prediction model training.
Upload CSV files to train XGBoost models and get comprehensive evaluation results.
"""

import os
import uuid
import json
import pickle
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from io import StringIO

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import warnings

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

warnings.filterwarnings('ignore')

# Directory to store temporary files
TEMP_DIR = Path("temp_models")
TEMP_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Clean up any existing temporary files
    for temp_file in TEMP_DIR.glob("*"):
        try:
            temp_file.unlink()
        except:
            pass
    yield
    # Shutdown: Clean up temporary files
    for temp_file in TEMP_DIR.glob("*"):
        try:
            temp_file.unlink()
        except:
            pass

app = FastAPI(
    title="Transportation Ridership Model Trainer API",
    description="Upload CSV files to train XGBoost ridership prediction models and get comprehensive evaluation results",
    version="1.0.0",
    lifespan=lifespan
)

class TrainingConfig(BaseModel):
    """Configuration for model training parameters"""
    target_column: Optional[str] = "ridership"
    exclude_columns: Optional[List[str]] = None
    test_size: float = 0.2
    random_state: int = 42
    tune_hyperparameters: bool = True
    time_aware_split: bool = True
    datetime_column: Optional[str] = "datetime"

class RidershipPredictor:
    def __init__(self, target_column='ridership'):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_column = target_column
        self.feature_columns = None
        self.exclude_columns = ['datetime', 'date', 'time']  # Default columns to exclude
        
    def prepare_data(self, df, config: TrainingConfig):
        """Prepare data for model training with time-aware splitting"""
        # Update configuration
        if config.target_column:
            self.target_column = config.target_column
        if config.exclude_columns:
            self.exclude_columns.extend(config.exclude_columns)
        
        # Check if target column exists
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset!")
        
        # Create feature columns list
        exclude_cols = [self.target_column] + self.exclude_columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical columns with label encoding
        for col in feature_cols:
            if df[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        X = df[feature_cols]
        y = df[self.target_column]
        
        self.feature_columns = feature_cols
        
        # Time-aware split if datetime column is available
        if config.time_aware_split and config.datetime_column in df.columns:
            df_sorted = df.sort_values(config.datetime_column)
            split_idx = int(len(df_sorted) * (1 - config.test_size))
            
            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]
            
            X_train = train_df[feature_cols]
            X_test = test_df[feature_cols]
            y_train = train_df[self.target_column]
            y_test = test_df[self.target_column]
        else:
            # Regular random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=config.random_state
            )
        
        return X_train, X_test, y_train, y_test
    
    def tune_hyperparameters(self, X_train, y_train):
        """Hyperparameter tuning for XGBoost regressor"""
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse'
        )
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
    
    def generate_prediction_plot(self, y_test, y_pred):
        """Generate predicted vs actual values scatter plot"""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        plt.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        max_val = max(max(y_test), max(y_pred))
        min_val = min(min(y_test), min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                label='Perfect Prediction', linewidth=2)
        
        # Calculate metrics for display
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        plt.xlabel('Actual Ridership', fontsize=12)
        plt.ylabel('Predicted Ridership', fontsize=12)
        plt.title('Ridership Prediction Performance: Predicted vs Actual', fontsize=14)
        
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
    
    def generate_feature_importance_plot(self, top_n=15):
        """Generate feature importance plot"""
        if self.model is None:
            return ""
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance (XGBoost)', fontsize=14)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()
    

    
    def generate_time_series_plot(self, df, y_test, y_pred, config: TrainingConfig):
        """Generate time series plot if datetime column is available"""
        if not config.time_aware_split or config.datetime_column not in df.columns:
            return ""
        
        # Get test data with datetime
        df_sorted = df.sort_values(config.datetime_column)
        split_idx = int(len(df_sorted) * (1 - config.test_size))
        test_df = df_sorted.iloc[split_idx:].copy()
        
        test_df['predicted'] = y_pred
        test_df['actual'] = y_test
        
        # Sample data for plotting (to avoid overcrowding)
        if len(test_df) > 500:
            test_df = test_df.sample(n=500).sort_values(config.datetime_column)
        
        plt.figure(figsize=(15, 8))
        
        plt.plot(test_df[config.datetime_column], test_df['actual'], 
                'o-', label='Actual Ridership', alpha=0.7, markersize=4)
        plt.plot(test_df[config.datetime_column], test_df['predicted'], 
                'o-', label='Predicted Ridership', alpha=0.7, markersize=4)
        
        plt.xlabel('Date/Time')
        plt.ylabel('Ridership')
        plt.title('Ridership Prediction Over Time (Test Set Sample)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()
    
    def train_and_evaluate(self, df, config: TrainingConfig):
        """Train XGBoost model and evaluate performance"""
        start_time = datetime.now()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, config)
        
        # Train model
        if config.tune_hyperparameters:
            self.model, best_params = self.tune_hyperparameters(X_train, y_train)
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.random_state,
                n_jobs=-1,
                eval_metric='rmse'
            )
            self.model.fit(X_train, y_train)
            best_params = self.model.get_params()
        
        training_time = (datetime.now() - start_time).total_seconds()
        
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
        
        # Model performance interpretation
        if test_metrics["r2"] > 0.8:
            performance_grade = "Excellent"
        elif test_metrics["r2"] > 0.6:
            performance_grade = "Good"
        elif test_metrics["r2"] > 0.4:
            performance_grade = "Fair"
        else:
            performance_grade = "Needs Improvement"
        
        # Check for overfitting
        r2_diff = train_metrics["r2"] - test_metrics["r2"]
        overfitting_status = "Potential Overfitting" if r2_diff > 0.1 else "Good Generalization"
        
        # Generate visualizations
        prediction_plot = self.generate_prediction_plot(y_test, y_test_pred)
        feature_importance_plot = self.generate_feature_importance_plot()
        time_series_plot = self.generate_time_series_plot(df, y_test, y_test_pred, config)
        
        # Feature importance data
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create comprehensive results
        results = {
            "model_info": {
                "model_type": "XGBoost Regressor",
                "target_column": self.target_column,
                "features": self.feature_columns,
                "training_samples": int(X_train.shape[0]),
                "testing_samples": int(X_test.shape[0]),
                "total_features": len(self.feature_columns),
                "time_aware_split": config.time_aware_split,
                "hyperparameter_tuning": config.tune_hyperparameters,
                "training_time_seconds": training_time,
                "best_parameters": best_params if config.tune_hyperparameters else None
            },
            "performance": {
                "training_metrics": train_metrics,
                "testing_metrics": test_metrics,
                "performance_grade": performance_grade,
                "overfitting_status": overfitting_status,
                "overfitting_score": float(r2_diff),
                "average_prediction_error": f"±{test_metrics['mae']:.0f} ridership units"
            },
            "feature_analysis": {
                "top_10_features": [
                    {
                        "feature": row.feature,
                        "importance": float(row.importance),
                        "rank": idx + 1
                    }
                    for idx, row in feature_importance.head(10).iterrows()
                ],
                "total_features_used": len(self.feature_columns)
            },
            "data_insights": {
                "ridership_statistics": {
                    "mean": float(df[self.target_column].mean()),
                    "std": float(df[self.target_column].std()),
                    "min": float(df[self.target_column].min()),
                    "max": float(df[self.target_column].max()),
                    "median": float(df[self.target_column].median())
                },
                "prediction_range": {
                    "min_predicted": float(y_test_pred.min()),
                    "max_predicted": float(y_test_pred.max()),
                    "mean_predicted": float(y_test_pred.mean())
                }
            },
            "visualizations": {
                "prediction_scatter": f"data:image/png;base64,{prediction_plot}",
                "feature_importance": f"data:image/png;base64,{feature_importance_plot}",
                "time_series_plot": f"data:image/png;base64,{time_series_plot}" if time_series_plot else None
            }
        }
        
        return results
    
    def save_model(self, output_path):
        """Save trained model and preprocessing components"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        model_dict = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'exclude_columns': self.exclude_columns,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_dict, f)

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Transportation Ridership Model Trainer API",
        "description": "Upload CSV files to train XGBoost ridership prediction models",
        "endpoints": {
            "train": "/train-ridership-model/",
            "predict": "/predict-ridership/",
            "download_model": "/download-model/{model_id}",
            "download_results": "/download-results/{model_id}",
            "health": "/health",
            "docs": "/docs"
        },
        "supported_features": [
            "Time-aware data splitting",
            "Hyperparameter tuning",
            "Comprehensive model evaluation",
            "Feature importance analysis",
            "Multiple visualization types",
            "Residuals analysis",
            "Time series plotting"
        ]
    }

@app.post("/train-ridership-model/")
async def train_ridership_model(
    file: UploadFile = File(..., description="CSV file with ridership data"),
    target_column: str = Form("ridership", description="Name of target column"),
    exclude_columns: Optional[str] = Form(None, description="Comma-separated list of columns to exclude"),
    test_size: float = Form(0.2, description="Proportion of data for testing (0.1-0.4)"),
    random_state: int = Form(42, description="Random state for reproducibility"),
    tune_hyperparameters: bool = Form(True, description="Whether to tune hyperparameters"),
    time_aware_split: bool = Form(True, description="Use time-aware train/test split"),
    datetime_column: str = Form("datetime", description="Name of datetime column for time-aware split")
):
    """
    Train XGBoost ridership prediction model on uploaded CSV data.
    
    Returns comprehensive evaluation results including metrics, feature analysis, and visualizations.
    """
    
    # Validate inputs
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    if not (0.1 <= test_size <= 0.4):
        raise HTTPException(status_code=400, detail="Test size must be between 0.1 and 0.4")
    
    # Generate unique IDs for this training session
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_DIR / f"ridership_data_{unique_id}.csv"
    model_path = TEMP_DIR / f"ridership_model_{unique_id}.pkl"
    results_path = TEMP_DIR / f"ridership_results_{unique_id}.json"
    
    try:
        # Save uploaded file
        with open(temp_csv_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load and validate CSV
        try:
            df = pd.read_csv(temp_csv_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{target_column}' not found. Available columns: {list(df.columns)}"
            )
        
        # Parse exclude columns
        exclude_cols = []
        if exclude_columns:
            exclude_cols = [col.strip() for col in exclude_columns.split(',')]
        
        # Create training configuration
        config = TrainingConfig(
            target_column=target_column,
            exclude_columns=exclude_cols,
            test_size=test_size,
            random_state=random_state,
            tune_hyperparameters=tune_hyperparameters,
            time_aware_split=time_aware_split,
            datetime_column=datetime_column
        )
        
        # Initialize and train model
        predictor = RidershipPredictor(target_column=target_column)
        results = predictor.train_and_evaluate(df, config)
        
        # Save trained model
        predictor.save_model(model_path)
        
        # Save results to JSON file
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Add model ID for downloading
        results["model_id"] = unique_id
        results["timestamp"] = datetime.now().isoformat()
        
        # Clean up temporary CSV file
        temp_csv_path.unlink()
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up files in case of error
        for path in [temp_csv_path, model_path, results_path]:
            if path.exists():
                path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/predict-ridership/")
async def predict_ridership(
    data_file: UploadFile = File(..., description="CSV file with data to predict"),
    model_file: UploadFile = File(..., description="Trained model file (.pkl)")
):
    """
    Make ridership predictions using a trained model on new data.
    """
    
    # Validate file types
    if not data_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Data file must be CSV")
    if not model_file.filename.endswith('.pkl'):
        raise HTTPException(status_code=400, detail="Model file must be .pkl")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_DIR / f"predict_data_{unique_id}.csv"
    temp_model_path = TEMP_DIR / f"predict_model_{unique_id}.pkl"
    
    try:
        # Save uploaded files
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
                model_dict = pickle.load(f)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading files: {str(e)}")
        
        # Prepare features using same preprocessing as training
        feature_columns = model_dict['feature_columns']
        label_encoders = model_dict.get('label_encoders', {})
        
        # Apply label encoding to categorical columns
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError as e:
                    # Handle unseen categories
                    df[col] = 0  # Or use a default encoding
        
        # Check for missing features
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )
        
        X = df[feature_columns]
        
        # Make predictions
        model = model_dict['model']
        predictions = model.predict(X)
        
        # Create results with additional info if available
        results = {
            "predictions": predictions.tolist(),
            "prediction_count": len(predictions),
            "prediction_statistics": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
                "median": float(np.median(predictions))
            }
        }
        
        # If actual values are available, calculate metrics
        target_column = model_dict.get('target_column', 'ridership')
        if target_column in df.columns:
            actual_values = df[target_column].values
            mae = mean_absolute_error(actual_values, predictions)
            rmse = np.sqrt(mean_squared_error(actual_values, predictions))
            r2 = r2_score(actual_values, predictions)
            
            results["evaluation_metrics"] = {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2)
            }
            
            results["actual_vs_predicted"] = [
                {"actual": float(actual), "predicted": float(pred)}
                for actual, pred in zip(actual_values, predictions)
            ]
        
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
        
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

@app.get("/download-model/{model_id}")
async def download_model(model_id: str):
    """Download a trained ridership prediction model."""
    model_path = TEMP_DIR / f"ridership_model_{model_id}.pkl"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        path=model_path,
        filename=f"ridership_model_{model_id}.pkl",
        media_type="application/octet-stream"
    )

@app.get("/download-results/{model_id}")
async def download_results(model_id: str):
    """Download the training results JSON file."""
    results_path = TEMP_DIR / f"ridership_results_{model_id}.json"
    
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        path=results_path,
        filename=f"ridership_training_results_{model_id}.json",
        media_type="application/json"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "Transportation Ridership Model Trainer API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)