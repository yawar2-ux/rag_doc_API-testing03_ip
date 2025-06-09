# Update model_trainer_and_predict.py to include MLflow integration

#!/usr/bin/env python3
"""
Machine learning model training and prediction utilities with MLflow integration.
Contains the FleetMaintenanceModel class and related helper functions.
"""

import os
import uuid
import json
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import base64
import io
import warnings
from pydantic import BaseModel

# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

warnings.filterwarnings('ignore')

# Use consolidated temp directory structure
TEMP_BASE_DIR = Path("temp_fleet_api")
TEMP_BASE_DIR.mkdir(exist_ok=True)
TEMP_DIR = TEMP_BASE_DIR / "models"
TEMP_DIR.mkdir(exist_ok=True)

# MLflow configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class TrainingConfig(BaseModel):
    """Configuration for model training parameters"""
    classification_target: Optional[str] = None
    regression_target: Optional[str] = None
    exclude_columns: Optional[List[str]] = None
    test_size: float = 0.2
    random_state: int = 42
    tune_hyperparameters: bool = True
    log_to_mlflow: bool = True  # New parameter for MLflow logging

def safe_float_conversion(value):
    """Safely convert value to JSON-compliant float."""
    if pd.isna(value) or np.isinf(value) or abs(value) > 1e308:
        return 0.0
    return float(np.clip(value, -1e308, 1e308))

def safe_metric_calculation(y_true, y_pred, metric_func, default_value=0.0):
    """Safely calculate metrics with fallback values."""
    try:
        result = metric_func(y_true, y_pred)
        return safe_float_conversion(result)
    except Exception as e:
        logger.warning(f"Metric calculation failed: {e}")
        return default_value

class FleetMaintenanceModel:
    def __init__(self, classification_target='component_at_risk', regression_target='days_till_breakdown'):
        self.label_encoder = LabelEncoder()
        self.classification_model = None
        self.regression_model = None
        self.classification_target = classification_target
        self.regression_target = regression_target
        self.exclude_columns = ['vehicle_id', 'reading_date']  # Default columns to exclude
        
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """Prepare data for model training without hardcoding feature columns"""
        # Create a list of columns to exclude from features
        exclude_cols = [self.classification_target, self.regression_target] + self.exclude_columns
        
        # Dynamically determine feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Check for remaining categorical columns
        cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            feature_cols = [col for col in feature_cols if col not in cat_cols]
        
        X = df[feature_cols]
        
        # Check if targets exist in the dataframe
        if self.classification_target not in df.columns:
            raise ValueError(f"Classification target '{self.classification_target}' not found in dataset")
        if self.regression_target not in df.columns:
            raise ValueError(f"Regression target '{self.regression_target}' not found in dataset")
        
        y_component = df[self.classification_target]
        y_days = df[self.regression_target]
        
        # Encode the classification target
        y_component_encoded = self.label_encoder.fit_transform(y_component)
        
        # Use stratified split to ensure all classes are represented in train and test sets
        X_train, X_test, y_comp_train, y_comp_test, y_days_train, y_days_test = train_test_split(
            X, y_component_encoded, y_days, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_component_encoded
        )
        
        return X_train, X_test, y_comp_train, y_comp_test, y_days_train, y_days_test, feature_cols

    def tune_classification_model(self, X_train, y_train):
        """Hyperparameter tuning for classification model"""
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            random_state=42,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def tune_regression_model(self, X_train, y_train):
        """Hyperparameter tuning for regression model"""
        param_grid = {
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
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
        return grid_search.best_estimator_

    def generate_roc_curve_image(self, y_test, y_pred_proba):
        """Generate ROC curve for multiclass classification"""
        unique_classes = sorted(np.unique(y_test))
        
        plt.figure(figsize=(12, 8))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in unique_classes:
            if i in unique_classes:
                y_score = y_pred_proba[:, i]
                y_binary = (y_test == i).astype(int)
                
                fpr[i], tpr[i], _ = roc_curve(y_binary, y_score)
                roc_auc[i] = auc(fpr[i], tpr[i])
                
                class_name = self.label_encoder.classes_[i]
                plt.plot(fpr[i], tpr[i], lw=2,
                         label=f'Class {class_name} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Fleet Component Classification')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()

    def generate_regression_plot_image(self, y_test, y_pred):
        """Generate scatter plot for regression results"""
        plt.figure(figsize=(12, 8))
        
        plt.scatter(y_test, y_pred, alpha=0.4, s=30)
        
        max_val = max(max(y_test), max(y_pred))
        min_val = min(min(y_test), min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
        
        plt.xlabel('Actual Days till Breakdown')
        plt.ylabel('Predicted Days till Breakdown')
        plt.title('Fleet Maintenance Regression: Predicted vs Actual')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        plt.text(0.05, 0.95, f'R² Score: {r2:.3f}\nRMSE: {rmse:.3f}', 
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.legend()
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()

    def generate_feature_importance_plot(self, model, feature_names, model_type="Classification"):
        """Generate feature importance plot"""
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        num_features = min(15, len(indices))
        indices = indices[:num_features]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Fleet {model_type} Model: Feature Importance')
        plt.bar(range(len(indices)), importance[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()

    def train_and_evaluate(self, df, config: TrainingConfig):
        """Train both models and evaluate performance with safe float handling"""
        # Update model configuration
        if config.classification_target:
            self.classification_target = config.classification_target
        if config.regression_target:
            self.regression_target = config.regression_target
        if config.exclude_columns:
            self.exclude_columns = config.exclude_columns
        
        # Start MLflow run if logging is enabled
        if config.log_to_mlflow:
            mlflow_run = mlflow.start_run(run_name=f"fleet_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Log training configuration
            mlflow.log_param("classification_target", self.classification_target)
            mlflow.log_param("regression_target", self.regression_target)
            mlflow.log_param("test_size", config.test_size)
            mlflow.log_param("random_state", config.random_state)
            mlflow.log_param("tune_hyperparameters", config.tune_hyperparameters)
            
            # Log tags
            mlflow.set_tags({
                "model_type": "fleet_maintenance",
                "training_type": "multi_model",
                "framework": "xgboost",
                "created_by": "fleet_pipeline"
            })
        
        try:
            # Load and prepare data
            X_train, X_test, y_comp_train, y_comp_test, y_days_train, y_days_test, feature_names = self.prepare_data(
                df, config.test_size, config.random_state
            )
            
            if config.log_to_mlflow:
                mlflow.log_param("training_samples", len(X_train))
                mlflow.log_param("testing_samples", len(X_test))
                mlflow.log_param("feature_count", len(feature_names))
                mlflow.log_param("features", feature_names)
            
            # Train classification model
            start_time_clf = datetime.now()
            if config.tune_hyperparameters:
                self.classification_model = self.tune_classification_model(X_train, y_comp_train)
            else:
                self.classification_model = xgb.XGBClassifier(
                    objective='multi:softprob', random_state=42, n_jobs=-1
                )
                self.classification_model.fit(X_train, y_comp_train)
            clf_execution_time = (datetime.now() - start_time_clf).total_seconds()
            
            # Log classification model
            if config.log_to_mlflow:
                mlflow.xgboost.log_model(
                    self.classification_model, 
                    "classification_model",
                    input_example=X_train.head(5)
                )
                mlflow.log_params({f"clf_{k}": v for k, v in self.classification_model.get_params().items()})
                mlflow.log_metric("classification_training_time", clf_execution_time)
            
            # Train regression model
            start_time_reg = datetime.now()
            mask_train = y_days_train != -1
            mask_test = y_days_test != -1
            
            if mask_train.sum() > 0:
                if config.tune_hyperparameters:
                    self.regression_model = self.tune_regression_model(
                        X_train[mask_train], y_days_train[mask_train]
                    )
                else:
                    self.regression_model = xgb.XGBRegressor(
                        objective='reg:squarederror', random_state=42, n_jobs=-1
                    )
                    self.regression_model.fit(X_train[mask_train], y_days_train[mask_train])
                reg_execution_time = (datetime.now() - start_time_reg).total_seconds()
                
                # Log regression model
                if config.log_to_mlflow:
                    mlflow.xgboost.log_model(
                        self.regression_model, 
                        "regression_model",
                        input_example=X_train[mask_train].head(5)
                    )
                    mlflow.log_params({f"reg_{k}": v for k, v in self.regression_model.get_params().items()})
                    mlflow.log_metric("regression_training_time", reg_execution_time)
            else:
                reg_execution_time = 0
            
            # Make predictions
            y_comp_pred = self.classification_model.predict(X_test)
            y_comp_pred_proba = self.classification_model.predict_proba(X_test)
            
            # Classification metrics with safe calculation
            conf_matrix = confusion_matrix(y_comp_test, y_comp_pred).tolist()
            accuracy = safe_metric_calculation(y_comp_test, y_comp_pred, accuracy_score)
            precision = safe_metric_calculation(y_comp_test, y_comp_pred, 
                                               lambda y_t, y_p: precision_score(y_t, y_p, average='weighted'))
            recall = safe_metric_calculation(y_comp_test, y_comp_pred, 
                                           lambda y_t, y_p: recall_score(y_t, y_p, average='weighted'))
            f1 = safe_metric_calculation(y_comp_test, y_comp_pred, 
                                       lambda y_t, y_p: f1_score(y_t, y_p, average='weighted'))
            
            # Log classification metrics
            if config.log_to_mlflow:
                mlflow.log_metric("classification_accuracy", accuracy)
                mlflow.log_metric("classification_precision", precision)
                mlflow.log_metric("classification_recall", recall)
                mlflow.log_metric("classification_f1", f1)
            
            # Safely calculate log_loss
            unique_classes = sorted(set(np.concatenate([y_comp_train, y_comp_test])))
            try:
                loss = log_loss(y_comp_test, y_comp_pred_proba, labels=unique_classes)
                loss = safe_float_conversion(loss)
                if config.log_to_mlflow:
                    mlflow.log_metric("classification_log_loss", loss)
            except Exception as e:
                logger.warning(f"Log loss calculation failed: {e}")
                loss = None
            
            # Generate classification visualizations
            roc_curve_base64 = self.generate_roc_curve_image(y_comp_test, y_comp_pred_proba)
            clf_importance_base64 = self.generate_feature_importance_plot(
                self.classification_model, feature_names, "Classification"
            )
            
            # Initialize regression results
            regression_results = {"status": "not_trained"}
            regression_plot_base64 = ""
            reg_importance_base64 = ""
            
            # Calculate regression metrics if model was trained
            if hasattr(self, 'regression_model') and self.regression_model is not None and mask_test.sum() > 0:
                y_days_pred = self.regression_model.predict(X_test[mask_test])
                
                mse = safe_metric_calculation(y_days_test[mask_test], y_days_pred, mean_squared_error)
                rmse = safe_float_conversion(np.sqrt(mse))
                mae = safe_metric_calculation(y_days_test[mask_test], y_days_pred, mean_absolute_error)
                r2 = safe_metric_calculation(y_days_test[mask_test], y_days_pred, r2_score)
                
                # Log regression metrics
                if config.log_to_mlflow:
                    mlflow.log_metric("regression_mse", mse)
                    mlflow.log_metric("regression_rmse", rmse)
                    mlflow.log_metric("regression_mae", mae)
                    mlflow.log_metric("regression_r2", r2)
                
                regression_plot_base64 = self.generate_regression_plot_image(
                    y_days_test[mask_test], y_days_pred
                )
                reg_importance_base64 = self.generate_feature_importance_plot(
                    self.regression_model, feature_names, "Regression"
                )
                
                regression_results = {
                    "status": "trained",
                    "performance": {
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                        "execution_time": safe_float_conversion(reg_execution_time)
                    }
                }
            
            # Log visualizations as artifacts
            if config.log_to_mlflow:
                # Save visualizations temporarily and log as artifacts
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save ROC curve
                    roc_path = os.path.join(temp_dir, "roc_curve.png")
                    with open(roc_path, "wb") as f:
                        f.write(base64.b64decode(roc_curve_base64))
                    mlflow.log_artifact(roc_path, "visualizations")
                    
                    # Save classification feature importance
                    clf_imp_path = os.path.join(temp_dir, "classification_feature_importance.png")
                    with open(clf_imp_path, "wb") as f:
                        f.write(base64.b64decode(clf_importance_base64))
                    mlflow.log_artifact(clf_imp_path, "visualizations")
                    
                    # Save regression plots if available
                    if regression_plot_base64:
                        reg_plot_path = os.path.join(temp_dir, "regression_scatter.png")
                        with open(reg_plot_path, "wb") as f:
                            f.write(base64.b64decode(regression_plot_base64))
                        mlflow.log_artifact(reg_plot_path, "visualizations")
                    
                    if reg_importance_base64:
                        reg_imp_path = os.path.join(temp_dir, "regression_feature_importance.png")
                        with open(reg_imp_path, "wb") as f:
                            f.write(base64.b64decode(reg_importance_base64))
                        mlflow.log_artifact(reg_imp_path, "visualizations")
            
            # Create results with safe conversions
            results = {
                "model_info": {
                    "classification_target": self.classification_target,
                    "regression_target": self.regression_target,
                    "features": feature_names,
                    "target_classes": self.label_encoder.classes_.tolist(),
                    "training_samples": int(X_train.shape[0]),
                    "testing_samples": int(X_test.shape[0])
                },
                "results": {
                    "classification": {
                        "performance": {
                            "confusion_matrix": conf_matrix,
                            "class_mapping": {i: cls for i, cls in enumerate(self.label_encoder.classes_)},
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "log_loss": loss,
                            "execution_time": safe_float_conversion(clf_execution_time)
                        }
                    },
                    "regression": regression_results
                },
                "visualizations": {
                    "classification_roc": f"data:image/png;base64,{roc_curve_base64}",
                    "classification_importance": f"data:image/png;base64,{clf_importance_base64}",
                    "regression_scatter": f"data:image/png;base64,{regression_plot_base64}",
                    "regression_importance": f"data:image/png;base64,{reg_importance_base64}"
                },
                "mlflow_info": {
                    "logged": config.log_to_mlflow,
                    "run_id": mlflow_run.info.run_id if config.log_to_mlflow else None,
                    "experiment_id": mlflow_run.info.experiment_id if config.log_to_mlflow else None,
                    "tracking_uri": MLFLOW_TRACKING_URI if config.log_to_mlflow else None
                }
            }
            
            return results
            
        finally:
            # End MLflow run
            if config.log_to_mlflow:
                mlflow.end_run()

    def save_models(self, output_path):
        """Save trained models to disk"""
        if self.classification_model is not None:
            models_dict = {
                'classification_model': self.classification_model,
                'label_encoder': self.label_encoder,
                'classification_target': self.classification_target,
                'regression_target': self.regression_target,
                'exclude_columns': self.exclude_columns
            }
            
            if self.regression_model is not None:
                models_dict['regression_model'] = self.regression_model
            
            with open(output_path, 'wb') as f:
                pickle.dump(models_dict, f)
        else:
            raise ValueError("Models not trained yet. Please train the models first.")