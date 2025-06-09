#!/usr/bin/env python3
"""
Model Inference Module for Ridership Prediction
Handles loading trained models and making predictions on new data.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')

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
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load trained model from pickle file.
        
        Args:
            model_path: Path to the trained model pickle file
            
        Returns:
            Dictionary with loading status and model info
        """
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
                "feature_count": len(self.feature_columns),
                "features": self.feature_columns
            }
            
        except Exception as e:
            self.is_loaded = False
            return {
                "status": "error",
                "message": f"Failed to load model: {str(e)}"
            }
    
    def validate_input_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input DataFrame for prediction.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
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
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
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
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make ridership predictions on input data.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Dictionary containing predictions and metadata
        """
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
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                
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
    
    def predict_single(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single instance.
        
        Args:
            feature_dict: Dictionary containing feature values
            
        Returns:
            Dictionary containing single prediction
        """
        # Convert single instance to DataFrame
        df = pd.DataFrame([feature_dict])
        
        # Use regular predict method
        result = self.predict(df)
        
        if result["status"] == "success":
            result["single_prediction"] = result["predictions"][0]
            
        return result
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance from the loaded model.
        
        Returns:
            Dictionary containing feature importance information
        """
        if not self.is_loaded:
            return {
                "status": "error",
                "message": "Model not loaded"
            }
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return {
                    "status": "success",
                    "feature_importance": [
                        {"feature": str(row.feature), "importance": float(row.importance), "rank": idx + 1}
                        for idx, row in importance_df.iterrows()
                    ]
                }
            else:
                return {
                    "status": "error",
                    "message": "Model does not support feature importance"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get feature importance: {str(e)}"
            }

# Utility function for standalone usage
def load_and_predict(model_path: str, data_path: str) -> Dict[str, Any]:
    """
    Convenience function to load model and make predictions in one call.
    
    Args:
        model_path: Path to trained model pickle file
        data_path: Path to CSV file with test data
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Load data
        df = pd.read_csv(data_path)
        
        # Initialize inference engine
        inference = RidershipInference()
        
        # Load model
        load_result = inference.load_model(model_path)
        if load_result["status"] != "success":
            return load_result
        
        # Make predictions
        prediction_result = inference.predict(df)
        
        # Combine results
        return {
            "model_info": load_result,
            "predictions": prediction_result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load and predict: {str(e)}"
        }

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python model_inference.py <model_path> <test_data_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    
    result = load_and_predict(model_path, data_path)
    
    if result.get("predictions", {}).get("status") == "success":
        predictions = result["predictions"]["predictions"]
        print(f"Predictions completed successfully!")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Mean prediction: {np.mean(predictions):.2f}")
        print(f"Prediction range: {np.min(predictions):.2f} - {np.max(predictions):.2f}")
        
        if "evaluation_metrics" in result["predictions"]:
            metrics = result["predictions"]["evaluation_metrics"]
            print(f"R² Score: {metrics['r2']:.3f}")
            print(f"RMSE: {metrics['rmse']:.2f}")
            print(f"MAE: {metrics['mae']:.2f}")
    else:
        print(f"Error: {result.get('message', 'Unknown error')}")