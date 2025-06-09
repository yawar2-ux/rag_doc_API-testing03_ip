#!/usr/bin/env python3
"""
SHAP Bee Swarm Plot Creator Module
Creates SHAP explanations and bee swarm plots using Decision Tree for ridership prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import base64
import io
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, Optional

warnings.filterwarnings('ignore')

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
    
    def train_decision_tree(self, df: pd.DataFrame, target_col: str = 'ridership', test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train Decision Tree model on cleaned data.
        
        Args:
            df: Cleaned DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion for test split
            
        Returns:
            Dictionary with training results and metrics
        """
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
    
    def compute_shap_values(self, background_samples: int = 100) -> Dict[str, Any]:
        """
        Compute SHAP values using TreeExplainer.
        
        Args:
            background_samples: Number of background samples for explainer
            
        Returns:
            Dictionary with SHAP computation results
        """
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
    
    def create_beeswarm_plot(self, max_display: int = 15, figsize: tuple = (12, 8)) -> str:
        """
        Create SHAP bee swarm plot and return as base64 string.
        
        Args:
            max_display: Maximum number of features to display
            figsize: Figure size (width, height)
            
        Returns:
            Base64 encoded plot image
        """
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
    
    def create_summary_plot(self, plot_type: str = "bar", max_display: int = 15, figsize: tuple = (10, 6)) -> str:
        """
        Create SHAP summary plot and return as base64 string.
        
        Args:
            plot_type: Type of plot ("bar" or "dot")
            max_display: Maximum number of features to display
            figsize: Figure size (width, height)
            
        Returns:
            Base64 encoded plot image
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        try:
            plt.figure(figsize=figsize)
            
            if plot_type == "bar":
                # Bar plot showing mean absolute SHAP values
                shap.plots.bar(self.shap_values, max_display=max_display, show=False)
                plt.title('SHAP Feature Importance - Average Impact Magnitude', 
                         fontsize=14, fontweight='bold')
            else:
                # Dot plot (similar to beeswarm but simplified)
                shap.summary_plot(self.shap_values.values, self.X_sample, 
                                plot_type="dot", max_display=max_display, show=False)
                plt.title('SHAP Summary Plot - Feature Impact Distribution', 
                         fontsize=14, fontweight='bold')
            
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
            raise Exception(f"Failed to create summary plot: {str(e)}")
    
    def get_feature_importance_data(self, top_n: int = 15) -> Dict[str, Any]:
        """
        Get feature importance data from SHAP values.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance information
        """
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
    
    def create_comprehensive_analysis(self, max_display: int = 15) -> Dict[str, Any]:
        """
        Create comprehensive SHAP analysis with multiple visualizations.
        
        Args:
            max_display: Maximum number of features to display in plots
            
        Returns:
            Dictionary containing all analysis results and visualizations
        """
        try:
            # Compute SHAP values if not already done
            if self.shap_values is None:
                shap_result = self.compute_shap_values()
                if shap_result["status"] != "success":
                    return shap_result
            
            # Create visualizations
            beeswarm_plot = self.create_beeswarm_plot(max_display=max_display)
            bar_plot = self.create_summary_plot(plot_type="bar", max_display=max_display)
            
            # Get feature importance data
            feature_importance = self.get_feature_importance_data(top_n=max_display)
            
            return {
                "status": "success",
                "message": "Comprehensive SHAP analysis completed",
                "visualizations": {
                    "beeswarm_plot": f"data:image/png;base64,{beeswarm_plot}",
                    "feature_importance_bar": f"data:image/png;base64,{bar_plot}"
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

def analyze_ridership_shap(df: pd.DataFrame, target_col: str = 'ridership', 
                          max_display: int = 15) -> Dict[str, Any]:
    """
    Convenience function to perform complete SHAP analysis on ridership data.
    
    Args:
        df: Cleaned DataFrame with features and target
        target_col: Name of target column
        max_display: Maximum number of features to display
        
    Returns:
        Dictionary containing complete SHAP analysis results
    """
    try:
        # Initialize analyzer
        analyzer = SHAPAnalyzer()
        
        # Train model
        train_result = analyzer.train_decision_tree(df, target_col=target_col)
        if train_result["status"] != "success":
            return train_result
        
        # Perform comprehensive analysis
        analysis_result = analyzer.create_comprehensive_analysis(max_display=max_display)
        
        # Combine results
        return {
            "model_training": train_result,
            "shap_analysis": analysis_result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to analyze ridership SHAP: {str(e)}"
        }

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python shap_analysis.py <cleaned_data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    try:
        # Load cleaned data
        df = pd.read_csv(data_path)
        
        # Perform SHAP analysis
        result = analyze_ridership_shap(df)
        
        if result.get("shap_analysis", {}).get("status") == "success":
            print("SHAP analysis completed successfully!")
            
            model_info = result["model_training"]["model_info"]
            print(f"Model trained on {model_info['training_samples']} samples")
            print(f"Model R² score: {result['model_training']['performance']['testing_metrics']['r2']:.3f}")
            
            analysis_info = result["shap_analysis"]["analysis_info"]
            print(f"SHAP analysis on {analysis_info['explained_samples']} samples")
            print(f"Top {analysis_info['max_features_displayed']} features analyzed")
            
            # Feature importance
            top_features = result["shap_analysis"]["feature_analysis"]["feature_importance"][:5]
            print("\nTop 5 most important features:")
            for feature in top_features:
                print(f"  {feature['rank']}. {feature['feature']}: {feature['percentage']:.1f}%")
        else:
            print(f"Error: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {str(e)}")