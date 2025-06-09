#!/usr/bin/env python3
"""
Partial Dependency Plot Creator Module
Creates partial dependency plots to show the effect of individual features on ridership prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, List, Union, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

warnings.filterwarnings('ignore')

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
        
        # Available features for PDP analysis
        self.available_features = []
        
    def train_model(self, df: pd.DataFrame, target_col: str = 'ridership', 
                   test_size: float = 0.2, model_type: str = None) -> Dict[str, Any]:
        """
        Train model for partial dependency analysis.
        
        Args:
            df: Cleaned DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion for test split
            model_type: Type of model ('decision_tree' or 'random_forest')
            
        Returns:
            Dictionary with training results and metrics
        """
        if model_type:
            self.model_type = model_type
            
        try:
            # Prepare features and target
            exclude_cols = [target_col, 'datetime', 'date', 'time']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols]
            y = df[target_col]
            
            self.feature_columns = feature_cols
            self.available_features = feature_cols.copy()
            
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
    
    def validate_features(self, features: List[str]) -> Dict[str, Any]:
        """
        Validate requested features for PDP analysis.
        
        Args:
            features: List of feature names to validate
            
        Returns:
            Dictionary with validation results
        """
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
    
    def create_single_feature_pdp(self, feature: str, grid_resolution: int = 100, 
                                 figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Create partial dependency plot for a single feature.
        
        Args:
            feature: Feature name to analyze
            grid_resolution: Number of points in the grid
            figsize: Figure size (width, height)
            
        Returns:
            Base64 encoded plot image
        """
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
            
            # Add confidence interval if available
            if hasattr(pd_result, 'individual') and pd_result['individual'] is not None:
                # Calculate percentiles for confidence band
                percentiles = np.percentile(pd_result['individual'][0], [10, 90], axis=0)
                ax.fill_between(pd_result['grid_values'][0], 
                              percentiles[0], percentiles[1], 
                              alpha=0.3, color='#1f77b4')
            
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
    
    def create_interaction_pdp(self, features: List[str], grid_resolution: int = 50,
                              figsize: Tuple[int, int] = (12, 8)) -> str:
        """
        Create 2D partial dependency plot for feature interactions.
        
        Args:
            features: List of exactly 2 feature names
            grid_resolution: Number of points in each dimension
            figsize: Figure size (width, height)
            
        Returns:
            Base64 encoded plot image
        """
        if len(features) != 2:
            raise ValueError("Interaction PDP requires exactly 2 features")
        
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        for feature in features:
            if feature not in self.available_features:
                raise ValueError(f"Feature '{feature}' not found in available features")
        
        try:
            # Get feature indices
            feature_indices = [self.feature_columns.index(f) for f in features]
            
            # Compute partial dependence
            pd_result = partial_dependence(
                self.model,
                self.X_train,
                features=[feature_indices],
                grid_resolution=grid_resolution,
                kind='average'
            )
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create 2D contour plot
            XX, YY = np.meshgrid(pd_result['grid_values'][0], pd_result['grid_values'][1])
            Z = pd_result['average'][0].T
            
            # Create filled contour plot
            contour_filled = ax.contourf(XX, YY, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
            
            # Add contour lines
            contour_lines = ax.contour(XX, YY, Z, levels=10, colors='black', alpha=0.4, linewidths=0.5)
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')
            
            # Add colorbar
            cbar = plt.colorbar(contour_filled, ax=ax, shrink=0.8)
            cbar.set_label('Partial Dependence\n(Change in Predicted Ridership)', 
                          fontsize=11, fontweight='bold')
            
            # Customize plot
            ax.set_xlabel(f'{features[0]}', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{features[1]}', fontsize=12, fontweight='bold')
            ax.set_title(f'2D Partial Dependency Plot: {features[0]} vs {features[1]}', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Add subtitle with explanation
            plt.figtext(0.5, 0.02, 
                       f'Shows how the interaction between {features[0]} and {features[1]} affects ridership predictions.',
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
            raise Exception(f"Failed to create interaction PDP for {features}: {str(e)}")
    
    def create_multiple_pdp(self, features: List[str], grid_resolution: int = 100,
                           figsize: Tuple[int, int] = (15, 10)) -> str:
        """
        Create multiple PDP plots in a grid layout.
        
        Args:
            features: List of feature names (max 9 features)
            grid_resolution: Number of points in the grid
            figsize: Figure size (width, height)
            
        Returns:
            Base64 encoded plot image
        """
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
    
    def analyze_feature_effects(self, features: List[str], grid_resolution: int = 100) -> Dict[str, Any]:
        """
        Analyze feature effects using partial dependency.
        
        Args:
            features: List of features to analyze
            grid_resolution: Number of points in the grid
            
        Returns:
            Dictionary with feature effect analysis
        """
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

def create_pdp_analysis(df: pd.DataFrame, features: List[str], 
                       target_col: str = 'ridership', model_type: str = 'decision_tree',
                       plot_type: str = 'single') -> Dict[str, Any]:
    """
    Convenience function to perform complete PDP analysis.
    
    Args:
        df: Cleaned DataFrame with features and target
        features: List of features to analyze
        target_col: Name of target column
        model_type: Type of model ('decision_tree' or 'random_forest')
        plot_type: Type of plot ('single', 'multiple', or 'interaction')
        
    Returns:
        Dictionary containing complete PDP analysis results
    """
    try:
        # Initialize analyzer
        analyzer = PartialDependencyAnalyzer(model_type=model_type)
        
        # Train model
        train_result = analyzer.train_model(df, target_col=target_col, model_type=model_type)
        if train_result["status"] != "success":
            return train_result
        
        # Create plots based on type
        if plot_type == 'multiple' and len(features) > 1:
            plot_base64 = analyzer.create_multiple_pdp(features)
        elif plot_type == 'interaction' and len(features) == 2:
            plot_base64 = analyzer.create_interaction_pdp(features)
        else:  # single feature plots
            if len(features) > 1:
                plot_base64 = analyzer.create_multiple_pdp(features)
            else:
                plot_base64 = analyzer.create_single_feature_pdp(features[0])
        
        # Analyze feature effects
        effects_analysis = analyzer.analyze_feature_effects(features)
        
        return {
            "status": "success",
            "model_training": train_result,
            "plot_data": f"data:image/png;base64,{plot_base64}",
            "feature_effects": effects_analysis,
            "plot_info": {
                "plot_type": plot_type,
                "features_plotted": features,
                "model_used": model_type
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create PDP analysis: {str(e)}"
        }

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python partial_dependency.py <cleaned_data_path> <feature1> [feature2] ...")
        sys.exit(1)
    
    data_path = sys.argv[1]
    features = sys.argv[2:]
    
    try:
        # Load cleaned data
        df = pd.read_csv(data_path)
        
        # Determine plot type
        plot_type = 'interaction' if len(features) == 2 else 'multiple' if len(features) > 1 else 'single'
        
        # Perform PDP analysis
        result = create_pdp_analysis(df, features, plot_type=plot_type)
        
        if result.get("status") == "success":
            print("PDP analysis completed successfully!")
            
            model_info = result["model_training"]["model_info"]
            print(f"Model: {model_info['model_type']}")
            print(f"R² score: {result['model_training']['performance']['testing_metrics']['r2']:.3f}")
            
            if result["feature_effects"]["status"] == "success":
                print(f"\nFeature Effects Analysis:")
                for effect in result["feature_effects"]["feature_effects"][:3]:
                    print(f"  {effect['feature']}: {effect['trend']} trend, range={effect['effect_range']:.1f}")
        else:
            print(f"Error: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {str(e)}")