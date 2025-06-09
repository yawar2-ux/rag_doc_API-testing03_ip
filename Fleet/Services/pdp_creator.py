#!/usr/bin/env python3
"""
Utility functions for Partial Dependence Plot (PDP) analysis.
Contains PDPAnalyzer class and related helper functions used by other modules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import base64
from pathlib import Path
from typing import Optional, List, Tuple
from io import BytesIO
import warnings
from pydantic import BaseModel

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)

class PDPConfig(BaseModel):
    """Configuration for PDP analysis"""
    num_points: int = 50
    top_n_features: Optional[int] = None
    include_classification: bool = True
    include_regression: bool = True
    feature_list: Optional[List[str]] = None

def safe_float_conversion(value):
    """Safely convert value to JSON-compliant float."""
    if pd.isna(value) or np.isinf(value) or abs(value) > 1e308:
        return 0.0
    return float(np.clip(value, -1e308, 1e308))

class PDPAnalyzer:
    """PDP analyzer for machine learning models"""
    
    def __init__(self):
        self.models = None
        self.data = None
        self.feature_names = None
        self.numerical_features = None
        
    def load_data_and_model(self, data_path: str, model_path: str):
        """Load data and models from files"""
        try:
            self.data = pd.read_csv(data_path)
            
            with open(model_path, 'rb') as f:
                self.models = pickle.load(f)
            
            # Handle both dict and direct model formats
            if isinstance(self.models, dict):
                if 'classification_model' in self.models:
                    if hasattr(self.models['classification_model'], 'feature_names_in_'):
                        self.feature_names = list(self.models['classification_model'].feature_names_in_)
                    else:
                        # Fallback: use all numerical columns from data
                        self.feature_names = self.data.select_dtypes(include=[np.number]).columns.tolist()
                else:
                    raise ValueError("No classification model found in the model file")
            else:
                raise ValueError("Model file must contain a dictionary with model components")
            
            # Get numerical features that exist in both data and model
            available_features = [f for f in self.feature_names if f in self.data.columns]
            self.numerical_features = self.data[available_features].select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            
            # Update feature names to only include available ones
            self.feature_names = available_features
            
        except Exception as e:
            raise Exception(f"Error loading data and models: {str(e)}")
    
    def calculate_pdp(
        self,
        model,
        feature: str,
        num_points: int = 50,
        is_classifier: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate partial dependence values for a feature"""
        try:
            X = self.data[self.feature_names].copy()
            
            # Create range of values for the feature
            feature_min = X[feature].min()
            feature_max = X[feature].max()
            feature_values = np.linspace(feature_min, feature_max, num=num_points)
            
            pdp_values = []
            for value in feature_values:
                X_temp = X.copy()
                X_temp[feature] = value
                
                if is_classifier:
                    predictions = model.predict_proba(X_temp)
                    pdp_values.append(np.mean(predictions, axis=0))
                else:
                    predictions = model.predict(X_temp)
                    pdp_values.append(np.mean(predictions))
            
            return feature_values, np.array(pdp_values)
            
        except Exception as e:
            raise Exception(f"Error calculating PDP for feature {feature}: {str(e)}")
    
    def plot_classification_pdp(
        self,
        feature: str,
        num_points: int = 50,
        figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """Generate classification PDP plot and return as base64 string"""
        try:
            model = self.models['classification_model']
            label_encoder = self.models.get('label_encoder')
            
            feature_values, pdp_values = self.calculate_pdp(
                model, feature, num_points, is_classifier=True
            )
            
            plt.figure(figsize=figsize)
            
            # Get class names
            if label_encoder:
                class_names = label_encoder.classes_
            else:
                class_names = [f'Class {i}' for i in range(pdp_values.shape[1])]
            
            # Plot each class
            for i, class_name in enumerate(class_names):
                plt.plot(
                    feature_values,
                    pdp_values[:, i],
                    label=f'{class_name}',
                    linewidth=2
                )
            
            plt.xlabel(feature.replace('_', ' ').title())
            plt.ylabel('Partial Dependence (Probability)')
            plt.title(f'Partial Dependence Plot: {feature.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            raise Exception(f"Error plotting classification PDP: {str(e)}")
    
    def plot_regression_pdp(
        self,
        feature: str,
        num_points: int = 50,
        figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """Generate regression PDP plot and return as base64 string"""
        try:
            if 'regression_model' not in self.models:
                raise ValueError("No regression model found")
                
            model = self.models['regression_model']
            
            feature_values, pdp_values = self.calculate_pdp(
                model, feature, num_points, is_classifier=False
            )
            
            plt.figure(figsize=figsize)
            plt.plot(
                feature_values,
                pdp_values,
                color='blue',
                linewidth=3
            )
            
            plt.xlabel(feature.replace('_', ' ').title())
            plt.ylabel('Partial Dependence (Predicted Value)')
            plt.title(f'Partial Dependence Plot: {feature.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            raise Exception(f"Error plotting regression PDP: {str(e)}")
    
    def analyze_feature_trend(self, values):
        """Analyze the trend in PDP values and return description"""
        if len(values.shape) > 1:
            # For classification (multiple classes)
            trends = []
            for i in range(values.shape[1]):
                class_values = values[:, i]
                if len(class_values) < 2:
                    trends.append("insufficient_data")
                    continue
                    
                diff = np.diff(class_values)
                if np.all(diff >= -1e-10):  # Allow for small numerical errors
                    trends.append("increasing")
                elif np.all(diff <= 1e-10):
                    trends.append("decreasing")
                else:
                    # Check if mostly increasing or decreasing
                    pos_changes = np.sum(diff > 1e-10)
                    neg_changes = np.sum(diff < -1e-10)
                    if pos_changes > neg_changes * 2:
                        trends.append("mostly_increasing")
                    elif neg_changes > pos_changes * 2:
                        trends.append("mostly_decreasing")
                    else:
                        trends.append("non_monotonic")
            return trends
        else:
            # For regression (single line)
            if len(values) < 2:
                return "insufficient_data"
                
            diff = np.diff(values)
            if np.all(diff >= -1e-10):
                return "increasing"
            elif np.all(diff <= 1e-10):
                return "decreasing"
            else:
                pos_changes = np.sum(diff > 1e-10)
                neg_changes = np.sum(diff < -1e-10)
                if pos_changes > neg_changes * 2:
                    return "mostly_increasing"
                elif neg_changes > pos_changes * 2:
                    return "mostly_decreasing"
                else:
                    return "non_monotonic"
    
    def get_feature_importance(self):
        """Get feature importance from models with safe conversion."""
        importance_data = {}
        
        # Classification importance
        if 'classification_model' in self.models:
            clf_importance = self.models['classification_model'].feature_importances_
            importance_data['classification'] = {
                feature: safe_float_conversion(importance) 
                for feature, importance in zip(self.feature_names, clf_importance)
            }
        
        # Regression importance
        if 'regression_model' in self.models:
            reg_importance = self.models['regression_model'].feature_importances_
            importance_data['regression'] = {
                feature: safe_float_conversion(importance) 
                for feature, importance in zip(self.feature_names, reg_importance)
            }
        
        return importance_data
    
    def analyze_multiple_features(self, config: PDPConfig):
        """Analyze multiple features with safe float handling."""
        try:
            # Get feature importance
            importance_data = self.get_feature_importance()
            
            # Determine which features to analyze
            if config.feature_list:
                features_to_analyze = [f for f in config.feature_list if f in self.numerical_features]
            elif config.top_n_features:
                # Use classification importance to select top features
                clf_importance = importance_data.get('classification', {})
                sorted_features = sorted(clf_importance.items(), key=lambda x: x[1], reverse=True)
                features_to_analyze = [f for f, _ in sorted_features[:config.top_n_features] if f in self.numerical_features]
            else:
                features_to_analyze = self.numerical_features
            
            if not features_to_analyze:
                raise ValueError("No valid numerical features found for analysis")
            
            # Analyze each feature
            results = []
            
            for feature in features_to_analyze:
                feature_result = {
                    'feature_name': feature,
                    'feature_stats': {
                        'min': safe_float_conversion(self.data[feature].min()),
                        'max': safe_float_conversion(self.data[feature].max()),
                        'mean': safe_float_conversion(self.data[feature].mean()),
                        'std': safe_float_conversion(self.data[feature].std())
                    }
                }
                
                # Add importance scores with safe conversion
                if 'classification' in importance_data:
                    feature_result['classification_importance'] = safe_float_conversion(
                        importance_data['classification'].get(feature, 0.0)
                    )
                if 'regression' in importance_data:
                    feature_result['regression_importance'] = safe_float_conversion(
                        importance_data['regression'].get(feature, 0.0)
                    )
                
                # Classification PDP
                if config.include_classification and 'classification_model' in self.models:
                    try:
                        # Calculate PDP values for trend analysis
                        _, pdp_values = self.calculate_pdp(
                            self.models['classification_model'], 
                            feature, 
                            config.num_points, 
                            is_classifier=True
                        )
                        
                        # Analyze trends
                        trends = self.analyze_feature_trend(pdp_values)
                        
                        # Generate plot
                        plot_base64 = self.plot_classification_pdp(feature, config.num_points)
                        
                        # Get class names
                        label_encoder = self.models.get('label_encoder')
                        class_names = label_encoder.classes_.tolist() if label_encoder else [f'Class {i}' for i in range(pdp_values.shape[1])]
                        
                        feature_result['classification_analysis'] = {
                            'plot': f"data:image/png;base64,{plot_base64}",
                            'trends': {class_names[i]: trends[i] for i in range(len(trends))},
                            'summary': f"Feature shows {', '.join(set(trends))} trends across classes"
                        }
                    except Exception as e:
                        feature_result['classification_analysis'] = {
                            'error': f"Failed to analyze classification PDP: {str(e)}"
                        }
                
                # Regression PDP
                if config.include_regression and 'regression_model' in self.models:
                    try:
                        # Calculate PDP values for trend analysis
                        _, pdp_values = self.calculate_pdp(
                            self.models['regression_model'], 
                            feature, 
                            config.num_points, 
                            is_classifier=False
                        )
                        
                        # Analyze trend
                        trend = self.analyze_feature_trend(pdp_values)
                        
                        # Generate plot
                        plot_base64 = self.plot_regression_pdp(feature, config.num_points)
                        
                        feature_result['regression_analysis'] = {
                            'plot': f"data:image/png;base64,{plot_base64}",
                            'trend': trend,
                            'summary': f"Feature shows {trend} relationship with target"
                        }
                    except Exception as e:
                        feature_result['regression_analysis'] = {
                            'error': f"Failed to analyze regression PDP: {str(e)}"
                        }
                
                results.append(feature_result)
            
            # Create summary insights
            summary_insights = self.generate_summary_insights(results, importance_data)
            
            return {
                'analysis_config': {
                    'num_points': int(config.num_points),
                    'features_analyzed': int(len(features_to_analyze)),
                    'include_classification': bool(config.include_classification),
                    'include_regression': bool(config.include_regression)
                },
                'feature_analyses': results,
                'summary_insights': summary_insights,
                'feature_importance': importance_data
            }
            
        except Exception as e:
            logger.error(f"Error in multi-feature analysis: {e}")
            raise Exception(f"Error in multi-feature analysis: {str(e)}")
    
    def generate_summary_insights(self, results, importance_data):
        """Generate summary insights from PDP analysis"""
        insights = {
            'top_features': {},
            'trend_patterns': {},
            'key_findings': []
        }
        
        # Top features by importance
        if 'classification' in importance_data:
            clf_sorted = sorted(importance_data['classification'].items(), key=lambda x: x[1], reverse=True)
            insights['top_features']['classification'] = clf_sorted[:5]
        
        if 'regression' in importance_data:
            reg_sorted = sorted(importance_data['regression'].items(), key=lambda x: x[1], reverse=True)
            insights['top_features']['regression'] = reg_sorted[:5]
        
        # Analyze trend patterns
        trend_counts = {
            'increasing': 0,
            'decreasing': 0,
            'non_monotonic': 0,
            'mostly_increasing': 0,
            'mostly_decreasing': 0
        }
        
        for result in results:
            # Count classification trends
            if 'classification_analysis' in result and 'trends' in result['classification_analysis']:
                for trend in result['classification_analysis']['trends'].values():
                    if trend in trend_counts:
                        trend_counts[trend] += 1
            
            # Count regression trends
            if 'regression_analysis' in result and 'trend' in result['regression_analysis']:
                trend = result['regression_analysis']['trend']
                if trend in trend_counts:
                    trend_counts[trend] += 1
        
        insights['trend_patterns'] = trend_counts
        
        # Generate key findings
        if trend_counts['increasing'] > 0:
            insights['key_findings'].append(f"{trend_counts['increasing']} features show increasing relationships")
        if trend_counts['decreasing'] > 0:
            insights['key_findings'].append(f"{trend_counts['decreasing']} features show decreasing relationships")
        if trend_counts['non_monotonic'] > 0:
            insights['key_findings'].append(f"{trend_counts['non_monotonic']} features show complex non-linear relationships")
        
        return insights