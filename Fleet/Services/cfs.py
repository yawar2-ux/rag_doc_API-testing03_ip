#!/usr/bin/env python3
"""
Counterfactual explanations utilities for fleet maintenance.
Contains the FleetMaintenanceExplainer class and related helper functions.
"""

import os
import uuid
import json
import pickle
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
import random
import warnings
from pydantic import BaseModel

warnings.filterwarnings('ignore')

class CounterfactualConfig(BaseModel):
    """Configuration for counterfactual analysis"""
    target_component: str = "None"  # Target component to explain (usually "None" for no risk)
    num_counterfactuals: int = 3
    similarity_threshold: float = 0.1
    include_synthetic: bool = True

def safe_float_for_cf(value):
    """Convert value to JSON-safe float for counterfactual analysis."""
    if pd.isna(value) or np.isinf(value) or abs(value) > 1e308:
        return 0.0
    return float(np.clip(value, -1e308, 1e308))

class FleetMaintenanceExplainer:
    """Class to generate counterfactual explanations for shifting to target risk category"""
    
    def __init__(self):
        self.model_dict = None
        self.df = None
        self.feature_names = None
        self.continuous_features = []
        self.classification_model = None
        self.label_encoder = None
        self.regression_model = None
        
    def load_model_from_file(self, model_path):
        """Load the model from pickle file"""
        with open(model_path, 'rb') as f:
            self.model_dict = pickle.load(f)
            
        # Extract components from the model
        if isinstance(self.model_dict, dict):
            self.classification_model = self.model_dict.get('classification_model')
            self.label_encoder = self.model_dict.get('label_encoder')
            self.regression_model = self.model_dict.get('regression_model')
        else:
            # Handle direct model object
            self.classification_model = self.model_dict
            
        if hasattr(self.classification_model, 'feature_names_in_'):
            self.feature_names = list(self.classification_model.feature_names_in_)
        else:
            raise ValueError("Could not find feature names in the model")
    
    def load_data(self, data_path):
        """Load data from CSV file"""
        self.df = pd.read_csv(data_path)
        
        # Identify continuous features
        if self.feature_names:
            available_features = [f for f in self.feature_names if f in self.df.columns]
            self.continuous_features = self.df[available_features].select_dtypes(
                include=['int64', 'float64']).columns.tolist()
    
    def get_available_components(self):
        """Get list of available components at risk"""
        if 'component_at_risk' in self.df.columns:
            return sorted(self.df['component_at_risk'].unique())
        return []
    
    def get_instances_by_component(self, component, limit=10):
        """Get instances with specified component at risk"""
        if 'component_at_risk' not in self.df.columns:
            return pd.DataFrame()
            
        filtered_df = self.df[self.df['component_at_risk'] == component].copy()
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
        
        # Add useful display columns if they exist
        display_cols = ['vehicle_id', 'vehicle_type', 'make', 'model', 'mileage', 'vehicle_age', 'days_till_breakdown']
        available_display_cols = [col for col in display_cols if col in filtered_df.columns]
        
        # Return limited results with display columns
        result_df = filtered_df.head(limit).copy()
        if available_display_cols:
            result_df = result_df[available_display_cols + ['component_at_risk']].copy()
        
        return result_df
    
    def create_vehicle_identifier(self, instance):
        """Create a user-friendly vehicle identifier string"""
        vehicle_info = []
        
        if 'vehicle_id' in instance.columns:
            vid = instance['vehicle_id'].values[0]
            vehicle_info.append(f"Vehicle {vid}")
        
        if 'vehicle_type' in instance.columns:
            vehicle_type = instance['vehicle_type'].values[0]
            if vehicle_type and pd.notna(vehicle_type):
                vehicle_info.append(str(vehicle_type))
        
        if 'make' in instance.columns and 'model' in instance.columns:
            make = instance['make'].values[0]
            model = instance['model'].values[0]
            if make and model and pd.notna(make) and pd.notna(model):
                vehicle_info.append(f"{make} {model}")
        
        return " - ".join(vehicle_info) if vehicle_info else "Vehicle instance"
    
    def generate_counterfactuals_to_target(self, instance, config: CounterfactualConfig):
        """Generate counterfactuals that change the prediction to target component"""
        # Get original component at risk
        orig_component = instance['component_at_risk'].values[0]
        vehicle_identifier = self.create_vehicle_identifier(instance)
        
        # Get features for prediction
        X_orig = instance[self.feature_names].copy()
        
        # Preserve vehicle identity columns
        identity_cols = ['vehicle_id', 'vehicle_type', 'make', 'model', 'route_type']
        identity_values = {}
        for col in identity_cols:
            if col in instance.columns:
                identity_values[col] = instance[col].values[0]
        
        # Find target component examples in the dataset
        target_instances = self.df[self.df['component_at_risk'] == config.target_component].copy()
        
        counterfactuals = []
        
        # Strategy 1: Find similar instances with target component
        if not target_instances.empty and len(self.continuous_features) > 0:
            # Calculate similarity to original instance
            for feature in self.continuous_features:
                if feature in target_instances.columns:
                    target_instances[f'diff_{feature}'] = abs(
                        target_instances[feature] - X_orig[feature].values[0]
                    )
            
            # Calculate overall similarity score
            diff_cols = [f'diff_{feature}' for feature in self.continuous_features 
                        if f'diff_{feature}' in target_instances.columns]
            if diff_cols:
                target_instances['similarity_score'] = target_instances[diff_cols].sum(axis=1)
                target_instances = target_instances.sort_values('similarity_score')
                
                # Get top similar instances
                similar_instances = target_instances.head(config.num_counterfactuals)
                
                for _, sim_instance in similar_instances.iterrows():
                    counterfactual = sim_instance.copy()
                    
                    # Transfer identity columns from original instance
                    for col, value in identity_values.items():
                        if col in counterfactual.index:
                            counterfactual[col] = value
                    
                    counterfactuals.append(counterfactual)
        
        # Strategy 2: Create synthetic counterfactuals if needed
        remaining = config.num_counterfactuals - len(counterfactuals)
        
        if remaining > 0 and config.include_synthetic and not target_instances.empty:
            # Get statistics of target instances
            target_mean = target_instances[self.continuous_features].mean()
            target_std = target_instances[self.continuous_features].std()
            
            # Replace NaN values with dataset statistics
            for feature in self.continuous_features:
                if pd.isna(target_mean[feature]) or pd.isna(target_std[feature]):
                    target_mean[feature] = self.df[feature].mean()
                    target_std[feature] = self.df[feature].std()
            
            for i in range(remaining):
                # Create synthetic counterfactual
                cf_instance = X_orig.copy()
                cf_full = instance.copy()
                
                # Modify features towards target distribution
                for feature in self.continuous_features:
                    if feature in cf_instance.columns:
                        orig_value = cf_instance[feature].values[0]
                        target_value = target_mean[feature]
                        
                        # Add controlled randomness
                        noise = random.uniform(-0.5, 0.5) * target_std[feature]
                        new_value = orig_value + 0.5 * (target_value - orig_value) + noise
                        
                        cf_instance[feature] = new_value
                        cf_full[feature] = new_value
                
                # Check if model predicts target component
                try:
                    cf_pred = self.classification_model.predict(cf_instance)
                    if self.label_encoder:
                        cf_pred_name = self.label_encoder.inverse_transform(cf_pred)[0]
                    else:
                        cf_pred_name = cf_pred[0]
                    
                    if cf_pred_name == config.target_component:
                        cf_full['component_at_risk'] = config.target_component
                        counterfactuals.append(cf_full.iloc[0])
                except Exception as e:
                    continue
        
        return pd.DataFrame(counterfactuals) if counterfactuals else pd.DataFrame()
    
    def generate_actionable_recommendation(self, feature, change_direction, avg_change):
        """Generate realistic, actionable maintenance recommendations"""
        
        # Dictionary of realistic maintenance actions for different features
        maintenance_actions = {
            # Mileage and usage related
            'mileage': {
                'increase': "Monitor vehicle usage patterns and plan for increased maintenance frequency",
                'decrease': "This vehicle shows high mileage risk - implement preventive maintenance schedule"
            },
            'vehicle_age': {
                'increase': "Vehicle aging detected - increase inspection frequency and replace aging components",
                'decrease': "Focus on preventive maintenance to maintain vehicle condition despite age"
            },
            'days_since_last_service': {
                'increase': "Schedule immediate service appointment - vehicle is overdue for maintenance",
                'decrease': "Maintain current service schedule and monitor for early warning signs"
            },
            
            # Engine related
            'engine_oil_level': {
                'increase': "Check and top up engine oil to recommended levels",
                'decrease': "Investigate oil consumption issues and check for leaks"
            },
            'engine_oil_pressure': {
                'increase': "Check oil pump and oil filter, replace if necessary",
                'decrease': "Investigate high oil pressure - check for blockages or viscosity issues"
            },
            'engine_temperature': {
                'increase': "Check cooling system - radiator, coolant levels, and thermostat",
                'decrease': "Monitor engine warm-up process and check thermostat operation"
            },
            'engine_rpm': {
                'increase': "Check idle speed settings and engine load conditions",
                'decrease': "Inspect for engine performance issues affecting RPM"
            },
            'engine_load': {
                'increase': "Reduce vehicle load or check for engine performance degradation",
                'decrease': "Monitor engine efficiency and check for optimal performance"
            },
            
            # Transmission related
            'transmission_fluid_level': {
                'increase': "Top up transmission fluid to recommended levels",
                'decrease': "Check for transmission fluid leaks and investigate consumption"
            },
            'transmission_temperature': {
                'increase': "Check transmission cooling system and fluid condition",
                'decrease': "Monitor transmission operation and fluid circulation"
            },
            'gear_shifts': {
                'increase': "Check transmission programming and driving patterns",
                'decrease': "Investigate transmission efficiency and shift quality"
            },
            
            # Brake system
            'brake_fluid_level': {
                'increase': "Top up brake fluid and check for leaks in brake system",
                'decrease': "Investigate brake fluid consumption and system integrity"
            },
            'brake_pad_thickness': {
                'increase': "Replace brake pads with new ones meeting specifications",
                'decrease': "Inspect brake pads for unusual wear patterns"
            },
            'brake_temperature': {
                'increase': "Check brake system for overheating - inspect pads, rotors, and calipers",
                'decrease': "Monitor brake cooling and ventilation systems"
            },
            
            # Electrical system
            'battery_voltage': {
                'increase': "Test charging system, replace battery or alternator if needed",
                'decrease': "Check for electrical drain and battery condition"
            },
            'alternator_output': {
                'increase': "Test and possibly replace alternator or voltage regulator",
                'decrease': "Check alternator belt tension and electrical connections"
            },
            
            # Tire and suspension
            'tire_pressure': {
                'increase': "Inflate tires to manufacturer recommended pressure",
                'decrease': "Check for tire pressure leaks and valve condition"
            },
            'tire_tread_depth': {
                'increase': "Replace tires with adequate tread depth for safety",
                'decrease': "Inspect for uneven tire wear patterns"
            },
            'suspension_height': {
                'increase': "Check suspension components and replace worn parts",
                'decrease': "Inspect suspension for proper operation and alignment"
            },
            
            # Fuel system
            'fuel_level': {
                'increase': "Maintain adequate fuel levels and check fuel gauge accuracy",
                'decrease': "Monitor fuel consumption patterns for efficiency"
            },
            'fuel_pressure': {
                'increase': "Check fuel pump, fuel filter, and fuel lines",
                'decrease': "Inspect fuel pressure regulator and fuel system integrity"
            },
            'fuel_efficiency': {
                'increase': "Tune engine, replace air filter, check tire pressure for optimal efficiency",
                'decrease': "Investigate causes of poor fuel economy - engine, tires, driving habits"
            },
            
            # Air system
            'air_filter_condition': {
                'increase': "Replace air filter with new clean filter",
                'decrease': "Monitor air filter replacement schedule"
            },
            'air_pressure': {
                'increase': "Check air compressor and air lines for proper operation",
                'decrease': "Inspect air system for leaks and pressure regulation"
            },
            
            # Maintenance scoring
            'maintenance_score': {
                'increase': "Implement comprehensive preventive maintenance program",
                'decrease': "Review current maintenance practices for optimization"
            },
            'service_history_score': {
                'increase': "Establish regular service records and documentation",
                'decrease': "Maintain current service documentation standards"
            },
            
            # Usage patterns
            'daily_mileage': {
                'increase': "Plan for increased maintenance frequency due to higher usage",
                'decrease': "Optimize route planning to reduce unnecessary mileage"
            },
            'operating_hours': {
                'increase': "Schedule maintenance based on operating hours rather than calendar time",
                'decrease': "Monitor equipment usage efficiency"
            },
            'idle_time': {
                'increase': "Reduce unnecessary idling to prevent engine wear",
                'decrease': "Monitor idle time for optimal engine operation"
            },
            
            # Environmental factors
            'ambient_temperature': {
                'increase': "Ensure adequate cooling systems for high temperature operation",
                'decrease': "Monitor cold weather starting and warming procedures"
            },
            'humidity': {
                'increase': "Check for corrosion protection and moisture-related issues",
                'decrease': "Monitor for dry climate effects on seals and gaskets"
            },
            
            # Driver behavior
            'harsh_braking_events': {
                'increase': "Provide driver training on smooth braking techniques",
                'decrease': "Maintain current good driving practices"
            },
            'rapid_acceleration_events': {
                'increase': "Train drivers on fuel-efficient acceleration patterns",
                'decrease': "Continue current smooth acceleration practices"
            },
            'speeding_events': {
                'increase': "Implement speed monitoring and driver coaching programs",
                'decrease': "Maintain current safe driving practices"
            }
        }
        
        # Get the specific recommendation or provide a generic one
        feature_lower = feature.lower().replace('_', '').replace(' ', '')
        
        # Try exact match first
        if feature in maintenance_actions:
            return maintenance_actions[feature][change_direction]
        
        # Try partial matches
        for key in maintenance_actions:
            if key.replace('_', '') in feature_lower or feature_lower in key.replace('_', ''):
                return maintenance_actions[key][change_direction]
        
        # Generic recommendations based on direction
        if change_direction == 'increase':
            return f"Take corrective action to increase {feature} through appropriate maintenance"
        else:
            return f"Address factors contributing to high {feature} levels"
    
    def analyze_counterfactuals(self, instance, counterfactuals, config: CounterfactualConfig):
        """Analyze counterfactuals and generate recommendations with safe float handling."""
        if counterfactuals.empty:
            return {
                "error": "No counterfactuals were generated",
                "recommendations": []
            }
        
        vehicle_identifier = self.create_vehicle_identifier(instance)
        orig_component = instance['component_at_risk'].values[0]
        
        # Analyze changes for each counterfactual
        all_changes = []
        counterfactual_analyses = []
        
        for i, (_, cf) in enumerate(counterfactuals.iterrows()):
            changes = []
            
            # Compare features with safe conversion
            for feature in self.feature_names:
                if feature in instance.columns and feature in cf.index:
                    orig_val = instance[feature].values[0]
                    cf_val = cf[feature]
                    
                    if pd.api.types.is_numeric_dtype(type(orig_val)) and pd.api.types.is_numeric_dtype(type(cf_val)):
                        if abs(orig_val - cf_val) > 0.01:
                            change_magnitude = abs(cf_val - orig_val)
                            change_info = {
                                'feature': feature,
                                'original_value': safe_float_for_cf(orig_val),
                                'counterfactual_value': safe_float_for_cf(cf_val),
                                'change': safe_float_for_cf(cf_val - orig_val),
                                'change_magnitude': safe_float_for_cf(change_magnitude),
                                'change_direction': 'increase' if cf_val > orig_val else 'decrease'
                            }
                            changes.append(change_info)
                            all_changes.append(change_info)
            
            # Sort changes by magnitude
            changes.sort(key=lambda x: x['change_magnitude'], reverse=True)
            
            counterfactual_analyses.append({
                'counterfactual_id': i + 1,
                'target_component': config.target_component,
                'changes': changes
            })
        
        # Generate consolidated recommendations
        feature_recommendations = {}
        for change in all_changes:
            feature = change['feature']
            if feature not in feature_recommendations:
                feature_recommendations[feature] = {
                    'total_magnitude': 0,
                    'direction_votes': {'increase': 0, 'decrease': 0},
                    'avg_change': 0,
                    'count': 0
                }
            
            feature_recommendations[feature]['total_magnitude'] += change['change_magnitude']
            feature_recommendations[feature]['direction_votes'][change['change_direction']] += 1
            feature_recommendations[feature]['avg_change'] += change['change']
            feature_recommendations[feature]['count'] += 1
        
        # Finalize recommendations with safe conversions
        recommendations = []
        for feature, data in feature_recommendations.items():
            avg_magnitude = data['total_magnitude'] / data['count']
            avg_change = data['avg_change'] / data['count']
            dominant_direction = 'increase' if data['direction_votes']['increase'] > data['direction_votes']['decrease'] else 'decrease'
            
            # Generate actionable recommendation
            actionable_advice = self.generate_actionable_recommendation(feature, dominant_direction, avg_change)
            
            recommendations.append({
                'feature': feature,
                'recommendation': actionable_advice,
                'technical_change': f"{dominant_direction.title()} {feature}",
                'average_change': safe_float_for_cf(avg_change),
                'current_value': safe_float_for_cf(instance[feature].values[0]) if feature in instance.columns else None,
                'target_range': f"{safe_float_for_cf(instance[feature].values[0] + avg_change):.2f}" if feature in instance.columns else None,
                'importance_score': safe_float_for_cf(avg_magnitude),
                'confidence': safe_float_for_cf(max(data['direction_votes'].values()) / data['count']),
                'priority': 'High' if avg_magnitude > np.percentile([r['total_magnitude']/r['count'] for r in feature_recommendations.values()], 75) else 
                          'Medium' if avg_magnitude > np.percentile([r['total_magnitude']/r['count'] for r in feature_recommendations.values()], 50) else 'Low'
            })
        
        # Sort by importance
        recommendations.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Create vehicle details
        vehicle_details = {
            'vehicle_identifier': vehicle_identifier,
            'current_risk': orig_component,
            'target_risk': config.target_component
        }
        
        # Add available vehicle information
        for attr in ['vehicle_id', 'vehicle_type', 'make', 'model', 'mileage', 'vehicle_age', 'days_till_breakdown']:
            if attr in instance.columns and pd.notna(instance[attr].values[0]):
                value = instance[attr].values[0]
                if pd.api.types.is_integer_dtype(type(value)):
                    vehicle_details[attr] = int(value)
                elif pd.api.types.is_float_dtype(type(value)):
                    vehicle_details[attr] = safe_float_for_cf(value)
                else:
                    vehicle_details[attr] = str(value)
        
        return {
            "vehicle_details": vehicle_details,
            "counterfactual_analyses": counterfactual_analyses,
            "recommendations": recommendations,
            "summary": {
                "total_counterfactuals": int(len(counterfactuals)),
                "total_recommendations": int(len(recommendations)),
                "high_priority_actions": sum(1 for r in recommendations if r['priority'] == 'High'),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }