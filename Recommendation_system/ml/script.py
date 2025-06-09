
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from datetime import datetime
import joblib
import os
warnings.filterwarnings('ignore')

class DynamicClassification:
    def __init__(self, dataset_path=None, target_column=None):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.target_column = target_column  # Dynamically set target column
        self.features = None
        self.column_types = {}
        self.sample_values = {}
        self.dataset_path = dataset_path
        self.one_hot_columns = {}
        self.datetime_columns = []
        self.original_columns = None
        self.target_mapping = None
        self.model_dir = f'{target_column}_classification_model'

    def get_manual_input(self):
        """Get manual input from user for prediction."""
        print("\nEnter values for prediction (just as they appear in your CSV):")
        input_data = {}
        
        # Show original column names from CSV
        for column in self.original_columns:
            while True:
                try:
                    print(f"\n{column}:")
                    print(f"Example value from dataset: {self.sample_values.get(column, 'N/A')}")
                    
                    if column in self.datetime_columns:
                        print("Enter date in format YYYY-MM-DD HH:MM:SS")
                    elif column in self.one_hot_columns:
                        print(f"Possible values: {', '.join(map(str, self.one_hot_columns[column]))}")
                    
                    value = input("Enter value: ")
                    
                    # Store the raw input - preprocessing will happen later
                    if column in self.datetime_columns:
                        # Validate datetime format
                        pd.to_datetime(value)
                        input_data[column] = value
                    elif self.column_types.get(column) in ['int64', 'int32']:
                        input_data[column] = int(value)
                    elif self.column_types.get(column) in ['float64', 'float32']:
                        input_data[column] = float(value)
                    else:
                        input_data[column] = value
                    break
                except ValueError:
                    print(f"Invalid input. Please enter a valid value matching the example format.")
                    
        return input_data
        
    def identify_column_types(self, df):
        """Identify and store column types for preprocessing"""
        print("\nIdentifying column types...")
        for column in df.columns:
            if column != self.target_column:
                # Store sample values and data types
                self.sample_values[column] = str(df[column].iloc[0])
                self.column_types[column] = str(df[column].dtype)
                
                # Try to identify datetime columns
                if df[column].dtype == 'object':
                    try:
                        first_valid = df[column].dropna().iloc[0]
                        pd.to_datetime(first_valid)
                        self.datetime_columns.append(column)
                        print(f"Detected datetime column: {column}")
                        continue
                    except (ValueError, TypeError):
                        pass
                
                # Identify categorical columns
                if df[column].dtype == 'object' or (
                    df[column].dtype in ['int64', 'float64'] and 
                    df[column].nunique() < min(20, len(df) * 0.1)  # Adjusted threshold for more categories
                ):
                    self.one_hot_columns[column] = sorted(df[column].unique())
                    print(f"Detected categorical column: {column} with values: {self.one_hot_columns[column]}")
                else:
                    print(f"Detected numeric column: {column}")

    def preprocess_data(self, df, is_training=True):
        df_processed = df.copy()
        
        # Handle missing values
        for column in df_processed.columns:
            if df_processed[column].isnull().sum() > 0:
                if df_processed[column].dtype in ['int64', 'float64']:
                    df_processed[column].fillna(df_processed[column].mean(), inplace=True)
                else:
                    df_processed[column].fillna(df_processed[column].mode()[0], inplace=True)
        
        # Handle datetime columns
        for column in self.datetime_columns:
            try:
                df_processed[f'{column}_year'] = pd.to_datetime(df_processed[column]).dt.year
                df_processed[f'{column}_month'] = pd.to_datetime(df_processed[column]).dt.month
                df_processed[f'{column}_day'] = pd.to_datetime(df_processed[column]).dt.day
                df_processed[f'{column}_hour'] = pd.to_datetime(df_processed[column]).dt.hour
                df_processed = df_processed.drop(columns=[column])
            except Exception as e:
                print(f"Error processing datetime column {column}: {str(e)}")
        
        # Handle categorical columns with one-hot encoding
        for column, unique_values in self.one_hot_columns.items():
            if column in df_processed.columns:
                for value in unique_values:
                    col_name = f"{column}_{value}"
                    df_processed[col_name] = (df_processed[column] == value).astype(int)
                df_processed = df_processed.drop(columns=[column])
        
        return df_processed
        
    def validate_dataset(self):
        try:
            df = pd.read_csv(self.dataset_path)
            if self.target_column not in df.columns:
                raise ValueError(f"Error: Target column '{self.target_column}' not found in dataset")
            
            # Store original columns except target column
            self.original_columns = [col for col in df.columns if col != self.target_column]
            
            # Handle multiclass target variable
            label_encoder = LabelEncoder()
            df[self.target_column] = label_encoder.fit_transform(df[self.target_column])
            
            # Store mapping for later use
            self.target_mapping = dict(zip(range(len(label_encoder.classes_)), 
                                         label_encoder.classes_))
            
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
    
    def save_model(self):
        """Save the model and all necessary preprocessing information"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'column_types': self.column_types,
            'sample_values': self.sample_values,
            'one_hot_columns': self.one_hot_columns,
            'datetime_columns': self.datetime_columns,
            'original_columns': self.original_columns,
            'target_mapping': self.target_mapping
        }
        
        joblib.dump(model_data, os.path.join(self.model_dir, 'model.pkl'))
        print(f"\nModel and preprocessing information saved to {self.model_dir}")
    
    def load_model(self):
        """Load the model and all necessary preprocessing information"""
        try:
            model_path = os.path.join(self.model_dir, 'model.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError("No saved model found. Please train the model first.")
            
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            self.column_types = model_data['column_types']
            self.sample_values = model_data['sample_values']
            self.one_hot_columns = model_data['one_hot_columns']
            self.datetime_columns = model_data['datetime_columns']
            self.original_columns = model_data['original_columns']
            self.target_mapping = model_data['target_mapping']
            
            print("\nModel loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    # def train_model(self):
    #     print("\nLoading and validating dataset...")
    #     df = self.validate_dataset()
        
    #     print("Identifying column types and preprocessing data...")
    #     self.identify_column_types(df)
        
    #     # Preprocess data
    #     df_processed = self.preprocess_data(df)
        
    #     # Separate features and target
    #     self.features = [col for col in df_processed.columns if col != self.target_column]
    #     X = df_processed[self.features]
    #     y = df_processed[self.target_column]
        
    #     # Split dataset
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    #     # Scale features
    #     X_train_scaled = self.scaler.fit_transform(X_train)
    #     X_test_scaled = self.scaler.transform(X_test)
        
    #     print("\nPerforming GridSearchCV for optimal parameters...")
    #     param_grid = {
    #         'n_estimators': [100, 200],
    #         'max_depth': [None, 10, 20],
    #         'min_samples_split': [2, 5],
    #         'class_weight': ['balanced', 'balanced_subsample']
    #     }
        
    #     self.model = GridSearchCV(
    #         RandomForestClassifier(random_state=42),
    #         param_grid,
    #         cv=3,
    #         scoring='accuracy',
    #         n_jobs=-1
    #     )
        
    #     print("\nTraining model...")
    #     self.model.fit(X_train_scaled, y_train)
        
    #     print(f"\nBest parameters found: {self.model.best_params_}")
        
    #     # Make predictions
    #     y_pred = self.model.predict(X_test_scaled)
        
    #     # Calculate metrics
    #     accuracy = accuracy_score(y_test, y_pred)
    #     conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
    #     # Print metrics
    #     print("\nModel Performance Metrics:")
    #     print(f"Accuracy: {accuracy:.4f}")
    #     print("\nClassification Report:")
    #     print(classification_report(y_test, y_pred, 
    #                              target_names=[self.target_mapping[i] for i in sorted(self.target_mapping.keys())]))
    #     print("\nConfusion Matrix:")
    #     print(conf_matrix)
        
    #     # Save the trained model
    #     self.save_model()
        
    #     return {
    #         'accuracy': float(accuracy),
    #         'confusion_matrix': conf_matrix,
    #         'best_parameters': self.model.best_params_
    #     }

    def train_model(self):
        print("\nLoading and validating dataset...")
        df = self.validate_dataset()
        
        print("Identifying column types and preprocessing data...")
        self.identify_column_types(df)
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Separate features and target
        self.features = [col for col in df_processed.columns if col != self.target_column]
        X = df_processed[self.features]
        y = df_processed[self.target_column]
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\nPerforming GridSearchCV for optimal parameters...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        self.model = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        print("\nTraining model...")
        self.model.fit(X_train_scaled, y_train)
        
        print(f"\nBest parameters found: {self.model.best_params_}")
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        # Get unique classes that actually appear in the test set and predictions
        unique_classes = sorted(list(set(np.union1d(y_test, y_pred))))
        target_names = [self.target_mapping[i] for i in unique_classes]
        
        # Print metrics
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Save the trained model
        self.save_model()
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix,
            'best_parameters': self.model.best_params_
        }

    def predict_single(self, input_data):
        try:
            # Verify the model is loaded
            if not self.model:
                raise ValueError("No model loaded. Please train or load a model first.")
            
            # Create DataFrame with original column names
            df = pd.DataFrame([input_data])
            
            # Apply the same preprocessing as training data
            df_processed = self.preprocess_data(df, is_training=False)
            
            # Ensure all features are present
            for feature in self.features:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0
            
            X = df_processed[self.features]
            X_scaled = self.scaler.transform(X)
            
            # Get raw prediction and map to original target category
            raw_prediction = self.model.predict(X_scaled)[0]
            predicted_target = self.target_mapping[raw_prediction]
            
            # Print the input data and prediction
            print("\nInput Values:")
            for col, val in input_data.items():
                print(f"{col}: {val}")
            
            print(f"\nPredicted {self.target_column}: {predicted_target}")
            
            return predicted_target
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

    def predict_bulk(self, csv_path):
        try:
            # Verify the model is loaded
            if not self.model:
                raise ValueError("No model loaded. Please train or load a model first.")
            
            df = pd.read_csv(csv_path)
            
            # Check if target column already exists
            if self.target_column in df.columns:
                print(f"\nWarning: Dataset already contains a '{self.target_column}' column.")
                print("Skipping prediction to avoid overwriting existing values.")
                return
            
            # Validate columns match original dataset
            missing_cols = set(self.original_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in input CSV: {missing_cols}")
            
            df_processed = self.preprocess_data(df, is_training=False)
            
            X = df_processed[self.features]
            X_scaled = self.scaler.transform(X)
            
            # Get predictions
            predictions = self.model.predict(X_scaled)
            
            # Map predictions to original target categories
            predicted_targets = [self.target_mapping[pred] for pred in predictions]
            
            # Add only predictions to original dataframe
            df[self.target_column] = predicted_targets
            
            output_path = csv_path.replace('.csv', f'_{self.target_column}_predictions.csv')
            df.to_csv(output_path, index=False)
            print(f"\nPredictions saved to: {output_path}")
            
        except Exception as e:
            print(f"Error during bulk prediction: {str(e)}")
