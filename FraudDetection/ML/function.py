

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
# import warnings
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
# from datetime import datetime
# import joblib
# import os
# warnings.filterwarnings('ignore')

# class AutoFraudDetection:
#     def __init__(self, dataset_path=None, target_column=None):
#         self.label_encoders = {}
#         self.scaler = StandardScaler()
#         self.model = None
#         self.target_column = target_column  # Now dynamic
#         self.features = None
#         self.column_types = {}
#         self.sample_values = {}
#         self.dataset_path = dataset_path
#         self.one_hot_columns = {}
#         self.datetime_columns = []
#         self.original_columns = None
#         self.target_mapping = None  # Store mapping of target values
#         self.model_dir = 'fraud_detection_model'

#     def validate_dataset(self):
#         try:
#             # Read the dataset
#             df = pd.read_csv(self.dataset_path)

#             # Validate target column exists
#             if self.target_column not in df.columns:
#                 raise ValueError(f"Error: Target column '{self.target_column}' not found in dataset")

#             # Store original columns except target column
#             self.original_columns = [col for col in df.columns if col != self.target_column]

#             # Handle different formats of target variable
#             unique_values = df[self.target_column].unique()

#             # Convert target values to standard format (0 and 1)
#             if set(unique_values) <= {'0', '1', 0, 1}:
#                 df[self.target_column] = df[self.target_column].astype(int)
#                 self.target_mapping = {0: 0, 1: 1}
#             else:
#                 # Convert various formats to binary
#                 positive_values = {'1', 'y', 'yes', 'true', 't', 'Y', 'YES', 'TRUE', 'True', True}
#                 df[self.target_column] = df[self.target_column].apply(
#                     lambda x: 1 if str(x).lower() in map(str.lower, positive_values) else 0
#                 )
#                 # Store original values for mapping back
#                 self.target_mapping = {
#                     0: sorted([str(v) for v in unique_values if str(v).lower() not in map(str.lower, positive_values)])[0],
#                     1: sorted([str(v) for v in unique_values if str(v).lower() in map(str.lower, positive_values)])[0]
#                 }

#             return df
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

#     def identify_column_types(self, df):
#         """Identify and store column types for preprocessing"""
#         print("\nIdentifying column types...")
#         for column in df.columns:
#             if column != self.target_column:
#                 # Store sample values and data types
#                 self.sample_values[column] = str(df[column].iloc[0])
#                 self.column_types[column] = str(df[column].dtype)

#                 # Try to identify datetime columns
#                 if df[column].dtype == 'object':
#                     try:
#                         # Check if first non-null value is datetime
#                         first_valid = df[column].dropna().iloc[0]
#                         pd.to_datetime(first_valid)
#                         self.datetime_columns.append(column)
#                         print(f"Detected datetime column: {column}")
#                         continue
#                     except (ValueError, TypeError):
#                         pass

#                 # Identify categorical columns
#                 if df[column].dtype == 'object' or (
#                     df[column].dtype in ['int64', 'float64'] and
#                     df[column].nunique() < min(10, len(df) * 0.05)  # Less than 10 unique values or 5% of data
#                 ):
#                     self.one_hot_columns[column] = sorted(df[column].unique())
#                     print(f"Detected categorical column: {column} with values: {self.one_hot_columns[column]}")
#                 else:
#                     print(f"Detected numeric column: {column}")

#     def preprocess_data(self, df, is_training=True):
#         df_processed = df.copy()

#         # Handle missing values
#         for column in df_processed.columns:
#             if df_processed[column].isnull().sum() > 0:
#                 if df_processed[column].dtype in ['int64', 'float64']:
#                     df_processed[column].fillna(df_processed[column].mean(), inplace=True)
#                 else:
#                     df_processed[column].fillna(df_processed[column].mode()[0], inplace=True)

#         # Handle datetime columns
#         for column in self.datetime_columns:
#             try:
#                 df_processed[f'{column}_year'] = pd.to_datetime(df_processed[column]).dt.year
#                 df_processed[f'{column}_month'] = pd.to_datetime(df_processed[column]).dt.month
#                 df_processed[f'{column}_day'] = pd.to_datetime(df_processed[column]).dt.day
#                 df_processed[f'{column}_hour'] = pd.to_datetime(df_processed[column]).dt.hour
#                 df_processed = df_processed.drop(columns=[column])
#             except Exception as e:
#                 print(f"Error processing datetime column {column}: {str(e)}")

#         # Handle categorical columns with one-hot encoding
#         for column, unique_values in self.one_hot_columns.items():
#             if column in df_processed.columns:
#                 for value in unique_values:
#                     col_name = f"{column}_{value}"
#                     df_processed[col_name] = (df_processed[column] == value).astype(int)
#                 df_processed = df_processed.drop(columns=[column])

#         return df_processed

#     def save_model(self):
#         """Save the model and all necessary preprocessing information"""
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)

#         model_data = {
#             'model': self.model,
#             'scaler': self.scaler,
#             'features': self.features,
#             'column_types': self.column_types,
#             'sample_values': self.sample_values,
#             'one_hot_columns': self.one_hot_columns,
#             'datetime_columns': self.datetime_columns,
#             'original_columns': self.original_columns,
#             'target_mapping': self.target_mapping,
#             'target_column': self.target_column  # Save the target column name
#         }

#         joblib.dump(model_data, os.path.join(self.model_dir, 'model.pkl'))
#         print(f"\nModel and preprocessing information saved to {self.model_dir}")

#     def load_model(self):
#         """Load the model and all necessary preprocessing information"""
#         try:
#             model_path = os.path.join(self.model_dir, 'model.pkl')
#             if not os.path.exists(model_path):
#                 raise FileNotFoundError("No saved model found. Please train the model first.")

#             model_data = joblib.load(model_path)

#             self.model = model_data['model']
#             self.scaler = model_data['scaler']
#             self.features = model_data['features']
#             self.column_types = model_data['column_types']
#             self.sample_values = model_data['sample_values']
#             self.one_hot_columns = model_data['one_hot_columns']
#             self.datetime_columns = model_data['datetime_columns']
#             self.original_columns = model_data['original_columns']
#             self.target_mapping = model_data['target_mapping']
#             self.target_column = model_data['target_column']  # Load the target column name

#             print("\nModel loaded successfully!")
#             return True
#         except Exception as e:
#             print(f"Error loading model: {str(e)}")
#             return False

#     def train_model(self):
#         print("\nLoading and validating dataset...")
#         df = self.validate_dataset()

#         print("Identifying column types and preprocessing data...")
#         self.identify_column_types(df)

#         # Preprocess data
#         df_processed = self.preprocess_data(df)

#         # Separate features and target
#         self.features = [col for col in df_processed.columns if col != self.target_column]
#         X = df_processed[self.features]
#         y = df_processed[self.target_column]

#         # Split dataset
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Scale features
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)

#         print("\nTraining model...")
#         self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
#         self.model.fit(X_train_scaled, y_train)

#         # Make predictions
#         y_pred = self.model.predict(X_test_scaled)
#         y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

#         # Calculate metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         conf_matrix = confusion_matrix(y_test, y_pred).tolist()
#         roc_auc = roc_auc_score(y_test, y_pred_proba)

#         # Print metrics
#         print("\nModel Performance Metrics:")
#         print(f"Accuracy: {accuracy:.4f}")
#         print(f"F1 Score: {f1:.4f}")
#         print(f"ROC-AUC Score: {roc_auc:.4f}")
#         print("\nClassification Report:")
#         print(classification_report(y_test, y_pred))
#         print("\nConfusion Matrix:")
#         print(conf_matrix)

#         # Save the trained model
#         self.save_model()

#         # Return metrics
#         return {
#             'accuracy': float(accuracy),
#             'f1_score': float(f1),
#             'confusion_matrix': conf_matrix,
#         }

#     def get_columns(self, df):
#         """Display available columns for selection"""
#         print("\nAvailable columns:")
#         for i, col in enumerate(df.columns, 1):
#             print(f"{i}. {col}")

#     def get_manual_input(self):
#         print("\nEnter values for prediction (just as they appear in your CSV):")
#         input_data = {}

#         # Show original column names from CSV
#         for column in self.original_columns:
#             while True:
#                 try:
#                     print(f"\n{column}:")
#                     print(f"Example value from dataset: {self.sample_values.get(column, 'N/A')}")

#                     if column in self.datetime_columns:
#                         print("Enter date in format YYYY-MM-DD HH:MM:SS")
#                     elif column in self.one_hot_columns:
#                         print(f"Possible values: {', '.join(map(str, self.one_hot_columns[column]))}")

#                     value = input("Enter value: ")

#                     # Store the raw input - preprocessing will happen later
#                     if column in self.datetime_columns:
#                         # Validate datetime format
#                         pd.to_datetime(value)
#                         input_data[column] = value
#                     elif self.column_types.get(column) in ['int64', 'int32']:
#                         input_data[column] = int(value)
#                     elif self.column_types.get(column) in ['float64', 'float32']:
#                         input_data[column] = float(value)
#                     else:
#                         input_data[column] = value
#                     break
#                 except ValueError:
#                     print(f"Invalid input. Please enter a valid value matching the example format.")

#         return input_data

#     def predict_bulk(self, csv_path):
#         try:
#             df = pd.read_csv(csv_path)

#             # If target column already exists in the CSV, remove it from validation
#             original_columns_check = [col for col in self.original_columns if col != self.target_column]

#             # Validate columns match original dataset
#             missing_cols = set(original_columns_check) - set(df.columns)
#             if missing_cols:
#                 raise ValueError(f"Missing columns in input CSV: {missing_cols}")

#             df_processed = self.preprocess_data(df, is_training=False)

#             X = df_processed[self.features]
#             X_scaled = self.scaler.transform(X)

#             # Get raw predictions (0s and 1s)
#             raw_predictions = self.model.predict(X_scaled)

#             # Convert predictions to integers (0 or 1)
#             predictions = [int(pred) for pred in raw_predictions]

#             # Add predictions to original dataframe
#             # Only add if target column doesn't already exist
#             if self.target_column not in df.columns:
#                 df[self.target_column] = predictions

#             output_path = csv_path.replace('.csv', '_predictions.csv')
#             df.to_csv(output_path, index=False)
#             print(f"\nPredictions saved to: {output_path}")

#         except Exception as e:
#             print(f"Error during bulk prediction: {str(e)}")

#     def predict_single(self, input_data):
#         try:
#             # Create DataFrame with original column names
#             df = pd.DataFrame([input_data])

#             # Apply the same preprocessing as training data
#             df_processed = self.preprocess_data(df, is_training=False)

#             # Ensure all features are present
#             for feature in self.features:
#                 if feature not in df_processed.columns:
#                     df_processed[feature] = 0

#             X = df_processed[self.features]
#             X_scaled = self.scaler.transform(X)

#             # Get raw prediction (0 or 1)
#             raw_prediction = self.model.predict(X_scaled)[0]

#             # Convert to integer (0 or 1)
#             prediction = int(raw_prediction)

#             # Create output dictionary with same format as input
#             result = input_data.copy()

#             # Only add prediction if target column doesn't exist
#             if self.target_column not in result:
#                 result[self.target_column] = prediction

#             # Print the input data and prediction in a clear format
#             print("\nInput Values:")
#             for col, val in input_data.items():
#                 print(f"{col}: {val}")

#             print(f"\nPrediction:")
#             print(f"{self.target_column}: {prediction}")

#             return result

#         except Exception as e:
#             print(f"Error during prediction: {str(e)}")
#             return None

# def main():
#     try:
#         print("\nAdvanced Machine Learning Prediction System")
#         print("1. Train new model")
#         print("2. Load existing model")
#         choice = input("Enter your choice (1-2): ")

#         model = AutoFraudDetection()

#         if choice == '1':
#             # Prompt for dataset path
#             dataset_path = input("\nEnter the path to your training dataset (.csv): ")

#             # Read the dataset to help user select target column
#             df = pd.read_csv(dataset_path)

#             # Display available columns
#             model.get_columns(df)

#             # Prompt for target column selection
#             while True:
#                 try:
#                     target_index = int(input("\nEnter the number of the target column: ")) - 1
#                     target_column = df.columns[target_index]

#                     # Validate target column selection
#                     if target_column is None:
#                         print("Invalid column selection. Please try again.")
#                         continue

#                     # Set dataset path and target column
#                     model.dataset_path = dataset_path
#                     model.target_column = target_column

#                     # Train the model
#                     model.train_model()
#                     break
#                 except (ValueError, IndexError):
#                     print("Invalid input. Please enter a valid column number.")

#         elif choice == '2':
#             if not model.load_model():
#                 print("Could not load model. Please train a new model.")
#                 return
#         else:
#             print("Invalid choice")
#             return

#         while True:
#             print("\nChoose prediction mode:")
#             print("1. Manual input")
#             print("2. Bulk prediction from CSV")
#             print("3. Exit")

#             choice = input("Enter your choice (1-3): ")

#             if choice == '1':
#                 input_data = model.get_manual_input()
#                 model.predict_single(input_data)
#             elif choice == '2':
#                 csv_path = input("Enter the path to your prediction CSV file: ")
#                 model.predict_bulk(csv_path)
#             elif choice == '3':
#                 break
#             else:
#                 print("Invalid choice. Please try again.")

#             if choice in ['1', '2'] and input("\nMake another prediction? (y/n): ").lower() != 'y':
#                 break

#     except Exception as e:
#         print(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from datetime import datetime
import joblib
import os
warnings.filterwarnings('ignore')

class AutoFraudDetection:
    def __init__(self, dataset_path=None, target_column=None):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.target_column = target_column  # Now dynamic
        self.features = None
        self.column_types = {}
        self.sample_values = {}
        self.dataset_path = dataset_path
        self.one_hot_columns = {}
        self.datetime_columns = []
        self.original_columns = None
        self.target_mapping = None  # Store mapping of target values
        self.model_dir = 'fraud_detection_model'
    
    def validate_dataset(self):
        try:
            df = pd.read_csv(self.dataset_path)
            
            if self.target_column not in df.columns:
                raise ValueError(f"Error: Target column '{self.target_column}' not found in dataset")
            
            self.original_columns = [col for col in df.columns if col != self.target_column]
            
            unique_values = df[self.target_column].unique()
            
            # First check if values are already 0/1 or their string representations
            if set(map(str, unique_values)) <= {'0', '1'}:
                df[self.target_column] = df[self.target_column].astype(int)
                self.target_mapping = {0: 0, 1: 1}
            else:
                # Convert various formats to binary
                positive_values = {'1', 'y', 'yes', 'true', 't', 'Y', 'YES', 'TRUE', 'True'}
                
                # Convert to string first, then check
                df[self.target_column] = df[self.target_column].apply(
                    lambda x: 1 if str(x).lower() in [v.lower() for v in positive_values] else 0
                )
                
                # Store original values for mapping back
                self.target_mapping = {
                    0: sorted([str(v) for v in unique_values if str(v).lower() not in [v.lower() for v in positive_values]])[0],
                    1: sorted([str(v) for v in unique_values if str(v).lower() in [v.lower() for v in positive_values]])[0]
                }
            
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
    
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
                        # Check if first non-null value is datetime
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
                    df[column].nunique() < min(10, len(df) * 0.05)  # Less than 10 unique values or 5% of data
                ):
                    unique_values = sorted(df[column].unique(), key=str)
                    self.one_hot_columns[column] = unique_values
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
            'target_mapping': self.target_mapping,
            'target_column': self.target_column  # Save the target column name
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
            self.target_column = model_data['target_column']  # Load the target column name
            
            print("\nModel loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

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
        
        print("\nTraining model...")
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print metrics
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Save the trained model
        self.save_model()
        
        # Return metrics
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix,
        }

    def get_columns(self, df):
        """Display available columns for selection"""
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")

    def get_manual_input(self):
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

    def predict_bulk(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            # If target column already exists in the CSV, remove it from validation
            original_columns_check = [col for col in self.original_columns if col != self.target_column]
            
            # Validate columns match original dataset
            missing_cols = set(original_columns_check) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in input CSV: {missing_cols}")
            
            df_processed = self.preprocess_data(df, is_training=False)
            
            X = df_processed[self.features]
            X_scaled = self.scaler.transform(X)
            
            # Get raw predictions (0s and 1s)
            raw_predictions = self.model.predict(X_scaled)
            
            # Convert predictions to integers (0 or 1)
            predictions = [int(pred) for pred in raw_predictions]
            
            # Add predictions to original dataframe 
            # Only add if target column doesn't already exist
            if self.target_column not in df.columns:
                df[self.target_column] = predictions
            
            output_path = csv_path.replace('.csv', '_predictions.csv')
            df.to_csv(output_path, index=False)
            print(f"\nPredictions saved to: {output_path}")
            
        except Exception as e:
            print(f"Error during bulk prediction: {str(e)}")

    def predict_single(self, input_data):
        try:
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
            
            # Get raw prediction (0 or 1)
            raw_prediction = self.model.predict(X_scaled)[0]
            
            # Convert to integer (0 or 1)
            prediction = int(raw_prediction)
            
            # Create output dictionary with same format as input
            result = input_data.copy()
            
            # Only add prediction if target column doesn't exist
            if self.target_column not in result:
                result[self.target_column] = prediction
            
            # Print the input data and prediction in a clear format
            print("\nInput Values:")
            for col, val in input_data.items():
                print(f"{col}: {val}")
            
            print(f"\nPrediction:")
            print(f"{self.target_column}: {prediction}")
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

def main():
    try:
        print("\nAdvanced Machine Learning Prediction System")
        print("1. Train new model")
        print("2. Load existing model")
        choice = input("Enter your choice (1-2): ")
        
        model = AutoFraudDetection()
        
        if choice == '1':
            # Prompt for dataset path
            dataset_path = input("\nEnter the path to your training dataset (.csv): ")
            
            # Read the dataset to help user select target column
            df = pd.read_csv(dataset_path)
            
            # Display available columns
            model.get_columns(df)
            
            # Prompt for target column selection
            while True:
                try:
                    target_index = int(input("\nEnter the number of the target column: ")) - 1
                    target_column = df.columns[target_index]
                    
                    # Validate target column selection
                    if target_column is None:
                        print("Invalid column selection. Please try again.")
                        continue
                    
                    # Set dataset path and target column
                    model.dataset_path = dataset_path
                    model.target_column = target_column
                    
                    # Train the model
                    model.train_model()
                    break
                except (ValueError, IndexError):
                    print("Invalid input. Please enter a valid column number.")
        
        elif choice == '2':
            if not model.load_model():
                print("Could not load model. Please train a new model.")
                return
        else:
            print("Invalid choice")
            return
        
        while True:
            print("\nChoose prediction mode:")
            print("1. Manual input")
            print("2. Bulk prediction from CSV")
            print("3. Exit")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                input_data = model.get_manual_input()
                model.predict_single(input_data)
            elif choice == '2':
                csv_path = input("Enter the path to your prediction CSV file: ")
                model.predict_bulk(csv_path)
            elif choice == '3':
                break
            else:
                print("Invalid choice. Please try again.")
            
            if choice in ['1', '2'] and input("\nMake another prediction? (y/n): ").lower() != 'y':
                break
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
