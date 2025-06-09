import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.cluster import KMeans
import ollama  # Replace AzureOpenAI with Ollama
import requests
import json  # For custom HTTP requests to Ollama

class DataPreprocessor:
    def __init__(self, required_columns=['Churn', 'Product', 'Recency', 'Frequency', 'Monetary', 'CLV', 'Default']):
        self.required_columns = required_columns
        self.le_dict = {}
        self.scaler = StandardScaler()
        self.product_mapping = {}


    def convert_to_binary(self, series):
        """Convert any series to binary values"""
        if series.dtype in ['object', 'string']:
            try:
                series = series.str.lower()
                series = series.map({'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0})
            except:
                le = LabelEncoder()
                series = le.fit_transform(series.astype(str))
        elif series.dtype in ['int64', 'float64']:
            median = series.median()
            series = (series > median).astype(int)

        unique_vals = series.unique()
        if len(unique_vals) > 2:
            median = series.median()
            series = (series > median).astype(int)
        elif len(unique_vals) == 2 and not all(val in [0, 1] for val in unique_vals):
            min_val = series.min()
            series = (series != min_val).astype(int)

        return series

    def preprocess_targets(self, df):
        """Preprocess all target variables"""
        if 'Churn' in df.columns:
            df['Churn'] = self.convert_to_binary(df['Churn'])
        if 'Default' in df.columns:
            df['Default'] = self.convert_to_binary(df['Default'])
        if 'Product' in df.columns and df['Product'].dtype in ['object', 'string']:
            self.le_dict['Product'] = LabelEncoder()
            df['Product'] = self.le_dict['Product'].fit_transform(df['Product'].astype(str))
            # Store the mapping from indices to product names
            self.product_mapping = {i: name for i, name in enumerate(self.le_dict['Product'].classes_)}
            print(f"Product mapping: {self.product_mapping}")  # Debug output
        return df


    def preprocess(self, df):
        df = df.copy()

        numeric_imputer = SimpleImputer(strategy='mean')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(numeric_cols) > 0:
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        if len(categorical_cols) > 0:
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

        # Preprocess targets first
        df = self.preprocess_targets(df)

        # Exclude targets from further transformations
        target_cols = ['Churn', 'Default', 'Product']
        feature_cols = [col for col in df.columns if col not in target_cols]

        for col in feature_cols:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df = df.drop(columns=[col])
                except:
                    self.le_dict[col] = LabelEncoder()
                    df[col] = self.le_dict[col].fit_transform(df[col].astype(str))


        # Scale only numeric feature columns
        numeric_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        if numeric_cols:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        return df

class ModelPipeline:
    def __init__(self, ollama_url, ollama_model):
        self.ollama_url = ollama_url
        self.model_name = ollama_model

    # def get_ai_insights(self, data_description):
    #     """Fetch AI insights from the Ollama server using HTTP requests"""
    #     try:
    #         payload = {
    #             "model": self.model_name,
    #             "messages": [
    #                 {"role": "system", "content": "You are a data science expert providing insights on model results."},
    #                 {"role": "user", "content": f"Provide insights on the following results: {data_description}"}
    #             ],
    #             "options": {
    #                 "temperature": 0.7,
    #                 "max_tokens": 500
    #             }
    #         }
    #         response = requests.post(f"{self.ollama_url}/api/chat", json=payload)
    #         response.raise_for_status()
    #         result = response.json()
    #         return result['message']['content']
    #     except requests.exceptions.RequestException as e:
    #         print(f"Error connecting to Ollama server: {str(e)}")
    #         return f"Failed to generate insights due to server error: {str(e)}"
    #     except KeyError as e:
    #         print(f"Error parsing Ollama response: {str(e)}")
    #         return f"Failed to parse insights from server response: {str(e)}"

    def get_ai_insights(self, data_description):
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a data science expert providing insights on model results."},
                    {"role": "user", "content": f"Provide insights on the following results: {data_description}"}
                ],
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            }

            response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=10, stream=True)
            response.raise_for_status()

            full_content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'message' in chunk and 'content' in chunk['message']:
                            full_content += chunk['message']['content']
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError as e:
                        print(f"Skipping malformed chunk: {line.decode('utf-8')} - Error: {e}")

            if not full_content:
                return f"No valid content received from Ollama: {response.text}"
            return full_content

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama server: {str(e)}")
            return f"Failed to generate insights due to server error: {str(e)}"
        except Exception as e:
            print(f"Unexpected error in get_ai_insights: {str(e)}")
            return f"Failed to generate insights: {str(e)}"

    def binary_classification(self, X, y, feature_name):
        """Perform binary classification with enhanced validation"""
        try:
            y = y.fillna(0)
            y = pd.to_numeric(y, errors='coerce').fillna(0)

            # Ensure binary target
            unique_vals = y.unique()
            if len(unique_vals) > 2 or y.dtype in ['float64', 'float32']:
                print(f"Warning: {feature_name} has continuous or multi-class values. Converting to binary.")
                y = (y > y.median()).astype(int)
            elif len(unique_vals) == 1:
                print(f"Warning: {feature_name} has only one class ({y.unique()[0]})")
                synthetic_size = int(len(y) * 0.1)
                indices = np.random.choice(len(y), synthetic_size, replace=False)
                y.iloc[indices] = 1 - y.iloc[0]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'class_distribution': dict(zip(['class_0', 'class_1'], np.bincount(y) / len(y))),
                'average_probability': np.mean(y_pred_proba)
            }

            insights = self.get_ai_insights(
                f"Binary classification results for {feature_name}:\n" +
                f"Accuracy: {results['accuracy']:.2f}\n" +
                f"F1 Score: {results['f1']:.2f}\n" +
                f"Class Distribution: {results['class_distribution']}"
            )
            return results, insights
        except Exception as e:
            print(f"Error in binary classification for {feature_name}: {str(e)}")
            return {'error': str(e)}, f"Analysis failed: {str(e)}"

    def multiclass_classification(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        class_probabilities = model.predict_proba(X_test)

        unique_products = pd.Series(y).unique()
        product_percentages = pd.Series(y).value_counts(normalize=True)
        top_3_products = {
            'products': unique_products[:3].tolist(),
            'percentages': product_percentages[:3].tolist()
        }

        insights = self.get_ai_insights(f"Multiclass classification results for products: {top_3_products}")
        return top_3_products, insights

    def regression(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        results = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mse': mean_squared_error(y_test, y_pred),
            'smape': mean_absolute_percentage_error(y_test, y_pred) * 100,
            'average_prediction': y_pred.mean()
        }

        insights = self.get_ai_insights(f"Regression results for profitability: {results}")
        return results, insights

    def clustering(self, X):
        numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
        X_cluster = X[numerical_columns].copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        X_scaled_df = pd.DataFrame(X_scaled, columns=X_cluster.columns)
        X_scaled_df['CLV'] = X['CLV']

        cluster_means = pd.DataFrame({'cluster': clusters, 'CLV': X_scaled_df['CLV']}).groupby('cluster')['CLV'].mean()

        cluster_mapping = {
            cluster_means.idxmax(): 'Platinum',
            cluster_means.idxmin(): 'Silver',
            list(set(range(3)) - set([cluster_means.idxmax(), cluster_means.idxmin()]))[0]: 'Gold'
        }

        segments = pd.Series(clusters).map(cluster_mapping)
        segment_distribution = segments.value_counts(normalize=True)

        cluster_profiles = {}
        for cluster in range(3):
            cluster_data = X_cluster[clusters == cluster]
            cluster_profiles[cluster_mapping[cluster]] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(X_cluster) * 100,
                'mean_values': cluster_data.mean().to_dict()
            }

        results = {
            'segment_distribution': segment_distribution.to_dict(),
            'cluster_profiles': cluster_profiles
        }

        insights = self.get_ai_insights(f"Clustering results: {results}")
        return results, insights

def main(csv_path):
    print("Loading data...")
    df = pd.read_csv(csv_path)

    print("Initializing pipeline...")
    preprocessor = DataPreprocessor()
    pipeline = ModelPipeline(
        ollama_url="http://168.138.199.17:31953",
        ollama_model="llama3.2:latest"
    )

    print("Preprocessing data...")
    processed_df = preprocessor.preprocess(df)

    # Debug target columns
    print("Preprocessed Data Types:")
    print(processed_df[['Churn', 'Default', 'Product']].dtypes)
    print("Unique values in Churn:", processed_df['Churn'].unique())
    print("Unique values in Default:", processed_df['Default'].unique())
    print("Unique values in Product:", processed_df['Product'].unique())

    print("Preparing features...")
    target_columns = ['Churn', 'Default', 'Product']
    feature_columns = [col for col in processed_df.columns if col not in target_columns]
    X_binary = processed_df[feature_columns]

    print("Running models...")
    if 'Churn' in processed_df.columns:
        print("Processing Churn prediction...")
        churn_results, churn_insights = pipeline.binary_classification(X_binary, processed_df['Churn'], 'Churn')
        if churn_results:
            print("\nChurn Analysis Results:", churn_results)
            print("Churn AI Insights:", churn_insights)

    if 'Default' in processed_df.columns:
        print("\nProcessing Default prediction...")
        default_results, default_insights = pipeline.binary_classification(X_binary, processed_df['Default'], 'Default')
        if default_results:
            print("\nDefault Analysis Results:", default_results)
            print("Default AI Insights:", default_insights)

    if 'Product' in processed_df.columns:
        product_results, product_insights = pipeline.multiclass_classification(X_binary, processed_df['Product'])
        print("\nProduct Analysis Results:", product_results)
        print("Product AI Insights:", product_insights)

    if all(col in processed_df.columns for col in ['Recency', 'Frequency', 'Monetary', 'CLV']):
        profitability_features = processed_df[['Recency', 'Frequency', 'Monetary', 'CLV']]
        profitability_target = processed_df['Monetary'] * processed_df['Frequency']
        profitability_results, profitability_insights = pipeline.regression(profitability_features, profitability_target)
        print("\nProfitability Analysis Results:", profitability_results)
        print("Profitability AI Insights:", profitability_insights)

    clustering_results, clustering_insights = pipeline.clustering(processed_df)
    print("\nClustering Results:", clustering_results)
    print("Clustering AI Insights:", clustering_insights)

if __name__ == "__main__":
    main("mix_data.csv")