import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import base64
import logging

# Initialize the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_data(file_path):
    try:
        # Load the dataset using the file path
        logger.info("Loading dataset...")
        df = pd.read_csv(file_path)

        # Drop unnecessary columns
        logger.info("Dropping unnecessary columns...")
        df.drop(columns=["customerId"], inplace=True)

        # Convert to proper types
        logger.info("Converting columns to appropriate data types...")
        df["amount"] = df["amount"].astype(float)
        df["customerAvgAmount"] = df["customerAvgAmount"].astype(float)
        df["customerTotalTransactions"] = df["customerTotalTransactions"].astype(int)
        df["geographicDistance"] = df["geographicDistance"].astype(float)

        # Extract target variable
        logger.info("Extracting target variable...")
        y_true = df["isSuspicious"].values
        df.drop(columns=["isSuspicious"], inplace=True)

        # Define categorical and numeric columns
        logger.info("Defining categorical and numeric columns...")
        categorical_cols = [
            "customerType", "transactionType", "originCountry", 
            "destinationCountry", "unusualForCustomer", "customerRiskProfile", 
            "transactionPurpose", "customerIndustry"
        ]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Ensure all categorical columns exist in the dataset
        missing_categorical_cols = [col for col in categorical_cols if col not in df.columns]
        if missing_categorical_cols:
            logger.warning(f"Missing categorical columns: {missing_categorical_cols}")
            categorical_cols = [col for col in categorical_cols if col in df.columns]

        # Preprocessing pipeline
        logger.info("Creating preprocessing pipeline...")
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=2))
        ])

        # Apply PCA transformation
        logger.info("Applying PCA transformation...")
        X_pca = pipeline.fit_transform(df)

        # Isolation Forest for anomaly detection
        logger.info("Fitting Isolation Forest for anomaly detection...")
        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        y_pred = iso_forest.fit_predict(X_pca)
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        accuracy = accuracy_score(y_true, y_pred_binary)
        logger.info(f"accuracy: {accuracy:.2f}")

        # PCA Plot
        logger.info("Generating PCA plot...")
        fig1 = plt.figure(figsize=(10, 5))
        plt.scatter(X_pca[y_pred_binary == 0, 0], X_pca[y_pred_binary == 0, 1], c='blue', s=10, label="Normal")
        plt.scatter(X_pca[y_pred_binary == 1, 0], X_pca[y_pred_binary == 1, 1], c='red', s=10, label="Anomaly")
        plt.title("Isolation Forest Anomaly Detection with PCA")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)

        buffer1 = BytesIO()
        plt.savefig(buffer1, format='png')
        buffer1.seek(0)
        pca_plot_base64 = base64.b64encode(buffer1.read()).decode()
        # plt.show()
        plt.close(fig1)

        # Correlation heatmap
        logger.info("Generating correlation heatmap...")
        correlation_matrix = df[numeric_cols].corr()
        fig2 = plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")

        buffer2 = BytesIO()
        plt.savefig(buffer2, format='png')
        buffer2.seek(0)
        heatmap_base64 = base64.b64encode(buffer2.read()).decode()
        # plt.show()
        plt.close(fig2)

        logger.info("Data processing completed successfully.")
        return {
            "accuracy": accuracy,
            "pca_plot_base64": pca_plot_base64,
            "heatmap_base64": heatmap_base64
        }

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise
