from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from pydantic import BaseModel
import pandas as pd
import io
from typing import Optional, Dict, Any, List
from Customer360.functions import DataPreprocessor, ModelPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import base64
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from io import BytesIO
import numpy as np
import shap
import os

router = APIRouter()

print("customer api" , os.getenv("OLLAMA_BASE_URL"))

preprocessor = DataPreprocessor()
pipeline = ModelPipeline(
    ollama_url=os.getenv("OLLAMA_BASE_URL"),
    ollama_model=os.getenv("OLLAMA_TEXT_MODEL"),
)

processed_data = None
raw_data = None
customer_ids = None

def standardize_binary_column(series):
    if series.dtype == bool:
        return series.astype(int)
    series = series.astype(str).str.lower()
    positive_values = {'yes', 'true', '1', 't', 'y'}
    return series.isin(positive_values).astype(int)


def preprocess_binary_columns(df):
    binary_columns = ['Churn', 'Default']
    df_processed = df.copy()
    for col in binary_columns:
        if col in df_processed.columns:
            df_processed[col] = standardize_binary_column(df_processed[col])
    return df_processed

def plot_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    return base64.b64encode(image_png).decode()

def create_enhanced_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16, "weight": "bold"},
                linewidths=0.5, linecolor='black')
    plt.title('Confusion Matrix', fontsize=16, pad=15)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    return plot_to_base64(fig)

# def create_product_heatmap(y_test, y_pred, unique_products):
#     cm = confusion_matrix(y_test, y_pred, labels=unique_products)
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
#                 xticklabels=unique_products, yticklabels=unique_products,
#                 annot_kws={"size": 14, "weight": "bold"},
#                 linewidths=0.5, linecolor='black',
#                 square=True)
#     plt.title('Product Prediction Heatmap', fontsize=18, pad=20)
#     plt.xlabel('Predicted Product', fontsize=14)
#     plt.ylabel('Actual Product', fontsize=14)
#     plt.xticks(fontsize=12, rotation=45, ha='right')
#     plt.yticks(fontsize=12, rotation=0)
#     plt.tight_layout()
#     return plot_to_base64(fig)

def create_enhanced_cluster_plot(X_scaled, clusters, feature_names, cluster_mapping):
    if not isinstance(feature_names, list):
        raise ValueError("feature_names must be a list of feature names")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))
    segment_colors = {'Silver': 'silver', 'Platinum': 'blue', 'Gold': 'gold'}
    segment_labels = [cluster_mapping[cluster] for cluster in clusters]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                        c=[segment_colors[label] for label in segment_labels],
                        alpha=0.6, s=100, edgecolors='k')

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
              c=['red'], marker='x', s=200, linewidth=3,
              label='Centroids')

    ax.set_xlabel('First Principal Component', fontsize=12)
    ax.set_ylabel('Second Principal Component', fontsize=12)
    ax.set_title('Customer Segments (Silver, Platinum, Gold) - PCA Visualization', fontsize=16)
    plt.colorbar(scatter, label='Segment', ax=ax)
    ax.legend()

    explained_var = pca.explained_variance_ratio_
    ax.text(0.02, 0.98,
            f'Explained variance:\nPC1: {explained_var[0]:.2%}\nPC2: {explained_var[1]:.2%}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=10)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plot_to_base64(fig)

class AnalysisResponse(BaseModel):
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None
    insights: Optional[str] = None
    confusion_matrix: Optional[str] = None
    roc_curve: Optional[str] = None
    elbow_plot: Optional[str] = None
    cluster_plot: Optional[str] = None

class ProductAnalysisResponse(BaseModel):
    status: str
    message: str
    top_products: Optional[Dict[str, List]] = None
    insights: Optional[str] = None
    product_heatmap: Optional[str] = None
    customer_products: Optional[Dict[str, List]] = None


class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    status: str
    message: str
    response: Optional[str] = None

@router.post("/upload", response_model=AnalysisResponse)
async def upload_and_preprocess(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        global processed_data, raw_data, customer_ids

        if 'Customer_ID' in df.columns:
            customer_ids = df['Customer_ID'].astype(str).tolist()
        else:
            customer_ids = [f"CUST_{i}" for i in df.index]

        df = preprocess_binary_columns(df)
        raw_data = df.copy()

        if 'Customer_ID' in df.columns:
            df_no_id = df.drop(columns=['Customer_ID'])
            processed_df = preprocessor.preprocess(df_no_id)
            # Use pd.concat to avoid fragmentation
            processed_data = pd.concat([processed_df, pd.Series(customer_ids, name='Customer_ID')], axis=1)
        else:
            processed_df = preprocessor.preprocess(df)
            processed_data = pd.concat([processed_df, pd.Series(customer_ids, name='Customer_ID')], axis=1)

        return AnalysisResponse(
            status="success",
            message="Data uploaded and preprocessed successfully",
            results={"shape": [int(processed_data.shape[0]), int(processed_data.shape[1])]}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze/churn", response_model=AnalysisResponse)
async def analyze_churn():
    if processed_data is None or customer_ids is None:
        raise HTTPException(status_code=400, detail="No data has been uploaded and preprocessed")

    try:
        if 'Churn' not in processed_data.columns:
            raise HTTPException(status_code=400, detail="Churn column not found in data")

        analysis_data = processed_data.drop(columns=['Customer_ID'])
        target_columns = ['Churn', 'Default', 'Product']
        feature_columns = [col for col in analysis_data.columns if col not in target_columns]
        X = analysis_data[feature_columns]
        y = analysis_data['Churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        avg_churn_rate = float(y.mean() * 100)
        total_customers = int(len(y))
        total_churn_predicted = int(sum(model.predict(X)))
        accuracy = float(model.score(X_test, y_test))
        probability_list = {cid: float(prob) for cid, prob in zip(customer_ids, y_pred_proba)}

        results = {
            "average_churn_rate": f"{avg_churn_rate:.2f}%",
            "total_customers": total_customers,
            "total_churn_predicted": total_churn_predicted,
            "churn_probability_by_customer": probability_list
        }

        conf_matrix_base64 = create_enhanced_confusion_matrix(y_test, y_pred, ['Not Churned', 'Churned'])

        insights_prompt = (
            f"Churn analysis results: Average churn rate: {avg_churn_rate:.2f}%, "
            f"Total customers: {total_customers}, Total predicted churn: {total_churn_predicted}, "
            f"Model accuracy: {accuracy:.2f}, Confusion matrix: TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)}. "
            "Provide a detailed explanation in 6 points using these exact numbers. For each point: "
            "1) Reference specific numbers from the results and confusion matrix, "
            "2) Explain what these numbers indicate about customer churn behavior, "
            "3) Provide a reason why this insight is meaningful for business decisions. "
            "For the 6th point, suggest a specific business strategy based on these churn insights."
        )
        insights = pipeline.get_ai_insights(insights_prompt)

        return AnalysisResponse(
            status="success",
            message="Churn analysis completed",
            results=results,
            insights=insights,
            confusion_matrix=conf_matrix_base64
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze/default",  response_model=AnalysisResponse)
async def analyze_default():
    if processed_data is None or customer_ids is None:
        raise HTTPException(status_code=400, detail="No data has been uploaded and preprocessed")

    try:
        if 'Default' not in processed_data.columns:
            raise HTTPException(status_code=400, detail="Default column not found in data")

        analysis_data = processed_data.drop(columns=['Customer_ID'])
        target_columns = ['Churn', 'Default', 'Product']
        feature_columns = [col for col in analysis_data.columns if col not in target_columns]
        X = analysis_data[feature_columns]
        y = analysis_data['Default']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        avg_default_rate = float(y.mean() * 100)
        total_customers = int(len(y))
        total_default_predicted = int(sum(model.predict(X)))
        accuracy = float(model.score(X_test, y_test))
        probability_list = {cid: float(prob) for cid, prob in zip(customer_ids, y_pred_proba)}

        results = {
            "average_default_rate": f"{avg_default_rate:.2f}%",
            "total_customers": total_customers,
            "total_default_predicted": total_default_predicted,
            "default_probability_by_customer": probability_list
        }

        conf_matrix_base64 = create_enhanced_confusion_matrix(y_test, y_pred, ['Not Defaulted', 'Defaulted'])

        insights_prompt = (
            f"Default analysis results: Average default rate: {avg_default_rate:.2f}%, "
            f"Total customers: {total_customers}, Total predicted defaults: {total_default_predicted}, "
            f"Model accuracy: {accuracy:.2f}, Confusion matrix: TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)}. "
            "Provide a detailed explanation in 6 points using these exact numbers. For each point: "
            "1) Reference specific numbers from the results and confusion matrix, "
            "2) Explain what these numbers indicate about customer default behavior, "
            "3) Provide a reason why this insight is meaningful for business decisions. "
            "For the 6th point, suggest a specific business strategy based on these default insights."
        )
        insights = pipeline.get_ai_insights(insights_prompt)

        return AnalysisResponse(
            status="success",
            message="Default analysis completed",
            results=results,
            insights=insights,
            confusion_matrix=conf_matrix_base64
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze/product", response_model=ProductAnalysisResponse)
async def analyze_product():
    if processed_data is None or customer_ids is None:
        raise HTTPException(status_code=400, detail="No data has been uploaded and preprocessed")

    try:
        if 'Product' not in processed_data.columns:
            raise HTTPException(status_code=400, detail="Product column not found in data")

        analysis_data = processed_data.drop(columns=['Customer_ID'])
        target_columns = ['Churn', 'Default', 'Product']
        feature_columns = [col for col in analysis_data.columns if col not in target_columns]
        X = analysis_data[feature_columns]
        y = analysis_data['Product']

        print(f"Full dataset product labels: {y.unique()}")
        print(f"Dataset size: {len(y)}")

        if len(y) < 5:
            raise HTTPException(status_code=400, detail="Dataset too small for analysis (minimum 5 samples required)")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"y_train unique: {set(y_train.unique())}")
        print(f"y_test unique: {set(y_test.unique())}")

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"y_pred unique: {set(y_pred)}")
        predicted_products = model.predict(X)

        product_counts = pd.Series(raw_data['Product']).value_counts()
        product_names = product_counts.index.tolist()
        all_products = {
            'products': product_names,
            'percentages': [float(p) for p in (product_counts / len(raw_data) * 100).tolist()]
        }
        accuracy = float(model.score(X_test, y_test))

        product_mapping = preprocessor.product_mapping
        if not product_mapping:
            raise HTTPException(status_code=500, detail="Product mapping is empty. Ensure preprocessing was successful.")
        print(f"Product mapping: {product_mapping}")

        y_test = y_test.astype(int).tolist()
        y_pred = y_pred.astype(int).tolist()
        print(f"y_test contents: {y_test}")
        print(f"y_pred contents: {y_pred}")

        unique_pred = set(predicted_products)
        missing_labels = unique_pred - set(product_mapping.keys())

        present_labels = sorted(set(y_test))
        print(f"Present labels: {present_labels}")

        if len(y_test) != len(y_pred):
            raise ValueError(f"Length mismatch: y_test ({len(y_test)}) vs y_pred ({len(y_pred)})")

        cm_auto = confusion_matrix(y_test, y_pred)
        auto_labels = sorted(set(y_test).union(set(y_pred)))
        print(f"Auto-detected labels: {auto_labels}")
        print(f"Auto confusion matrix:\n{cm_auto}")

        all_labels = [0, 1, 2, 3, 4]
        cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

        for true_label, pred_label in zip(y_test, y_pred):
            if true_label in label_to_idx and pred_label in label_to_idx:
                cm[label_to_idx[true_label], label_to_idx[pred_label]] += 1

        print(f"Manual confusion matrix:\n{cm}")

        if missing_labels:
            print(f"Warning: Predicted labels {missing_labels} not in product_mapping. Assigning 'Unknown'.")
            customer_products = {
                cid: [product_mapping.get(pred, "Unknown")]
                for cid, pred in zip(customer_ids, predicted_products)
            }
            cm_dict = {
                product_mapping.get(label, f"Unknown_{label}"): [int(x) for x in cm[i].tolist()]
                for i, label in enumerate(all_labels)
            }
            unique_products = [product_mapping.get(label, f"Unknown_{label}") for label in all_labels]
        else:
            customer_products = {
                cid: [product_mapping[pred]]
                for cid, pred in zip(customer_ids, predicted_products)
            }
            cm_dict = {
                product_mapping[label]: [int(x) for x in cm[i].tolist()]
                for i, label in enumerate(all_labels)
            }
            unique_products = [product_mapping[label] for label in all_labels]

        # Pass integer labels to confusion_matrix, use strings for display only
        product_heatmap_base64 = create_product_heatmap(y_test, y_pred, all_labels, unique_products)

        insights_prompt = (
            f"Product analysis results: Products: {all_products['products']}, "
            f"Percentages: {all_products['percentages']}, Total customers: {int(len(customer_ids))}, "
            f"Model accuracy: {accuracy:.2f}, Confusion matrix: {cm_dict}. "
            "Provide a detailed explanation in 6 points using these exact numbers. For each point: "
            "1) Reference specific numbers from the results and confusion matrix, "
            "2) Explain what these numbers indicate about product preferences, "
            "3) Provide a reason why this insight is meaningful for business decisions. "
            "For the 6th point, suggest a specific business strategy based on these product insights."
        )
        insights = pipeline.get_ai_insights(insights_prompt)

        return ProductAnalysisResponse(
            status="success",
            message="Product analysis completed",
            top_products=all_products,
            insights=insights,
            product_heatmap=product_heatmap_base64,
            customer_products=customer_products
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Product analysis failed: {str(e)}")

@router.post("/analyze/profitability", response_model=AnalysisResponse)
async def analyze_profitability():
    if raw_data is None or customer_ids is None:
        raise HTTPException(status_code=400, detail="No data has been uploaded and preprocessed")

    try:
        possible_columns = {
            'recency': ['Recency', 'recency', 'R', 'days_since_last_purchase'],
            'frequency': ['Frequency', 'frequency', 'F', 'purchase_count'],
            'monetary': ['Monetary', 'monetary', 'M', 'spend', 'revenue'],
            'clv': ['CLV', 'clv', 'Customer_Lifetime_Value', 'lifetime_value']
        }

        required_columns = {}
        for key, aliases in possible_columns.items():
            for alias in aliases:
                if alias in raw_data.columns:
                    required_columns[key] = alias
                    break
            if key not in required_columns:
                raise HTTPException(status_code=400, detail=f"Could not find a column matching {key} in the dataset")

        def preprocess_dynamic_data(df, required_cols):
            processed_df = df.copy()
            for col_name in required_cols.values():
                if col_name in processed_df.columns:
                    processed_df[col_name] = pd.to_numeric(processed_df[col_name], errors='coerce')
                    if processed_df[col_name].isna().any():
                        median_value = processed_df[col_name].median()
                        if pd.isna(median_value):
                            processed_df[col_name] = processed_df[col_name].fillna(0)
                        else:
                            processed_df[col_name] = processed_df[col_name].fillna(median_value)
                    if col_name in [required_columns['monetary'], required_columns['frequency'], required_columns['clv']]:
                        processed_df[col_name] = processed_df[col_name].clip(lower=0)
            processed_df = processed_df.dropna(subset=list(required_columns.values()))
            return processed_df

        preprocessed_data = preprocess_dynamic_data(raw_data, required_columns)
        profitability_features = preprocessed_data[list(required_columns.values())]
        profitability_target = profitability_features[required_columns['monetary']] * profitability_features[required_columns['frequency']]

        if profitability_target.isna().any():
            raise HTTPException(status_code=400, detail="Profitability target contains NaN values after preprocessing")

        X_train, X_test, y_train, y_test = train_test_split(
            profitability_features, profitability_target, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        total_customers = int(len(preprocessed_data))
        overall_monetary = float(profitability_features[required_columns['monetary']].mean())
        overall_recency = float(profitability_features[required_columns['recency']].mean())
        overall_frequency = float(profitability_features[required_columns['frequency']].mean())
        overall_clv = float(profitability_features[required_columns['clv']].mean())
        total_monetary_sum = float(profitability_features[required_columns['monetary']].sum())
        avg_profitability = float((profitability_target.sum() / total_monetary_sum) * 100)
        r2 = float(r2_score(y_test, y_pred))

        customer_id_col = next((alias for alias in ['Customer_ID', 'customer_id', 'ID', 'cust_id'] if alias in preprocessed_data.columns), None)

        if customer_id_col:
            per_customer_metrics = preprocessed_data.groupby(customer_id_col)[list(required_columns.values())].mean()
            per_customer_monetary = float(per_customer_metrics[required_columns['monetary']].mean())
            per_customer_recency = float(per_customer_metrics[required_columns['recency']].mean())
            per_customer_frequency = float(per_customer_metrics[required_columns['frequency']].mean())
            per_customer_clv = float(per_customer_metrics[required_columns['clv']].mean())
            per_customer_profitability = float((preprocessed_data.groupby(customer_id_col)[[required_columns['monetary'], required_columns['frequency']]].mean().prod(axis=1)).mean())
            per_customer_values = preprocessed_data.groupby(customer_id_col)[list(required_columns.values())].mean()
            per_customer_profitability_values = preprocessed_data.groupby(customer_id_col)[[required_columns['monetary'], required_columns['frequency']]].mean().prod(axis=1)
            customer_averages = {
                str(cust_id): {
                    "monetary": float(per_customer_values.loc[cust_id, required_columns['monetary']]),
                    "recency": float(per_customer_values.loc[cust_id, required_columns['recency']]),
                    "frequency": float(per_customer_values.loc[cust_id, required_columns['frequency']]),
                    "clv": float(per_customer_values.loc[cust_id, required_columns['clv']]),
                    "profitability": float(per_customer_profitability_values.loc[cust_id])
                } for cust_id in preprocessed_data[customer_id_col].astype(str).tolist()
            }
        else:
            per_customer_monetary = overall_monetary
            per_customer_recency = overall_recency
            per_customer_frequency = overall_frequency
            per_customer_clv = overall_clv
            per_customer_profitability = float((profitability_features[required_columns['monetary']] * profitability_features[required_columns['frequency']]).mean())
            customer_averages = {
                cid: {
                    "monetary": overall_monetary,
                    "recency": overall_recency,
                    "frequency": overall_frequency,
                    "clv": overall_clv,
                    "profitability": per_customer_profitability
                } for cid in customer_ids
            }

        results = {
            "average_profitability": avg_profitability,
            "overall_monetary": overall_monetary,
            "overall_recency": overall_recency,
            "overall_frequency": overall_frequency,
            "overall_clv": overall_clv,
            "customer_averages": customer_averages
        }

        insights_prompt = (
            f"Profitability analysis results: Average profitability: {avg_profitability:.2f}%, "
            f"Overall averages: Monetary {overall_monetary:.2f}, Recency {overall_recency:.2f}, "
            f"Frequency {overall_frequency:.2f}, CLV {overall_clv:.2f}, "
            f"Per customer averages: Profitability {per_customer_profitability:.2f}, Monetary {per_customer_monetary:.2f}, "
            f"Recency {per_customer_recency:.2f}, Frequency {per_customer_frequency:.2f}, CLV {per_customer_clv:.2f}, "
            f"Total customers: {total_customers}, Model R²: {r2:.2f}. "
            "Provide a detailed explanation in 6 points using these exact numbers. For each point: "
            "1) Reference specific numbers from the results, "
            "2) Explain what these numbers indicate about customer profitability, "
            "3) Provide a reason why this insight is meaningful for business decisions. "
            "For the 6th point, suggest a specific business strategy based on these profitability insights."
        )
        insights = pipeline.get_ai_insights(insights_prompt)

        return AnalysisResponse(
            status="success",
            message="Profitability analysis completed",
            results=results,
            insights=insights
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def create_product_heatmap(y_test, y_pred, numeric_labels, display_labels):
    cm = confusion_matrix(y_test, y_pred, labels=numeric_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=display_labels, yticklabels=display_labels,
                annot_kws={"size": 14, "weight": "bold"},
                linewidths=0.5, linecolor='black',
                square=True)
    plt.title('Product Prediction Heatmap', fontsize=18, pad=20)
    plt.xlabel('Predicted Product', fontsize=14)
    plt.ylabel('Actual Product', fontsize=14)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    return plot_to_base64(fig)

@router.post("/analyze/clustering", response_model=AnalysisResponse)
async def analyze_clustering():
    if processed_data is None or customer_ids is None:
        raise HTTPException(status_code=400, detail="No data has been uploaded and preprocessed")

    try:
        analysis_data = processed_data.drop(columns=['Customer_ID'])
        expected_numerical_columns = ['Recency', 'Frequency', 'Monetary', 'CLV']
        numerical_columns = [col for col in expected_numerical_columns if col in analysis_data.columns and analysis_data[col].dtype in ['float64', 'int64']]

        if len(numerical_columns) < 2:
            raise HTTPException(status_code=400, detail="Insufficient numerical columns for clustering (minimum 2 required)")

        X_cluster = analysis_data[numerical_columns].copy()

        if X_cluster.isna().any().any():
            X_cluster = X_cluster.fillna(X_cluster.median())
            if X_cluster.isna().any().any():
                raise HTTPException(status_code=400, detail="Data contains unresolvable NaN values after preprocessing")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        if X_scaled.shape[0] < 3 or X_scaled.shape[1] < 1:
            raise HTTPException(status_code=400, detail="Data shape too small for clustering (min 3 samples, 1 feature)")

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        unique, counts = np.unique(clusters, return_counts=True)
        cluster_sizes = dict(zip(unique, [int(c) for c in counts]))
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        cluster_mapping = {
            sorted_clusters[0][0]: "Silver",
            sorted_clusters[1][0]: "Platinum",
            sorted_clusters[2][0]: "Gold"
        }
        customer_segments = [cluster_mapping[cluster] for cluster in clusters]

        cluster_centroids = {}
        for cluster_label in unique:
            cluster_data = X_cluster[clusters == cluster_label]
            centroid = {col: float(cluster_data[col].mean()) for col in numerical_columns}
            cluster_centroids[cluster_mapping[cluster_label]] = centroid

        all_customers_with_segments = {}
        for cid, segment, row in zip(customer_ids, customer_segments, X_cluster.itertuples(index=False)):
            customer_values = {col: float(getattr(row, col)) for col in numerical_columns}
            centroid_values = cluster_centroids[segment]

            differences = {
                col: abs(customer_values[col] - centroid_values[col]) / (centroid_values[col] + 1e-6)
                for col in numerical_columns
            }
            key_features = sorted(differences.items(), key=lambda x: x[1])[:2]

            reason_parts = []
            for feature, _ in key_features:
                cust_val = customer_values[feature]
                cent_val = centroid_values[feature]
                if feature == "Recency":
                    if cust_val < cent_val:
                        reason_parts.append("bought recently")
                    else:
                        reason_parts.append("hasn’t bought lately")
                elif feature == "Frequency":
                    if cust_val > cent_val:
                        reason_parts.append("buys often")
                    else:
                        reason_parts.append("buys occasionally")
                elif feature == "Monetary":
                    if cust_val > cent_val:
                        reason_parts.append("spends a lot")
                    else:
                        reason_parts.append("spends a little")
                elif feature == "CLV":
                    if cust_val > cent_val:
                        reason_parts.append("has high lifetime value")
                    else:
                        reason_parts.append("has moderate lifetime value")

            reason = f"{cid} fits {segment} because they {reason_parts[0]} and {reason_parts[1]}."
            all_customers_with_segments[cid] = {"segment": segment, "reason": reason}

        total_customers = int(len(customer_ids))
        segment_counts = {
            "Silver": int(cluster_sizes[sorted_clusters[0][0]]),
            "Platinum": int(cluster_sizes[sorted_clusters[1][0]]),
            "Gold": int(cluster_sizes[sorted_clusters[2][0]])
        }
        segment_percentages = {seg: float(count / total_customers * 100) for seg, count in segment_counts.items()}

        cluster_plot_base64 = create_enhanced_cluster_plot(X_scaled, clusters, numerical_columns, cluster_mapping)

        explainer = shap.KernelExplainer(kmeans.predict, X_scaled)
        shap_values = explainer.shap_values(X_scaled)
        shap_summary = {col: float(np.abs(shap_values[:, i]).mean()) for i, col in enumerate(numerical_columns)}
        shap_importance = dict(sorted(shap_summary.items(), key=lambda x: x[1], reverse=True))

        results = {
            "total_customers": total_customers,
            "customer_segments": all_customers_with_segments,
            "segment_distribution": {
                "Silver": {"count": segment_counts["Silver"], "percentage": segment_percentages["Silver"]},
                "Platinum": {"count": segment_counts["Platinum"], "percentage": segment_percentages["Platinum"]},
                "Gold": {"count": segment_counts["Gold"], "percentage": segment_percentages["Gold"]}
            }
        }

        insights_prompt = (
            f"Clustering results: Total customers {total_customers}, Segments: Silver {segment_counts['Silver']} ({segment_percentages['Silver']:.1f}%), "
            f"Platinum {segment_counts['Platinum']} ({segment_percentages['Platinum']:.1f}%), Gold {segment_counts['Gold']} ({segment_percentages['Gold']:.1f}%), "
            f"Features: {numerical_columns}, Centroids: {cluster_centroids}, SHAP importance: {shap_importance}. "
            "Provide concise 6-point insights using these numbers. For each point: "
            "1) Cite a specific number or value, "
            "2) Briefly explain its segment or feature impact, "
            "3) Note its business relevance. "
            "For the 6th point, propose a brief strategy using these insights."
        )
        insights = pipeline.get_ai_insights(insights_prompt)

        return AnalysisResponse(
            status="success",
            message="Clustering analysis completed",
            results=results,
            insights=insights,
            cluster_plot=cluster_plot_base64
        )
    except Exception as e:
        print(f"Clustering error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Clustering failed: {str(e)}")


# @router.post("/chat", response_model=ChatResponse)
# async def chat_with_data(request: ChatRequest):
#     if processed_data is None or customer_ids is None:
#         raise HTTPException(status_code=400, detail="No data has been uploaded and preprocessed")

#     try:
#         # Limit dataset to top 100 rows (as per your code)
#         limited_data = processed_data.head(100).copy()
#         limited_customer_ids = customer_ids[:100]

#         # Convert limited dataset to a string representation for context
#         data_context = limited_data.to_string(index=False)

#         # Check for "thank you" to end the chat
#         query_lower = request.query.lower().strip()
#         thank_you_variations = ["thank you", "thanks", "thx", "thank u"]
#         if any(thank_phrase in query_lower for thank_phrase in thank_you_variations):
#             return ChatResponse(
#                 status="success",
#                 message="Chat ended",
#                 response="You're welcome! The chat session has ended. Feel free to start a new one if you have more questions about the dataset."
#             )

#         # Check if query seems to be outside the dataset scope
#         outside_keywords = [
#             "weather", "news", "stock market", "global", "world", "internet",
#             "outside", "external", "real-time", "search", "web"
#         ]
#         if any(keyword in query_lower for keyword in outside_keywords):
#             return ChatResponse(
#                 status="error",
#                 message="This chatbot can only respond to queries based on the uploaded dataset (top 100 rows). External or unrelated queries are not supported."
#             )

#         # Improved prompt for concise, dataset-based responses
#         prompt = (
#             f"You are a smart, business-focused chatbot that answers queries dynamically using only this dataset (top 100 rows):\n\n"
#             f"{data_context}\n\n"
#             f"Query: {request.query}\n\n"
#             f"Provide a concise, insightful response in 6-7 sentences, using key numbers and their business meaning from the dataset. "
#             f"Avoid step-by-step calculations unless explicitly requested, and keep it relevant to customer analysis. "
#             f"If the query cannot be answered with the dataset, respond: 'This query cannot be answered with the provided dataset.'"
#         )

#         # Get response from the ModelPipeline
#         response = pipeline.get_ai_insights(prompt)

#         return ChatResponse(
#             status="success",
#             message="Query processed successfully",
#             response=response
#         )
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error processing chat request: {str(e)}")
