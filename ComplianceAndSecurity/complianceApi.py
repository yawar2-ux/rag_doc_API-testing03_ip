import base64
import json
import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form,APIRouter
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pydantic import BaseModel
from io import BytesIO
from typing import Optional
from datetime import datetime, timedelta, timezone
import uvicorn
from ComplianceAndSecurity.function import early_warning_system 
from ComplianceAndSecurity.visualize_dataset import process_data 

# Initialize the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


router = APIRouter()

class TransactionResponse(BaseModel):
    alert_type: Optional[str] = None
    severity: Optional[str] = None
    action_recommendation: Optional[str] = None
    transaction_details: Optional[dict] = None
    timestamp: Optional[str] = None

# Define the required columns for validation
REQUIRED_COLUMNS = {
    'amount', 'frequency30Day', 'customerAvgAmount', 'customerTotalTransactions',
    'geographicDistance', 'cityDistance', 'transactionType', 'unusualForCustomer',
    'customerRiskProfile', 'transactionPurpose', 'historicalRiskScore',
    'customerIndustry', 'originCountry', 'destinationCountry', 'originCity', 'destinationCity', 'customerType'
} 

# Load and prepare data
data = os.path.join(os.getcwd(), "ComplianceAndSecurity", "compliance_alerts.json")
df = pd.read_json(data)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.insert(0, 'Alert_ID', range(1, len(df) + 1))

# Load JSON data and convert to DataFrame
def load_data():
    data = os.path.join(os.getcwd(), "ComplianceAndSecurity", "compliance_alerts.json")
    file_path = data
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.json_normalize(data)
    # Specify the format to handle ISO 8601 strings with timezone offsets
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f%z', errors='coerce')
    return df

alerts_df = load_data()

# Helper function to convert plots to base64
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return base64_image

@router.post("/analyze_transaction", response_model=TransactionResponse)
async def analyze_transaction(
    amount: float = Form(...),
    frequency30Day: int = Form(...),
    customerAvgAmount: float = Form(...),
    customerTotalTransactions: int = Form(...),
    geographicDistance: float = Form(...),
    cityDistance: Optional[float] = Form(None),
    transactionType: Optional[str] = Form(...),
    unusualForCustomer: Optional[str] = Form(...),
    customerRiskProfile: Optional[str] = Form(...),
    transactionPurpose: Optional[str] = Form(...),
    historicalRiskScore: Optional[float] = Form(None),
    customerIndustry: Optional[str] = Form(...),
    originCountry: Optional[str] = Form(...),
    destinationCountry: Optional[str] = Form(...),
    originCity: Optional[str] = Form(...),
    destinationCity: Optional[str] = Form(...),
    customerType: Optional[str] = Form(...),
    customer_id: Optional[str] = Form(None)
):
    """
    Analyze a transaction for anomalies and compliance risks.
    """
    try:
        # Construct the transaction dictionary with correct keys
        transaction = {
            "amount": amount,
            "frequency30Day": frequency30Day,
            "customerAvgAmount": customerAvgAmount,
            "customerTotalTransactions": customerTotalTransactions,
            "geographicDistance": geographicDistance,
            "cityDistance": cityDistance,
            "transactionType": transactionType,
            "unusualForCustomer": unusualForCustomer,
            "customerRiskProfile": customerRiskProfile,
            "transactionPurpose": transactionPurpose,
            "historicalRiskScore": historicalRiskScore,
            "customerIndustry": customerIndustry,
            "originCountry": originCountry,
            "destinationCountry": destinationCountry,
            "originCity": originCity,
            "destinationCity": destinationCity,
            "customerType": customerType,
            "customer_id": customer_id,
        }

        # Log the transaction for debugging
        logger.info(f"Transaction data: {transaction}")

        # Call the EarlyWarningSystem's detect_anomalies method
        result = early_warning_system.detect_anomalies(transaction)

        # Log the result for debugging
        logger.info(f"Anomaly detection result: {result}")

        # Get the current timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if severity is present in the result
        severity = result.get("severity")
        if severity in ["HIGH", "MEDIUM", "LOW"]:
            response = {
                "alert_type": "Transaction Anomaly",
                "severity": severity,
                "action_recommendation": result.get("genai_analysis"),
                "transaction_details": transaction,
                "timestamp": timestamp
            }
            logger.info(f"Response: {response}")
            return response
        else:
            raise ValueError("Invalid severity level returned by the anomaly detection system.")

    except Exception as e:
        # Log the error and raise an HTTPException
        logger.error(f"Error analyzing transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing transaction: {str(e)}")



# Endpoint to analyze a dataset
@router.post("/analyze/")
async def analyze_dataset(file: UploadFile = File(None)):
    try:
        if file is None:
            # Return an error if no file is uploaded
            return JSONResponse(
                status_code=400,
                content={"error": "Please upload a dataset file."}
            )

        # Save the uploaded file as test_dataset.csv
        file_location = os.path.join(os.getcwd(), "ComplianceAndSecurity", "test_dataset.csv")
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(file_location)
        
        # Validate the uploaded dataset
        if not REQUIRED_COLUMNS.issubset(df.columns):
            os.remove(file_location)  # Clean up the file
            return JSONResponse(
                status_code=400,
                content={"error": "Uploaded dataset must contain the required columns: " + ", ".join(REQUIRED_COLUMNS)}
            )
        
        # Process the data using the file location
        results = process_data(file_location)
        snap_df = df.head(100)
        df_dict = snap_df.to_dict(orient="records")
        
        return JSONResponse(content={
            "accuracy": results["accuracy"],
            "pca_plot_base64": results["pca_plot_base64"],
            "heatmap_base64": results["heatmap_base64"],
            "dataframe": df_dict
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    

# Combined endpoint for alerts and visualizations
@router.get("/alerts_dashboard")
def alerts_dashboard():
    # Total alerts
    total_alerts = len(alerts_df)

    # Alerts over time
    alerts_over_time = alerts_df.resample('D', on='timestamp').size()
    fig1 = plt.figure(figsize=(10, 6))
    alerts_over_time.plot(title="Alerts Over Time", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Number of Alerts")
    plt.tight_layout()
    alerts_over_time_base64 = plot_to_base64(fig1)
    plt.close(fig1)

    # Alerts by severity
    severity_counts = alerts_df['severity'].value_counts()
    fig2 = plt.figure(figsize=(8, 6))
    severity_counts.plot(kind="bar", color=["red", "orange"], title="Alerts by Severity")
    plt.xlabel("Severity")
    plt.ylabel("Count")
    plt.tight_layout()
    alerts_by_severity_base64 = plot_to_base64(fig2)
    plt.close(fig2)

    # Save all alerts to CSV
    alert_path = os.path.join(os.getcwd(), "ComplianceAndSecurity", "all_alerts.csv")

    alerts_df.to_csv(alert_path, index=False)

    return {
        "total_alerts": total_alerts,
        "alerts_by_severity": severity_counts.to_dict(),
        "alerts_over_time_plot": alerts_over_time_base64,
        "alerts_by_severity_plot": alerts_by_severity_base64,
        "csv_file": "all_alerts.csv"
    }

# Search alerts by customer ID
# @router.get("/search_alerts-all-data/{customer_id}")
# def search_alerts_by_customer(customer_id: str):
#     try:
#         # Ensure the column exists and is of the correct type
#         if 'transaction_details.customer_id' not in alerts_df.columns:
#             raise HTTPException(status_code=500, detail="Column 'transaction_details.customer_id' not found in the dataset.")

#         # Convert the column to string for consistent comparison
#         alerts_df['transaction_details.customer_id'] = alerts_df['transaction_details.customer_id'].astype(str)

#         # Filter the DataFrame
#         customer_alerts = alerts_df[alerts_df['transaction_details.customer_id'] == customer_id]

#         # Check if the result is empty
#         if customer_alerts.empty:
#             raise HTTPException(status_code=404, detail=f"No alerts found for customer ID: {customer_id}")

#         # Replace NaN and infinite values
#         customer_alerts = customer_alerts.fillna("N/A")
#         customer_alerts = customer_alerts.replace([float("inf"), float("-inf")], "Infinity")

#         return customer_alerts.to_dict(orient="records")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




