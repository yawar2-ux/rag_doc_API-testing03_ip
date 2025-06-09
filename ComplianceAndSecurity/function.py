import os
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from datetime import datetime, timedelta, timezone
import time
from sklearn.metrics import accuracy_score

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Groq API Key is missing. Set GROQ_API_KEY in .env file.")

# ---------------------- 1. Load & Preprocess AML Dataset ----------------------
# Define the dataset path
data = os.path.join(os.getcwd(), "ComplianceAndSecurity", "test_dataset.csv")
if data:
    logger.info(f"Loading dataset from {data}")
else:
    logger.info("Initially moving without test_dataset.csv")
def load_and_preprocess_data(file_path=data):
    """
    Load dataset and preprocess it for anomaly detection.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Validate data
    required_columns = [
        'amount', 'frequency30Day', 'customerAvgAmount', 'customerTotalTransactions',
        'geographicDistance', 'cityDistance', 'transactionType', 'unusualForCustomer',
        'customerRiskProfile', 'transactionPurpose', 'historicalRiskScore',
        'customerIndustry', 'originCountry', 'destinationCountry', 'originCity', 'destinationCity', 'customerType'
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}")

    # Select numeric features
    numeric_features = [
        'amount', 'frequency30Day', 'customerAvgAmount', 'customerTotalTransactions',
        'geographicDistance', 'cityDistance', 'historicalRiskScore'
    ]
    df_numeric = df[numeric_features]

    # Encode categorical features
    categorical_features = [
        'transactionType', 'unusualForCustomer', 'customerRiskProfile', 'transactionPurpose',
        'customerIndustry', 'originCountry', 'destinationCountry', 'originCity', 'destinationCity', 'customerType'
    ]
    encoder = OneHotEncoder(sparse_output=False)
    df_categorical = encoder.fit_transform(df[categorical_features])
    df_categorical = pd.DataFrame(df_categorical, columns=encoder.get_feature_names_out(categorical_features))

    # Combine numeric and encoded categorical features
    df_combined = pd.concat([df_numeric, df_categorical], axis=1)

    # Normalize data
    scaler = StandardScaler()
    df_combined_scaled = scaler.fit_transform(df_combined)

    return df, df_combined_scaled, scaler, encoder

df, df_scaled, scaler, encoder = load_and_preprocess_data()

# ---------------------- 2. Train Anomaly Detection Model ----------------------

def train_anomaly_model(data):
    """
    Train an Isolation Forest model for anomaly detection.
    """
    try:
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(data)
    except Exception as e:
        logger.error(f"Error training anomaly detection model: {e}")
        raise
    return model

anomaly_model = train_anomaly_model(df_scaled)

# Evaluation
def evaluate_model_with_accuracy(model, data, labels):
    """
    Evaluate the Isolation Forest model using accuracy.
    """
    try:
        predictions = model.predict(data)
        predictions = [1 if p == -1 else 0 for p in predictions]  # Convert -1 to 1 (anomaly), 1 to 0 (normal)
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, predictions)
        logger.info(f"Model Evaluation - Accuracy: {accuracy}")
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

# ---------------------- 3. GenAI AML Analysis ----------------------

class ComplianceAIAnalyzer:
    def __init__(self, groq_api_key):
        """
        Initialize GenAI-based Compliance Analyzer using Groq API.
        """
        self.llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")
        self.last_api_call_time = 0
        self.api_call_interval = 1  # Minimum interval between API calls in seconds

        # Compliance analysis prompt template
        self.compliance_prompt = PromptTemplate(
            input_variables=["transaction_data", "regulatory_context"],
            template="""
            Analyze the following transaction data in the context of AML/KYC regulations:
            
            **Transaction Data:**
            {transaction_data}
            
            **Regulatory Context:**
            {regulatory_context}
            
            **Key Considerations:**
            - All amounts and transactions must be analyzed in INR (Indian Rupees) (use "INR" explicitly instead of symbols).
            - All distances must be analyzed in kilometers (kms).
            - All provided transaction fields must be used in the analysis, even if they do not have a significant impact.
            - Consider customer-specific risk factors such as PEP (Politically Exposed Person) status, high-risk jurisdictions, and sanctions list matches.
            - Evaluate transaction patterns for anomalies, including velocity, round-number transactions, and repeated transfers.
            - Assess counterparty details, including risk profiles and relationships with the customer.
            - Identify potential structuring, smurfing, or layering activities.

            **Analysis Requirements:**
            1. Provide a detailed overview of all transaction fields and their relevance to AML/KYC compliance.
            2. Highlight potential compliance risks, including high-risk jurisdictions, unusual transaction patterns, or deviations from customer profiles.
            3. Identify anomaly detection flags based on transaction data and patterns.
            4. Recommend specific actions to mitigate identified risks (e.g., enhanced due diligence, SAR filing).
            5. Determine the severity of potential violations (e.g., HIGH, MEDIUM, LOW) and justify the classification.
            6. Highlight specific regulatory concerns, such as violations of thresholds, suspicious patterns, or high-risk counterparties.
            7. Suggest whether the transaction warrants escalation or further investigation.

            Ensure that the analysis is consistent with the provided transaction data and regulatory context. No fields should be ignored, and the output must be actionable and aligned with AML/KYC best practices.
            """
        )

        self.compliance_chain = self.compliance_prompt | self.llm | StrOutputParser()

    def analyze_transaction(self, transaction_data, regulatory_context):
        """
        Analyze a transaction for compliance risks using GenAI.
        """
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_api_call_time < self.api_call_interval:
            time.sleep(self.api_call_interval - (current_time - self.last_api_call_time))
        self.last_api_call_time = time.time()

        try:
            # Convert transaction data to a formatted string for the prompt
            transaction_data_str = "\n".join([f"{key}: {value}" for key, value in transaction_data.items()])
            print("Transaction Data for Analysis:")
            print(transaction_data_str)

            # Prepare the context for the prompt
            context = {
                "transaction_data": transaction_data_str,
                "regulatory_context": regulatory_context
            }

            # Invoke the compliance chain
            analysis = self.compliance_chain.invoke(context)

            # Replace Unicode INR symbol with "INR"
            analysis = analysis.replace("\u20b9", "INR")

            return analysis
        except Exception as e:
            logger.error(f"Error during GenAI analysis: {e}")
            raise

ai_analyzer = ComplianceAIAnalyzer(GROQ_API_KEY)

# ---------------------- 4. Early Warning System ----------------------

class EarlyWarningSystem:
    def __init__(self, ai_analyzer, model):
        """
        Initialize the Early Warning System.
        """
        self.ai_analyzer = ai_analyzer
        self.model = model
        self.alerts_file = os.path.join(os.getcwd(), "ComplianceAndSecurity", "compliance_alerts.json")

        # Ensure alerts file exists
        if not os.path.exists(self.alerts_file):
            with open(self.alerts_file, 'w') as f:
                json.dump([], f)

    def log_alert(self, alert_details):
        """
        Log compliance alerts to a JSON file.
        """
        logger.info("log_alert called with alert: %s", alert_details)
        try:
            # Try to read existing alerts
            with open(self.alerts_file, 'r') as f:
                alerts = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Handle empty or invalid JSON or missing file
            alerts = []

        # Get current UTC time with timezone info
        utc_now = datetime.now(timezone.utc)
        # Convert UTC to IST (UTC+5:30)
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        # Store timestamp in ISO format
        alert_details['timestamp'] = ist_now.isoformat()

        # Append the new alert
        alerts.append(alert_details)

        # Write back to the file
        with open(self.alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)

        logger.info(f"New Alert Logged: {alert_details.get('severity', 'UNKNOWN')} - {alert_details.get('description', 'No description')}")

    def get_recent_alerts(self, hours=24):
        """
        Retrieve recent alerts.
        """
        with open(self.alerts_file, 'r') as f:
            all_alerts = json.load(f)

        # Convert cutoff time to IST
        # Get current UTC time with timezone info
        utc_now = datetime.now(timezone.utc)
        # Calculate 24 hours ago in UTC
        cutoff_time_utc = utc_now - timedelta(hours=24)
        # Convert UTC to IST
        cutoff_time_ist = cutoff_time_utc + timedelta(hours=5, minutes=30)

        return [alert for alert in all_alerts if datetime.fromisoformat(alert['timestamp']) >= cutoff_time_ist]

    def detect_anomalies(self, transaction):
        """
        Detect anomalies in transactions using ML and GenAI.
        """
        try:
            # Validate transaction data
            required_keys = [
                'amount', 'frequency30Day', 'customerAvgAmount', 'customerTotalTransactions',
                'geographicDistance', 'cityDistance', 'transactionType', 'unusualForCustomer',
                'customerRiskProfile', 'transactionPurpose', 'historicalRiskScore',
                'customerIndustry', 'originCountry', 'destinationCountry', 'originCity', 'destinationCity', 'customerType'
            ]
            if not all(key in transaction for key in required_keys):
                raise ValueError(f"Transaction data must contain {required_keys}.")

            # Prepare transaction features
            numeric_features = [
                'amount', 'frequency30Day', 'customerAvgAmount', 'customerTotalTransactions',
                'geographicDistance', 'cityDistance', 'historicalRiskScore'
            ]
            categorical_features = [
                'transactionType', 'unusualForCustomer', 'customerRiskProfile', 'transactionPurpose',
                'customerIndustry', 'originCountry', 'destinationCountry', 'originCity', 'destinationCity', 'customerType'
            ]

            transaction_numeric = pd.DataFrame([transaction], columns=numeric_features)
            transaction_categorical = self.encoder.transform(pd.DataFrame([transaction], columns=categorical_features))
            transaction_categorical = pd.DataFrame(transaction_categorical, columns=self.encoder.get_feature_names_out(categorical_features))

            transaction_features = pd.concat([transaction_numeric, transaction_categorical], axis=1)
            transaction_features_scaled = scaler.transform(transaction_features)

            # Predict anomaly score
            anomaly_score = self.model.predict(transaction_features_scaled)[0]
            is_suspicious_ml = 1 if anomaly_score == -1 else 0

            # Use GenAI for analysis
            regulatory_context = """
            The transaction must be evaluated in accordance with global Anti-Money Laundering (AML) and Know Your Customer (KYC) regulations. These regulations require financial institutions to monitor and report suspicious activity that may indicate money laundering, terrorist financing, or other financial crimes.

            Key regulatory guidelines include:
            - Customer due diligence (CDD) and enhanced due diligence (EDD) for high-risk customers or jurisdictions.
            - Monitoring of large or unusual transactions, especially those involving offshore financial centers, high-risk countries, or politically exposed persons (PEPs).
            - Identification of patterns such as rapid movement of funds, frequent transactions just under reporting thresholds, and transactions with no apparent economic or lawful purpose.
            - Record-keeping and prompt reporting of suspicious activity reports (SARs) to regulatory authorities.

            Please assess the transaction for any indicators of potential non-compliance with these rules, and suggest necessary actions for risk mitigation.
            """
            ai_analysis = self.ai_analyzer.analyze_transaction(transaction, regulatory_context)

            # Refined severity determination
            if is_suspicious_ml:
                severity = "HIGH"
            elif "high risk" in ai_analysis.lower():
                severity = "MEDIUM"
            elif transaction['amount'] > 50000 or transaction['unusualForCustomer'].lower() == 'yes':
                severity = "MEDIUM"
            elif transaction['historicalRiskScore'] > 50 or transaction['frequency30Day'] > 40:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            # Log alerts if flagged as suspicious
            if severity in ["HIGH", "MEDIUM", "LOW"]:
                self.log_alert({
                    'alert_type': 'Transaction Anomaly',
                    'severity': severity,
                    'action_recommendation': ai_analysis,
                    'transaction_details': transaction
                })

            return {"is_suspicious_ml": is_suspicious_ml, "genai_analysis": ai_analysis, "severity": severity}
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise

early_warning_system = EarlyWarningSystem(ai_analyzer, anomaly_model)
early_warning_system.encoder = encoder 


