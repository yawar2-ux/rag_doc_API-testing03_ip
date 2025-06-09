from click import Tuple
import pandas as pd
import numpy as np
import requests
import os
import json
from typing import List, Dict
import plotly.express as px
from ydata_profiling import ProfileReport
import base64
import io
import re
from typing import Any, Type, List,Tuple

class FraudDetectionAssistant:
    def __init__(self):
        # Initialize states for file data, reports, and history
        self.chat_history = {}
        self.uploaded_files_data = {}
        self.profile_reports = {}
        self.current_dataset = None

    def validate_file_size(self, file_size) -> bool:
        MAX_FILE_SIZE = 50 * 1024 * 1024
        return file_size <= MAX_FILE_SIZE

    def validate_fraud_dataset(self, df) -> tuple[bool, str]:
        required_columns = ['transaction_id', 'amount', 'timestamp', 'merchant', 'is_fraudulent']
        
        # Check if minimum required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Validate column names (checking if 'is_fraudulent' exists as it is used in your code)
        if 'is_fraudulent' not in df.columns:
            return False, "'is_fraudulent' column is missing, please ensure this column exists and is properly named."
        
        # Validate data types
        if not pd.api.types.is_numeric_dtype(df['amount']):
            return False, "Amount column must be numeric"
        
        if not pd.api.types.is_numeric_dtype(df['is_fraudulent']):
            return False, "'is_fraudulent' column must be binary (0/1)"
        
        return True, "Valid fraud detection dataset"

    def check_inappropriate_content(self, text: str) -> tuple[bool, str]:
        inappropriate_terms = [
            'hate', 'stupid', 'shit', 'dumb', 'fool', 'idiot', 
            'violence', 'discriminate', 'profanity', 'racism', 'abuse'
        ]
        text_lower = text.lower()
        for term in inappropriate_terms:
            if term in text_lower:
                return False, "Inappropriate content detected. Please rephrase your question."
        return True, ""
    
    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input"""
        forbidden_chars = ['--', ';', '/*', '*/', 'exec', 'eval', 'SELECT', 'DELETE', 'DROP']
        sanitized_input = user_input
        for char in forbidden_chars:
            sanitized_input = sanitized_input.replace(char, '')
        return sanitized_input

    def enforce_guardrails(self, query: str) -> tuple[bool, str]:
        if len(query) > 500:
            return False, "Query is too long. Please limit your input to 500 characters."
        if not query.strip():
            return False, "Empty input detected. Please provide a valid question."
        return True, ""
    
    def is_greeting(self, text: str) -> bool:
        """Check if the input is a greeting"""
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        return text.lower().strip().replace('!', '') in greetings
    
    def generate_fraud_context(self, df: pd.DataFrame) -> str:
        """Generate context about the fraud detection dataset"""
        context = "Fraud Detection Dataset Analysis:\n\n"
        
        total_transactions = len(df)
        fraudulent_transactions = df['is_fraudulent'].sum()
        fraud_percentage = (fraudulent_transactions / total_transactions) * 100
        
        context += f"Total Transactions: {total_transactions}\n"
        context += f"Fraudulent Transactions: {fraudulent_transactions}\n"
        context += f"Fraud Percentage: {fraud_percentage:.2f}%\n\n"
        return context

    def create_fraud_detection_prompt(self, df: pd.DataFrame, user_question: str) -> str:
        """
        Creates a highly specialized prompt for precise fraud detection analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe containing fraud-related data
            user_question (str): User's query about fraud detection
                
        Returns:
            str: Optimized prompt for accurate fraud analysis
        """
        fraud_context = self.generate_fraud_context(df)
        
        system_prompt = """You are an Expert Fraud Detection Analyst with advanced specialization in pattern recognition and risk assessment.
    Your responses must be based exclusively on empirical data analysis and statistical evidence, achieving maximum accuracy."""

        analysis_requirements = """COMPREHENSIVE ANALYSIS REQUIREMENTS:

    1. Data Analysis Protocol:
    A. Core Data Requirements:
        - Use ONLY provided dataset
        - NO external information
        - NO assumptions or hypotheticals
        - Exact numerical analysis
    
    B. Statistical Validation:
        - Confidence intervals
        - Sample size verification
        - Statistical significance testing
        - Error margin calculation
    
    C. Pattern Analysis:
        - Multi-factor correlation
        - Temporal pattern matching
        - Geographic pattern analysis
        - Behavioral pattern verification

    2. Fraud Determination Criteria:
    A. Primary Indicators:
        - Transaction characteristics
        - Location-based patterns
        - Temporal patterns
        - Amount patterns
        
    B. Secondary Indicators:
        - Historical data correlation
        - Behavioral patterns
        - Risk factor accumulation
        - Anomaly detection

    3. Validation Requirements:
    A. Data Validation:
        - Schema verification
        - Data type compatibility
        - Range validation
        - Completeness check
    
    B. Pattern Validation:
        - Cross-reference verification
        - Pattern consistency check
        - Historical correlation
        - Anomaly verification

    OUTPUT SPECIFICATIONS:
    FRAUD DETERMINATION
    Status: [FRAUD/NOT FRAUD]
    Confidence Level: [X%]
    Risk Level: [HIGH/MEDIUM/LOW]
    Fraud Probability: [X%]

    FRAUD REASONING
    • [Specific reasons with exact data points]
    • [Statistical evidence for each reason]
    • [Pattern matches identified]

    RELEVANT STATISTICS
    • [Location-based fraud metrics]
    • [Transaction type analysis]
    • [Payment method statistics]
    • [Merchant category data]

    RECOMMENDATION
    • [Single, data-backed recommendation]

    ERROR HANDLING:
    {error_handling_rules}"""

        validation_protocol = """RESPONSE VALIDATION PROTOCOL:
    1. Accuracy Verification:
    - Cross-reference all data points
    - Verify statistical calculations
    - Validate pattern matches
    - Confirm confidence levels

    2. Quality Control:
    - Check numerical precision
    - Verify statistical significance
    - Validate confidence intervals
    - Confirm data relationships

    3. Response Requirements:
    - Direct and precise answers
    - Data-backed conclusions
    - Specific evidence points
    - Clear reasoning chain"""

        prompt = f"""{system_prompt}

    {analysis_requirements}

    

    Dataset Context:
    {fraud_context}

    Analyze this transaction:
    {user_question}

    Note: Ensure all responses follow the exact output format with maximum precision and accuracy."""

        return prompt.strip()

        
                
    def interpret_with_ollama(self, prompt: str,) -> str:
        """Interpret with Ollama model"""
        ollama_host = os.getenv("OLLAMA_BASE_URL", "https://ollama.devai.tantor.io")
        url = f"{ollama_host}/api/chat"
        payload = {
            "model": os.getenv("OLLAMA_TEXT_MODEL", "llama3.2:latest"),
            "stream": False,
        "messages": [{"role": "user", "content": prompt}]
        }
        try:
            response = requests.post(url, json=payload, verify=False)
            if response.status_code == 200:
                response_dict = response.json()
                return response_dict.get("message", {}).get("content", "")
            else:
                return f"Error: {response.status_code}, {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

   
    def generate_visualizations(self, df: pd.DataFrame, x_axis: str, y_axis: str = None) -> Dict[str, str]:

        """Generates predefined visualizations for fraud detection."""
        visuals = {}

        # Transaction Amount Distribution by Fraud Status
        fig_amount = px.box(
            df,
            x='is_fraudulent',
            y='amount',
            title='Transaction Amount Distribution by Fraud Status',
            labels={'is_fraudulent': 'Is Fraudulent', 'amount': 'Transaction Amount'}
        )
        buffer_amount = io.BytesIO()
        fig_amount.write_image(buffer_amount, format="png")
        buffer_amount.seek(0)
        visuals['amount_distribution'] = base64.b64encode(buffer_amount.read()).decode("utf-8")

        # Top Merchants by Fraud Count
        merchant_fraud = df[df['is_fraudulent'] == 1]['merchant'].value_counts().head(10)
        fig_merchant = px.bar(
            x=merchant_fraud.index,
            y=merchant_fraud.values,
            title='Top 10 Merchants by Fraud Count',
            labels={'x': 'Merchant', 'y': 'Number of Fraudulent Transactions'}
        )
        buffer_merchant = io.BytesIO()
        fig_merchant.write_image(buffer_merchant, format="png")
        buffer_merchant.seek(0)
        visuals['top_merchants'] = base64.b64encode(buffer_merchant.read()).decode("utf-8")

        return visuals

    def generate_profile_report(self, df: pd.DataFrame) -> ProfileReport:
        """Generate YData Profiling Report"""
        profile = ProfileReport(df, title="Fraud Detection Dataset Profiling Report", minimal=True)
        html_report = profile.to_html()


        navbar_pattern = re.compile(r'<nav class="navbar navbar-default navbar-fixed-top">.*?</nav>', re.DOTALL)
        html_report = re.sub(navbar_pattern, '', html_report)

        brought_by_pattern = re.compile(r'<p class="text-muted text-right">Brought to you by <a href="https://ydata.ai/.*?</p>', re.DOTALL)
        html_report = re.sub(brought_by_pattern, '', html_report)

        footer_pattern = re.compile(r'<footer>.*?</footer>', re.DOTALL)
        html_report = re.sub(footer_pattern, '', html_report)

        css_rule_pattern = re.compile(r'body\s*{\s*padding-top:\s*80px;\s*}', re.DOTALL)
        html_report = re.sub(css_rule_pattern, '', html_report)

        software_version_pattern = re.compile(r'<th>Software version</th>\s*<td.*?>.*?</td>', re.DOTALL)
        html_report = re.sub(software_version_pattern, '', html_report)

        download_config_pattern = re.compile(r'<tr>\s*<th>Download configuration</th>\s*<td[^>]*>.*?</td>\s*</tr>', re.DOTALL)
        html_report = re.sub(download_config_pattern, '', html_report)
        
        return html_report
    
    
    def process_file(self, filename: str, file_content: bytes) -> Tuple[bool, str]:
        """Process uploaded file"""
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            is_valid, message = self.validate_fraud_dataset(df)
            
            if not is_valid:
                return False, message
            
            self.uploaded_files_data[filename] = df
            self.current_dataset = filename
            if filename not in self.chat_history:
                self.chat_history[filename] = []
            
            return True, "File processed successfully"
            
        except Exception as e:
            return False, f"Error reading file {filename}: {str(e)}"

    def process_question(self, question: str) -> Tuple[bool, str]:
        """Process user question with enhanced greeting handling"""
        is_appropriate, message = self.check_inappropriate_content(question)
        if not is_appropriate:
            return False, message

        is_valid, message = self.enforce_guardrails(question)
        if not is_valid:
            return False, message

        # Handle greetings - ONLY respond with greeting if user greets first
        if self.is_greeting(question):
            greeting_response = "Hello! How may I assist you?"
            if self.current_dataset:
                self.chat_history[self.current_dataset].append({
                    "role": "user",
                    "content": question
                })
                self.chat_history[self.current_dataset].append({
                    "role": "assistant",
                    "content": greeting_response
                })
            return True, greeting_response

        # For non-greeting questions, proceed with analysis
        if not self.current_dataset:
            return False, "No dataset loaded"

        sanitized_question = self.sanitize_input(question)
        df = self.uploaded_files_data.get(self.current_dataset)
        
        model_prompt = self.create_fraud_detection_prompt(df, sanitized_question)  # Fixed: Removed self.current_dataset argument
        model_output = self.interpret_with_ollama(model_prompt)
        
        self.chat_history[self.current_dataset].append({
            "role": "user",
            "content": sanitized_question
        })
        self.chat_history[self.current_dataset].append({
            "role": "assistant",
            "content": model_output
        })
        
        return True, model_output