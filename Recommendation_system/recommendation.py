import pandas as pd
import numpy as np
import requests
import os
import json
import time
from typing import List, Dict, Tuple
import plotly.express as px
import io
from ydata_profiling import ProfileReport
import base64
import re

class ChatAssistant:
    def __init__(self):
        self.chat_history = {}  # Dictionary to store chat history per dataset
        self.uploaded_files_data = {}
        self.profile_reports = {}
        self.show_profile_reports = {}
        self.current_dataset = None

    def clear_state(self):
        """Clear all stored state"""
        self.chat_history = {}
        self.uploaded_files_data = {}
        self.profile_reports = {}
        self.show_profile_reports = {}
        self.current_dataset = None
        return True

    def validate_file_size(self, file_size: int) -> bool:
        """Validate file size without Streamlit dependency"""
        MAX_FILE_SIZE = 50 * 1024 * 1024
        return file_size <= MAX_FILE_SIZE

    def validate_data_format(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Validate DataFrame format"""
        if len(df.columns) < 2:
            return False, "Dataset must have at least 2 columns."
        if len(df) < 1:
            return False, "Dataset must contain at least 1 row."
        if not any(df.select_dtypes(include=[np.number]).columns):
            return False, "Dataset must contain at least one numeric column."
        return True, "Valid dataset."

    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """Get dataset information without UI dependencies"""
        return {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist()
        }

    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input"""
        forbidden_chars = ['--', ';', '/*', '*/', 'exec', 'eval', 'SELECT', 'DELETE', 'DROP']
        sanitized_input = user_input
        for char in forbidden_chars:
            sanitized_input = sanitized_input.replace(char, '')
        return sanitized_input

    def check_content_moderation(self, text: str) -> tuple[bool, str]:
        """Check content for inappropriate terms"""
        inappropriate_terms = [
            'hate','stupid','shit','dumb','fool','idiot', 'violence', 
            'discriminate', 'profanity', 'racism', 'abuse'
        ]
        text_lower = text.lower()
        for term in inappropriate_terms:
            if term in text_lower:
                return False, "Inappropriate input detected. Please rephrase your question."
        return True, ""

    def enforce_guardrails(self, query: str) -> tuple[bool, str]:
        """Enforce input guardrails"""
        if len(query) > 500:
            return False, "Query is too long. Please limit your input to 500 characters."
        if not query.strip():
            return False, "Empty input detected. Please provide a valid question."
        return True, ""

    def is_greeting(self, text: str) -> bool:
        """Check if the input is a greeting"""
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        return text.lower().strip().replace('!', '') in greetings

    def generate_data_context(self, filename: str, df: pd.DataFrame) -> str:
        """Generate data context for prompts"""
        context = f"Current Dataset Analysis: {filename}\n\n"
        context += f"Total Records: {len(df)}\n"
        context += f"Columns: {', '.join(df.columns)}\n\n"
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            context += "Numeric Columns Summary:\n"
            for col in numeric_cols:
                context += f"{col}:\n"
                context += f"  Mean: {df[col].mean():.2f}\n"
                context += f"  Median: {df[col].median():.2f}\n"
                context += f"  Min: {df[col].min():.2f}\n"
                context += f"  Max: {df[col].max():.2f}\n"
        
        return context

    def create_dataset_specific_prompt(self, filename: str, df: pd.DataFrame, user_question: str) -> str:
        """Create prompt for model with specific rules"""
        data_context = self.generate_data_context(filename, df)
        prompt = f"""You should read the whole dataset from 1st row and column to n number of rows and columns and then you should response. You are a data analyst focusing solely on the following dataset. Provide answers based strictly on this data only.

        {data_context}


    User Question: {user_question}

    Important: 
    You are an advanced, precise data analysis assistant. Follow these rules exactly:

    DATA HANDLING:
    - Use only the provided dataset values for analysis and responses.
    - Return exact numbers as they appear in the dataset, preserving original precision.
    - Include complete row/column context when presenting data.
    - No external data or assumptions are allowed.

    RECOMMENDATION RULES:
    - Use only the provided dataset values for recommendation and responses.
    - Generate recommendations strictly from patterns within the dataset.
    - Base similarity scores solely on exact data matches.
    - Provide top N recommendations, including confidence scores for each.
    - Include supporting data points for every recommendation.
    - Sort recommendations by relevance score in descending order.
    - Use collaborative filtering only within the dataset's scope.
    - Apply content-based filtering based on dataset features.
    - Calculate item-item similarity using attributes from the dataset.

    LANGUAGE RULES:
    - Communicate strictly in English.
    - NEVER start your response with a greeting unless the user's input is specifically a greeting.
    - Start your response directly with the analysis or answer to the user's question.
    - For non-English queries, respond: "Please use English language."

    VALIDATION RULES:
    - Validate the query against the dataset's schema.
    - For invalid requests, respond with: "Not found".
    - Match query data types exactly.
    - Avoid interpolation, approximation, or estimates.
    - Display the user's question along with the answer.

    RESPONSE FORMAT:
    - Maintain consistent decimal places in numerical data.
    - Include units of measurement if present in the dataset.
    - Use comma-separated lists for presenting data.
    - Default sorting for numerical outputs should be ascending.
    - Start analysis responses directly without any greeting or introduction.
    - while respondng do not give question Start analysis responses directly
    """
        return prompt

    def interpret_with_ollama(self, prompt: str, model: str = "llama3.2:latest") -> str:
        """Interpret with Ollama model"""
        ollama_host = os.getenv("OLLAMA_BASE_URL")
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

    def generate_profile_report(self, df: pd.DataFrame) -> str:
        """Generate profile report"""
        profile = ProfileReport(df, title="Dataset Profiling Report")
        html_report = profile.to_html()

        # Remove navbar
        navbar_pattern = re.compile(r'<nav class="navbar navbar-default navbar-fixed-top">.*?</nav>', re.DOTALL)
        html_report = re.sub(navbar_pattern, '', html_report)

        # Remove Brought by section
        brought_by_pattern = re.compile(r'<p class="text-body-secondary text-end">Brought to you by <a href="https://ydata\.ai/[^<]*</a></p>', re.DOTALL)
        html_report = re.sub(brought_by_pattern, '', html_report)

        # Remove footer
        footer_pattern = re.compile(r'<footer>.*?</footer>', re.DOTALL)
        html_report = re.sub(footer_pattern, '', html_report)

        # Remove CSS rule
        css_rule_pattern = re.compile(r'body\s*{\s*padding-top:\s*80px;\s*}', re.DOTALL)
        html_report = re.sub(css_rule_pattern, '', html_report)

        # Remove software version
        software_version_pattern = re.compile(r'<th>Software version</th>\s*<td.*?>.*?</td>', re.DOTALL)
        html_report = re.sub(software_version_pattern, '', html_report)

        # Remove download configuration
        download_config_pattern = re.compile(r'<tr>\s*<th>Download configuration</th>\s*<td[^>]*>.*?</td>\s*</tr>', re.DOTALL)
        html_report = re.sub(download_config_pattern, '', html_report)
        
        return html_report

    def generate_visualizations(self, df: pd.DataFrame, x_axis: str, y_axis: str = None) -> Dict[str, str]:
        """Generate visualizations and return base64 encoded images"""
        plots = {}
        
        # Create distribution plot
        fig_dist = px.histogram(df, x=x_axis, title=f"Distribution of {x_axis}")
        dist_buffer = io.BytesIO()
        fig_dist.write_image(dist_buffer, format='png')
        dist_buffer.seek(0)
        
        base64_1 = base64.b64encode(dist_buffer.getvalue()).decode()
        temp = f"data:image/png;base64,{base64_1}"
        plots['distribution'] = temp

        # Create scatter plot if y_axis is provided
        if y_axis:
            fig_scatter = px.scatter(df, x=x_axis, y=y_axis,
                                   title=f"{x_axis} vs {y_axis}")
            scatter_buffer = io.BytesIO()
            fig_scatter.write_image(scatter_buffer, format='png')
            scatter_buffer.seek(0)
            base64_2 = base64.b64encode(scatter_buffer.getvalue()).decode()
            temp = f"data:image/png;base64,{base64_2}"
            plots['scatter'] = temp
        
        return plots

    def process_file(self, filename: str, file_content: bytes) -> Tuple[bool, str]:
        """Process uploaded file"""
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            is_valid, message = self.validate_data_format(df)
            
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
        is_appropriate, message = self.check_content_moderation(question)
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
        
        model_prompt = self.create_dataset_specific_prompt(self.current_dataset, df, sanitized_question)
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