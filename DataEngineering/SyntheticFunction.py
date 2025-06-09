import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import requests
import os
from typing import Dict, Any, List
import warnings
import urllib3
from tenacity import retry, stop_after_attempt, wait_exponential
import math
from tqdm import tqdm

# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.devai.tantor.io")
MAX_RETRIES = 3
INITIAL_WAIT = 1  # Initial wait time in seconds
MAX_WAIT = 10  # Maximum wait time in seconds

class SyntheticDataOllama:
    def __init__(self):
        self.api_url = f"{OLLAMA_URL}/api/chat"

    def generate_prompt(self, data: pd.DataFrame) -> str:
        """Generate a detailed prompt for Ollama-based synthetic data generation"""
        data_sample = data.head(5).to_string()
        columns_info = self.get_columns_info(data)

        prompt = f"""As a data synthesis expert, analyze this dataset and generate synthetic data following these rules:

Dataset Sample:
{data_sample}

Column Statistics:
{columns_info}

Task:
1. Generate synthetic data that precisely matches these statistical properties
2. Maintain the following for each column:
   - Data type and range constraints
   - Statistical distribution (mean, std, min, max)
   - Inter-column correlations
   - Any sequential or time-based patterns
3. Ensure generated values follow the same patterns as the original data

Please provide synthetic data values for each column maintaining these exact statistical properties.
Format your response as JSON with column names as keys and generated values as arrays."""

        return prompt

    def get_columns_info(self, data: pd.DataFrame) -> str:
        """Get detailed information about each column with error handling"""
        info_parts = []
        for column in data.columns:
            try:
                stats = {
                    'dtype': str(data[column].dtype),
                    'unique_values': data[column].nunique(),
                    'missing_values': data[column].isnull().sum(),
                }

                # Safely add numeric statistics
                if pd.api.types.is_numeric_dtype(data[column]):
                    non_null_data = data[column].dropna()
                    if len(non_null_data) > 0:
                        stats.update({
                            'min': non_null_data.min(),
                            'max': non_null_data.max(),
                            'mean': non_null_data.mean(),
                            'std': non_null_data.std()
                        })

                info_parts.append(f"Column '{column}':\n" +
                                '\n'.join(f"- {k}: {v}" for k, v in stats.items() if v is not None))
            except Exception as e:
                info_parts.append(f"Column '{column}':\n- Error analyzing column: {str(e)}")

        return '\n\n'.join(info_parts)

    @retry(stop=stop_after_attempt(MAX_RETRIES),
           wait=wait_exponential(multiplier=INITIAL_WAIT, max=MAX_WAIT))
    def query_ollama(self, prompt: str, model: str = "llama3.2:latest") -> str:
        """Query Ollama with improved error handling and response parsing"""
        try:
            payload = {
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": "You are a data synthesis expert who generates statistically accurate synthetic data."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "top_p": 0.95
            }

            response = requests.post(
                self.api_url,
                json=payload,
                verify=False,
                timeout=60  # Increased timeout for complex generations
            )
            response.raise_for_status()

            content = response.json().get("message", {}).get("content", "")
            if not content:
                raise ValueError("Empty response from Ollama")

            return content

        except requests.exceptions.ReadTimeout:
            print("Warning: Ollama request timed out. Retrying...")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Warning: Ollama request failed: {str(e)}")
            raise

    def generate_synthetic_data(self, df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data using Ollama without batch processing"""
        try:
            # Generate a single prompt for the entire dataset
            prompt = f"""As a data synthesis expert, generate {num_samples} rows of synthetic data based on the following dataset:

Dataset Sample:
{df.head(5).to_string()}

Column Statistics:
{self.get_columns_info(df)}

Task:
1. Generate synthetic data that matches the statistical properties of the original dataset.
2. Maintain the following for each column:
   - Data type and range constraints
   - Statistical distribution (mean, std, min, max)
   - Inter-column correlations
   - Any sequential or time-based patterns
3. Ensure generated values follow the same patterns as the original data.

Provide the synthetic data as JSON with column names as keys and generated values as arrays."""

            # Query Ollama for synthetic data
            ollama_response = self.query_ollama(prompt)
            print("Received Ollama response for synthetic data generation")

            # Validate and process the response
            if not ollama_response.strip().startswith("{"):
                raise ValueError("Ollama response is not valid JSON. Falling back to statistical generation.")

            synthetic_data = self._process_ollama_response(ollama_response, df.columns)
            df_synthetic = pd.DataFrame(synthetic_data)

            # Enforce original data types
            self._enforce_dtypes(df_synthetic, df)

            return df_synthetic

        except Exception as e:
            print(f"Error generating synthetic data using Ollama: {str(e)}")
            print("Falling back to statistical generation method...")
            return self._generate_statistical_fallback(df, num_samples)

    def _enforce_dtypes(self, df_synthetic: pd.DataFrame, df_original: pd.DataFrame) -> None:
        """Enforce original datatypes with error handling"""
        for col in df_original.columns:
            try:
                df_synthetic[col] = df_synthetic[col].astype(df_original[col].dtype)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to type {df_original[col].dtype}: {str(e)}")

    def analyze_column(self, data: pd.Series) -> dict:
        """Analyze a single column with error handling"""
        try:
            stats = {
                'dtype': str(data.dtype),
                'unique_values': data.nunique(),
                'existing_values': list(data.dropna().unique()),
                'is_sequential': False,
                'sequence_step': None
            }

            non_null_data = data.dropna()
            if len(non_null_data) > 0:
                if pd.api.types.is_numeric_dtype(data):
                    stats.update({
                        'min': non_null_data.min(),
                        'max': non_null_data.max(),
                        'mean': non_null_data.mean(),
                        'std': non_null_data.std()
                    })

                    # Check for sequential pattern
                    sorted_values = sorted(non_null_data.unique())
                    if len(sorted_values) > 1:
                        differences = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values)-1)]
                        if len(set(differences)) == 1:
                            stats['is_sequential'] = True
                            stats['sequence_step'] = differences[0]

            return stats
        except Exception as e:
            print(f"Warning: Error analyzing column: {str(e)}")
            return {'dtype': str(data.dtype), 'error': str(e)}

    def generate_value(self, column_stats: dict, index: int, total_samples: int) -> Any:
        """Generate a value with improved distribution matching"""
        if 'error' in column_stats:
            return None

        try:
            dtype = column_stats['dtype']
            existing_values = column_stats['existing_values']

            # For numeric data, use a combination of sampling and distribution
            if pd.api.types.is_numeric_dtype(dtype):
                # Ensure the maximum value is included in the generated data
                if index == total_samples - 1:  # For the last sample, return the max value
                    return column_stats['max']

                # 50% chance to sample from existing values
                if random.random() < 0.5:
                    return random.choice(existing_values)

                # Otherwise use distribution-based generation
                if 'std' in column_stats and 'mean' in column_stats:
                    # Generate value within 2 standard deviations
                    value = random.gauss(column_stats['mean'], column_stats['std'])
                    # Clip to min/max range of original data
                    return max(min(value, column_stats['max']), column_stats['min'])
                else:
                    return round(random.uniform(column_stats['min'], column_stats['max']), 2)

            if dtype == 'object':
                return random.choice(existing_values)

            if dtype.startswith('datetime'):
                # Sample from existing dates with some variation
                base_date = random.choice(existing_values)
                variation_days = random.randint(-30, 30)  # Adjust range as needed
                return base_date + timedelta(days=variation_days)

            return None
        except Exception as e:
            print(f"Warning: Error generating value: {str(e)}")
            return None

    def _generate_statistical_fallback(self, df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        """Generate synthetic data using statistical methods as fallback"""
        try:
            synthetic_data = []
            column_stats = {col: self.analyze_column(df[col]) for col in df.columns}

            # Use tqdm for progress tracking
            for _ in tqdm(range(num_samples), desc="Generating synthetic data"):
                row = {col: self.generate_value(column_stats[col], _, num_samples)
                      for col in df.columns}
                synthetic_data.append(row)

            df_synthetic = pd.DataFrame(synthetic_data)
            self._enforce_dtypes(df_synthetic, df)

            return df_synthetic

        except Exception as e:
            print(f"Error in statistical fallback generation: {str(e)}")
            # Return empty DataFrame with same columns as original
            return pd.DataFrame(columns=df.columns)

    def _process_ollama_response(self, response: str, columns: List[str]) -> List[Dict]:
        """Process Ollama response and convert to list of dictionaries"""
        try:
            import json
            import re

            # Find JSON-like content in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                # Convert the JSON response to list of row dictionaries
                rows = []
                if isinstance(data, dict):
                    # Assume response has column names as keys and value arrays
                    num_rows = len(next(iter(data.values())))
                    for i in range(num_rows):
                        row = {col: data.get(col, [])[i] if col in data else None for col in columns}
                        rows.append(row)
                return rows

            else:
                raise ValueError("No valid JSON content found in response")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Ollama response: {str(e)}")
            print("Response content (truncated):", response[:200])  # Print first 200 chars for debugging
            raise
        except Exception as e:
            print(f"Error processing Ollama response: {str(e)}")
            print("Response content (truncated):", response[:200])  # Print first 200 chars for debugging
            raise




