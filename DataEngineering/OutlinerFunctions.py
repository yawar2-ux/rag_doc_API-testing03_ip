from typing import List, Dict, Any
import pandas as pd
import numpy as np
import requests
import json
import time
import os

def detect_outliers(df: pd.DataFrame) -> Dict[str, float]:
    """
    Detect outliers in numerical columns using IQR method and return percentage of outliers.
    """
    outlier_percentages = {}

    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        percentage = (len(outliers) / len(df)) * 100
        outlier_percentages[column] = round(percentage, 2)

    return outlier_percentages

def generate_outlier_prompt(df: pd.DataFrame, column: str, outlier_values: List[float]) -> str:
    """
    Generate a prompt for Ollama to decide whether to remove outlier values.
    """
    stats = {
        "mean": df[column].mean(),
        "median": df[column].median(),
        "std": df[column].std(),
        "min": df[column].min(),
        "max": df[column].max(),
        "q1": df[column].quantile(0.25),
        "q3": df[column].quantile(0.75)
    }

    prompt = f"""You are a data cleaning expert. Analyze these outlier values and decide which ones should be removed:

Column: {column}
Column Statistics:
- Mean: {stats['mean']}
- Median: {stats['median']}
- Standard Deviation: {stats['std']}
- Q1: {stats['q1']}
- Q3: {stats['q3']}
- Min (overall): {stats['min']}
- Max (overall): {stats['max']}

Outlier Values to Analyze: {outlier_values}

Please analyze each value and decide if it should be removed based on:
1. How extreme the value is compared to the normal range
2. Whether it could be a legitimate data point
3. The potential impact on the dataset's integrity

Return ONLY a JSON array with boolean values (true for remove, false for keep) in this format:
{{"remove": [true, false, ...]}}

Important:
- Return one boolean for each outlier value
- True means remove the value
- False means keep the value
- Do not include explanations, only the JSON response"""

    return prompt

def process_outliers(df: pd.DataFrame, column: str, model: str, max_retries: int, retry_delay: int) -> pd.DataFrame:
    """
    Process outliers in a given column using Ollama decisions for removal.
    """
    df_copy = df.copy()

    # Calculate bounds
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outlier indices
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_indices = df[outlier_mask].index
    outlier_values = df.loc[outlier_indices, column].tolist()

    if not outlier_values:
        return df_copy

    # Generate and send prompt to Ollama
    prompt = generate_outlier_prompt(df, column, outlier_values)

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(
                f"{os.getenv('OLLAMA_BASE_URL')}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                verify=False,
                timeout=120
            )
            response.raise_for_status()
            content = response.json().get('message', {}).get('content', '')

            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                removal_decisions = json.loads(content[json_start:json_end])['remove']

                # Remove outliers based on Ollama's decisions
                indices_to_remove = [idx for idx, remove in zip(outlier_indices, removal_decisions) if remove]
                if indices_to_remove:
                    df_copy.drop(indices_to_remove, inplace=True)
                    print(f"Removed {len(indices_to_remove)} outliers from {column}")
                break

        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            retries += 1
            print(f"Error processing outliers for column {column} (attempt {retries}/{max_retries}): {str(e)}")
            time.sleep(retry_delay * retries)

    return df_copy