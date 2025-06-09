from typing import List, Dict, Any
import pandas as pd
import requests
import json
import time
import os
from ydata_profiling import ProfileReport
from dotenv import load_dotenv


load_dotenv()
# Ollama configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
MODEL = os.getenv('OLLAMA_TEXT_MODEL')

def generate_column_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Generate comprehensive statistics for a column to determine its type.
    """
    stats = {
        "name": column,
        "dtype": str(df[column].dtype),
        "unique_values": df[column].nunique(),
        "missing_values": df[column].isnull().sum(),
    }

    if pd.api.types.is_numeric_dtype(df[column]):
        stats.update({
            "min": float(df[column].min()) if not df[column].empty else None,
            "max": float(df[column].max()) if not df[column].empty else None,
            "mean": float(df[column].mean()) if not df[column].empty else None,
            "type": "numeric"
        })
    elif pd.api.types.is_datetime64_any_dtype(df[column]):
        stats.update({
            "min_date": str(df[column].min()) if not df[column].empty else None,
            "max_date": str(df[column].max()) if not df[column].empty else None,
            "type": "datetime"
        })
    else:
        value_counts = df[column].value_counts().head(5).to_dict()
        stats.update({
            "common_values": value_counts,
            "type": "categorical"
        })

    return stats

def generate_smart_prompt(column_stats: Dict[str, Any]) -> str:
    """
    Generate an intelligent prompt based on column statistics and data type.
    """
    base_prompt = f"""You are a data cleaning expert. Analyze the following column information and suggest appropriate values for missing data:

Column: {column_stats['name']}
Data Type: {column_stats['dtype']}

Additional Context:"""

    if column_stats['type'] == "numeric":
        base_prompt += f"""
- This is a numeric column
- Value Range: {column_stats['min']} to {column_stats['max']}
- Mean: {column_stats['mean']}

Generate numeric values that:
1. Fall within the existing range
2. Follow the current distribution pattern
3. Are appropriate for the column's context
"""

    elif column_stats['type'] == "datetime":
        base_prompt += f"""
- This is a datetime column
- Date Range: {column_stats['min_date']} to {column_stats['max_date']}

Generate datetime values that:
1. Fall within the existing date range
2. Follow realistic temporal patterns
3. Use the format: YYYY-MM-DD HH:MM:SS
"""

    else:  # categorical
        base_prompt += f"""
- This is a categorical column
- Most Common Values: {column_stats['common_values']}

Generate categorical values that:
1. Use existing categories when appropriate
2. Maintain the distribution pattern
3. Are contextually relevant
"""

    base_prompt += """
Return ONLY a JSON array with the suggested values in this exact format:
{"values": [value1, value2, ...]}

Important:
- Match the existing data type exactly
- Ensure values are consistent with the column's context
- Do not include explanations, only the JSON response
"""

    return base_prompt

def query_ollama(prompt: str) -> List[Any]:
    """
    Query Ollama for missing value suggestions.
    """
    try:
        payload = {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a data cleaning expert."},
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result.get("message", {}).get("content", "")
        if content:
            # Remove triple backticks and whitespace
            cleaned_content = content.strip("```").strip()
            return json.loads(cleaned_content).get("values", [])
        print("Ollama returned an empty content.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"HTTP error querying Ollama: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error querying Ollama: {e}")
        print(f"Response text: {response.text}")
        return []
    except Exception as e:
        print(f"Unexpected error querying Ollama: {e}")
        return []

def process_column(df: pd.DataFrame, column: str, column_stats: Dict[str, Any]) -> None:
    """
    Process missing values for a column with consistent filling logic.
    """
    if column_stats['type'] == 'numeric':
        # Fill missing values with the mean
        df[column].fillna(df[column].mean(), inplace=True)
    elif column_stats['type'] == 'categorical':
        # Fill missing values with the mode
        df[column].fillna(df[column].mode()[0], inplace=True)
    elif column_stats['type'] == 'datetime':
        # Fill missing values with the median
        df[column].fillna(df[column].median(), inplace=True)
    else:
        # Default fallback for unknown types
        df[column].fillna("Unknown", inplace=True)

    print(f"Filled missing values in column: {column}")

def generate_profiling_report(df: pd.DataFrame) -> str:
    """
    Generate and clean up the profiling report HTML.
    """
    import re

    # Generate profiling report
    profile = ProfileReport(df, title="Profiling Report", minimal=False)
    html_report = profile.to_html()

    # Clean up HTML report
    patterns_to_remove = [
        (r'<nav class="navbar navbar-default navbar-fixed-top">.*?</nav>', ''),
        (r'<p class="text-muted text-right">Brought to you by <a href="https://ydata.ai/.*?</p>', ''),
        (r'<footer>.*?</footer>', ''),
        (r'body\s*{\s*padding-top:\s*80px;\s*}', ''),
        (r'<th>Software version</th>\s*<td.*?>.*?</td>', ''),
        (r'<tr>\s*<th>Download configuration</th>\s*<td[^>]*>.*?</td>\s*</tr>', '')
    ]

    for pattern, replacement in patterns_to_remove:
        html_report = re.sub(pattern, replacement, html_report, flags=re.DOTALL)

    return html_report