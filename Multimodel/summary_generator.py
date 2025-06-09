from pathlib import Path
from typing import List
import pandas as pd
from PIL import Image
import io
import logging
import base64
import json
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama Configuration from env
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
MODEL = os.getenv('OLLAMA_MODEL')
TEXTMODEL = os.getenv('OLLAMA_TEXT_MODEL')

def compress_image(image_path: Path, max_size: int = 800) -> str:
    """Compress and resize image before sending to API"""
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            width, height = img.size
            ratio = max_size / max(width, height)
            
            if ratio < 1:
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            img_byte_arr.seek(0)
            
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
    except Exception as e:
        logger.error(f"Error compressing image {image_path}: {str(e)}")
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def send_ollama_request(prompt: str, image_data: str = None) -> str:
    """Send request to Ollama API"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 500,
        }
    }
    
    # Add image data if provided
    if image_data:
        payload["images"] = [image_data]
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            verify=False
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")
        
        result = response.json()
        return result.get('response', '')
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in Ollama API: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in Ollama request: {e}")
        return "Error generating description"

@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def get_image_description(image_path: Path) -> str:
    """Get LLM description for an image using Ollama"""
    try:
        img_data = compress_image(image_path)
        
        prompt = """As a precise and detailed image analyzer focused on identification, analyze this image and provide a detailed description focusing on:
        1. Physical appearance
        2. Clothing and accessories
        3. Any visible identifiers (tattoos, scars, marks)
        4. Any texts or symbols
        
        Format the description in clear and objective language."""
        
        description = send_ollama_request(prompt, img_data)
        
        if not description:
            return "Error: Failed to generate image description"
            
        logger.info(f"Successfully generated description for image: {image_path}")
        return description
        
    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}")
        return f"Error: Unexpected error during image analysis - {str(e)}"

@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)

def send_ollama_request_table(prompt: str, image_data: str = None) -> str:
    """Send request to Ollama API for table description"""
    payload = {
        "model": TEXTMODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 500,
        }
    }
    
    # Add image data if provided
    # if image_data:
    #     payload["images"] = [image_data]
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            verify=False
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")
        
        result = response.json()
        return result.get('response', '')
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in Ollama API: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in Ollama request: {e}")
        return "Error generating description"

async def get_table_description(table_path: Path) -> str:
    """Get LLM description for a table using Ollama"""
    try:
        # Read CSV file
        df = pd.read_csv(table_path)
        if df.empty:
            return "Error: Table is empty"

        # Get basic table stats
        num_rows = len(df)
        columns = df.columns.tolist()
        table_content = df.to_string()
        
        prompt = f"""Analyze this record table data and provide a detailed summary focusing on:
        1. Key information
        2. Patterns and relationships
        3. Notable insights
        4. Temporal patterns or trends
        
        Table Information:
        - Number of records: {num_rows}
        - Fields available: {', '.join(columns)}
        
        Table content:
        {table_content}
        
        Provide a comprehensive summary in clear and professional language."""
        
        description = send_ollama_request_table(prompt)
        
        if not description:
            return "Error: Failed to generate table description"
            
        logger.info(f"Successfully generated description for table: {table_path}")
        return description
        
    except pd.errors.EmptyDataError:
        return "Error: Table is empty or contains no valid data"
    except Exception as e:
        logger.error(f"Error processing table {table_path}: {str(e)}")
        return f"Error: Unexpected error during table analysis - {str(e)}"
