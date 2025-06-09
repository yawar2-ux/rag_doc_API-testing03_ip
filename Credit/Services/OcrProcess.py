from groq import Groq
import json
import re
from .FieldConfig import keys

def extract_structured_data(ocr_text, client):
    """
    Process OCR text to extract structured data for predefined keys.
    
    Args:
        ocr_text (str): The OCR text to process
        client: Groq client instance
        
    Returns:
        dict: Dictionary with extracted key-value pairs
    """
    # Create the prompt for the LLM
    prompt = f"""
Extract the following information from the OCR text. For each key, provide the corresponding value.
If the information is not available or cannot be found, write "NA".

IMPORTANT: For NSC/KVP/Bank/Post Office deposit details, if the table exists but contains only "None" values, 
this should still be treated as valid data (indicating no deposits) rather than missing data.

Return the result as a JSON object with keys and values.
Use the exact key names provided below without changing the format.

OCR Text:
{ocr_text}

Keys to extract:
{', '.join(keys)}
"""
    
    try:
        # Call the LLM using Groq
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant that extracts structured information from OCR text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000,
            top_p=1,
            stream=False
        )
        
        # Get the response content
        content = response.choices[0].message.content
        
        # Try to extract JSON from the response
        try:
            # Look for JSON-like structure in the response
            json_match = re.search(r'(\{[\s\S]*\})', content)
            if json_match:
                extracted_data = json.loads(json_match.group(1))
            else:
                # Fallback to manual extraction
                extracted_data = {}
                for key in keys:
                    pattern = rf'"{key}":\s*"([^"]*)"'
                    match = re.search(pattern, content)
                    extracted_data[key] = match.group(1) if match else "NA"
        except Exception:
            # If JSON parsing fails completely
            extracted_data = {}
        
        # Ensure all keys are present in the result
        result = {key: extracted_data.get(key, "NA") for key in keys}
        return result
        
    except Exception as e:
        # In case of any error, return dictionary with NA values
        print(f"Error extracting data: {str(e)}")
        return {key: "NA" for key in keys}
