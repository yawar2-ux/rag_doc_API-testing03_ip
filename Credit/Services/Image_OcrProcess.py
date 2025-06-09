import os
import base64
import re
import json
import csv
import io
from collections import defaultdict
from typing import List, Tuple
import numpy as np
from groq import Groq

import aiofiles
from starlette.concurrency import run_in_threadpool
from PIL import Image

from .FieldConfig import keys as FIELDS_TO_EXTRACT, STANDARD_CSV_FILENAME, UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, STANDARD_CSV_PATH
from .CleanupUtils import cleanup_temp_files

# Global variables are now imported from FieldConfig

def get_application_number(filename):
    """Extract the application number from the filename."""
    match = re.match(r'([^-]+)', filename)
    if match:
        return match.group(1)
    return "UNKNOWN"

def group_images_by_application(image_files):
    """Group image files by application number."""
    applications = defaultdict(list)
    
    for image_file in image_files:
        app_num = get_application_number(image_file.filename)
        applications[app_num].append(image_file)
    
    return applications

def encode_image_to_base64(image_path):
    """Convert an image file to base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def detect_and_correct_orientation(image_path, client, skip_orientation=False) -> str:
    """
    Simplified function that just returns the original image path.
    Rotation functionality has been removed as it's no longer required.
    
    Args:
        image_path: Path to the image file
        client: Groq client instance
        skip_orientation: Not used anymore, kept for API compatibility
        
    Returns:
        Original image path
    """
    print(f"Skipping orientation detection for {image_path} (functionality removed)")
    return image_path

async def extract_form_fields(image_path, fields, client):
    """Extract specified fields from an image using Llama-4-Scout model."""
    base64_image = await run_in_threadpool(encode_image_to_base64, image_path)
    
    try:
        completion = await run_in_threadpool(
            lambda: client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Extract the following fields from this loan application form image:

{chr(10).join(fields)}

For each field, provide the value if it's present in the image, or "NA" if it's not visible or empty.

IMPORTANT: 
- For NSC/KVP/Bank/Post Office deposit details, if the table exists but contains only "None" values, 
this should still be treated as valid data (indicating no deposits) rather than missing data.
- For Repayment period:
  - Convert abbreviations like "Yrs", "Mths", "Days" to their full forms: "Years", "Months", "Days".
- For Income and monetary values:
  - Convert terms like "Lakhs", "Crores", and "Thousand" to full numeric values.
  - Example: "2.5 Lakhs" → 250000, "1,25,000" → 125000, "2 Crores" → 20000000
  - Remove currency symbols and commas from numbers.
- For dates, try to standardize to DD/MM/YYYY format.
- If any field is empty, missing, crossed out, or cannot be clearly extracted, return "NA".

Return the results as a JSON object with field names as keys and extracted values as values.
Focus only on extracting the requested fields - don't include any explanations or commentary."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0,
                max_tokens=4096,
                top_p=1,
                stream=False,
                stop=None,
            )
        )
        
        response = completion.choices[0].message.content.strip()
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str), None
            else:
                field_values = {}
                for field in fields:
                    if field in response:
                        field_parts = response.split(field + ":", 1)
                        if len(field_parts) > 1:
                            value = field_parts[1].split("\n", 1)[0].strip()
                            field_values[field] = value
                        else:
                            field_values[field] = "NA"
                    else:
                        field_values[field] = "NA"
                return field_values, {"error_type": "json_format", "message": "Failed to extract JSON from response"}
                
        except json.JSONDecodeError:
            print(f"Could not parse JSON from response for {image_path}")
            return {field: "NA" for field in fields}, {"error_type": "json_decode", "message": "Failed to parse JSON response"}
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return {field: "NA" for field in fields}, {"error_type": "api_error", "message": str(e)}

async def process_application(app_num, image_files, fields, client):
    """Process all images for a single application."""
    app_values = {field: "NA" for field in fields}
    errors = []
    
    for i, file in enumerate(image_files):
        filename = f"{app_num}-{i+1}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        try:
            async with aiofiles.open(filepath, 'wb') as out_file:
                # Save the uploaded file
                content = await file.read()
                await out_file.write(content)
            
            # Use the filepath directly instead of calling orientation detection
            image_path = filepath
            
            # Extract form fields from the image
            extracted_values, error = await extract_form_fields(image_path, fields, client)
            
            # Track any errors that occurred
            if error:
                errors.append({
                    "image": filename,
                    "path": filepath,
                    "error_type": error["error_type"],
                    "message": error["message"]
                })
            
            # Update app values with extracted values (keeping any non-NA values)
            for field, value in extracted_values.items():
                if app_values[field] == "NA" and value != "NA":
                    app_values[field] = value
                    
        except Exception as e:
            errors.append({
                "image": filename,
                "path": filepath,
                "error_type": "file_processing",
                "message": str(e)
            })
    
    return app_values, errors

async def generate_csv(applications_data):
    """Generate a CSV file from the extracted application data."""
    csv_fields = ["Application"] + FIELDS_TO_EXTRACT
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=csv_fields)
    writer.writeheader()
    
    for app_num, app_values in applications_data.items():
        row_values = {"Application": app_num}
        row_values.update(app_values)
        writer.writerow(row_values)
    
    async with aiofiles.open(STANDARD_CSV_PATH, 'w', newline='') as f:
        await f.write(output.getvalue())
    
    return STANDARD_CSV_PATH

async def process_all_applications(applications, background_tasks):
    """Process all applications and generate CSV."""
    try:
        api_key = os.environ.get("GROQ_API_KEY_2")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        client = Groq(api_key=api_key)
        
        applications_data = {}
        all_errors = {}
        
        for app_num, image_files in applications.items():
            print(f"Processing Application: {app_num} with {len(image_files)} image(s)")
            
            app_values, errors = await process_application(app_num, image_files, FIELDS_TO_EXTRACT, client)
            applications_data[app_num] = app_values
            
            if errors:
                all_errors[app_num] = errors
        
        # Generate and return the main CSV path
        csv_path = await generate_csv(applications_data)
        
        background_tasks.add_task(cleanup_files)
        
        # Return both the CSV path and error information
        return {
            "csv_path": csv_path,
            "errors": all_errors if all_errors else None
        }
    except Exception as e:
        print(f"Error processing applications: {str(e)}")
        raise Exception(f"Error processing applications: {str(e)}")

async def cleanup_files():
    """
    Legacy wrapper for cleanup_temp_files - forwards to centralized implementation
    """
    # Call only the temp file cleanup, not output cleanup
    cleanup_temp_files()
