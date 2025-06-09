import os
import base64  # Add this import
import shutil
import tempfile
import csv
import json
from typing import Dict, List, Any
import pandas as pd

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from credit_underwriting.function import process_loan_application
from credit_underwriting.ocr import OCRProcessor

router = APIRouter()

# Initialize OCR processor
ocr_processor = OCRProcessor()

# Define Pydantic model for loan application form data
class LoanApplicationForm(BaseModel):
    # Personal Information
    customer_id: str
    name: str
    email_id: str
    mobile_number: str
    
    # Address Information
    residence_address: str
    city: str
    state: str
    pin_code: str
    
    # Financial Details  
    annual_income: float
    net_monthly_income: float
    monthly_fixed_expenses: float
    debt: float
    credit_score: float
    credit_utilization_ratio: float
    dti: float
    
    # Loan Details
    loan_type: str
    loan_amount: float
    loan_purpose: str
    
    # Employment Details
    employement_type: str
    work_experience: float
    education_level: str
    
    # Additional Information
    marital_status: str
    dependents: int
    insurance_coverage: str
    loan_repayment_history: str
    owns_a_house: str
    
    # Aadhaar Information
    aadhaar_present_address: str
    aadhaar_permanent_address: str
    
    # Optional fields with default values
    saving_account: str = "Yes"
    constitution: str = "Resident Individual"

@router.post("/process-loan-applications")
async def process_loan_applications(file: UploadFile = File(...)):
    """Process uploaded CSV file containing loan applications"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format: {file.filename}. Only CSV files are supported."
        )

    try:
        shaukat_dir = os.path.join(os.getcwd(), "shaukat")
        os.makedirs(shaukat_dir, exist_ok=True)

        # Save the file in the 'shaukat' directory
        file_path = os.path.join(shaukat_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return StreamingResponse(
            process_loan_application(file_path),
            media_type='text/event-stream'
        )

    finally:
        file.file.close()

@router.post("/process-to-csv")
async def process_to_csv(files: List[UploadFile] = File(...)):
    """Process uploaded images and extract loan application data"""
    temp_paths = []
    try:
        # Validate file types
        for file in files:
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file format: {file.filename}. Only image files are supported."
                )

        # Process files
        base64_images = []
        for file in files:
            temp_path = os.path.join(ocr_processor.temp_dir, file.filename)
            temp_paths.append(temp_path)

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            with open(temp_path, "rb") as image_file:
                base64_images.append(base64.b64encode(image_file.read()).decode('utf-8'))

        # Process images and get CSV file
        csv_file, _ = ocr_processor.process_images(base64_images)

        if not os.path.exists(csv_file):
            raise HTTPException(status_code=500, detail="Failed to generate CSV file")

        # Process the CSV file through loan application
        return StreamingResponse(
            process_loan_application(csv_file),
            media_type='text/event-stream'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary files
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temporary file {temp_path}: {str(e)}")

@router.post("/process-form-application")
async def process_form_application(application_data: LoanApplicationForm):
    """
    Process a loan application submitted through a form
    and pass it to the credit underwriting system.
    """
    try:
        # Create a temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
        temp_file_path = temp_file.name
        
        try:
            # Get the field names from the model
            field_names = list(application_data.dict().keys())
            
            # Create CSV writer
            writer = csv.DictWriter(temp_file, fieldnames=field_names)
            
            # Write header
            writer.writeheader()
            
            # Write single application data row
            writer.writerow(application_data.dict())
            
            # Close the file to ensure all data is written
            temp_file.close()
            
            # Pass the generated CSV to the existing loan application process
            return StreamingResponse(
                process_loan_application(temp_file_path),
                media_type='text/event-stream'
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing application: {str(e)}")
        
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except:
            pass