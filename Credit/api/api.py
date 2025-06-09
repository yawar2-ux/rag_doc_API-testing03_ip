import os
import uuid
import tempfile
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from groq import Groq
from typing import List
import re
from dotenv import load_dotenv
from ..Services import OCR, OcrProcess, Image_OcrProcess
from ..AgentsServices import AgentOrchestration
from ..Services.ManualFormFieldConfig import LoanApplication
from ..Services.FieldConfig import STANDARD_CSV_FILENAME, TEMP_DIR, OUTPUT_DIR, STANDARD_CSV_PATH
from ..Services.CleanupUtils import cleanup_temp_files, cleanup_output_files

# Load environment variables from .env file
load_dotenv()

credit_router = APIRouter()

def save_to_csv(data, file_path=STANDARD_CSV_PATH):
    """
    Save data (single dict or list of dicts) to a CSV file
    
    Args:
        data (dict or list): Data to save (single dict or list of dicts)
        file_path (str): Path to save the CSV file, defaults to STANDARD_CSV_PATH
    """
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    df.to_csv(file_path, index=False)

def cleanup_temp_directories():
    """
    Legacy wrapper for cleanup_temp_files - forwards to centralized implementation
    """
    cleanup_temp_files()

@credit_router.post("/process-pdf")
async def process_pdf(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDF files, perform OCR, extract structured data, and save results to a combined CSV
    
    Returns:
    - File paths for the generated CSV and OCR text files
    """
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")
    
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, 
                detail=f"File '{file.filename}' is not a PDF"
            )
    
    try:
        # Create unique identifier for this batch process (for OCR text files only)
        batch_process_id = str(uuid.uuid4())
        
        # Initialize Groq client with proper error handling
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        client = Groq(api_key=api_key)
        
        all_extracted_data = []
        ocr_text_file_paths = []
        all_ocr_errors = {}
        
        # Process each PDF file
        for file in files:
            file_base_name = os.path.splitext(file.filename)[0]
            safe_filename = "".join(c if c.isalnum() else "_" for c in file_base_name)
            process_id = str(uuid.uuid4())
            
            # Define output paths
            ocr_text_file_path = os.path.join(OUTPUT_DIR, f"{safe_filename}_{process_id}_ocr.txt")
            ocr_text_file_paths.append(ocr_text_file_path)
            
            # Create temporary files and directories
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=TEMP_DIR) as temp_file:
                pdf_path = temp_file.name
                content = await file.read()
                temp_file.write(content)
            
            temp_image_folder = os.path.join(TEMP_DIR, f"{process_id}")
            os.makedirs(temp_image_folder, exist_ok=True)
            
            # Extract OCR text from PDF
            image_files = OCR.split_pdf_to_images(pdf_path, temp_image_folder)
            
            # Modified to catch and track OCR errors
            ocr_text, page_errors = OCR.extract_text_from_images(image_files, client)
            
            if page_errors:
                all_ocr_errors[file.filename] = page_errors
            
            # Save OCR text to file
            with open(ocr_text_file_path, "w", encoding="utf-8") as f:
                f.write(ocr_text)
            
            # Extract structured data from OCR text
            extracted_data = OcrProcess.extract_structured_data(ocr_text, client)
            extracted_data["Application"] = file.filename
            all_extracted_data.append(extracted_data)
            
            # Clean up temporary files (but not output files)
            try:
                os.remove(pdf_path)
                import shutil
                shutil.rmtree(temp_image_folder)
            except:
                pass  # Ignore cleanup errors
        
        # Save data and process applications using standard path
        save_to_csv(all_extracted_data)
        processing_results = AgentOrchestration.process_applications(STANDARD_CSV_PATH)
        
        # Only clean temporary files (not output) during processing
        try:
            cleanup_temp_files()
        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {str(cleanup_error)}")
        
        # Add OCR errors to the response if they exist
        response_content = processing_results["processing_results"]
        if all_ocr_errors:
            response_content["ocr_errors"] = all_ocr_errors
        
        # Return the processing_results part
        response = JSONResponse(
            status_code=200,
            content=response_content
        )
        
        # Clean output files only after response is prepared
        cleanup_output_files()
        
        return response
    
    except ValueError as e:
        # Handle known configuration errors separately
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@credit_router.post("/process-csv")
async def process_csv(file: UploadFile = File(...)):
    """
    Process a CSV file directly with loan application data
    
    Returns:
    - Processing results from the AgentOrchestration service
    """
    # Validate file is a CSV
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Save uploaded CSV file to standard path
        with open(STANDARD_CSV_PATH, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the application using standard path
        processing_results = AgentOrchestration.process_applications(STANDARD_CSV_PATH)
        
        # Clean up temp files only (not output)
        cleanup_temp_files()
        
        # Prepare response
        response = JSONResponse(
            status_code=200,
            content=processing_results["processing_results"]
        )
        
        # Clean output files after response is prepared
        cleanup_output_files()
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@credit_router.post("/process-manualapplication")
async def create_application_csv(application_data: LoanApplication):
    """
    Process manually entered application data
    """
    try:
        # Use the custom method to convert field names correctly
        app_dict = application_data.to_original_field_dict()
        
        # Add unique application ID if not present
        if "Application" not in app_dict:
            app_dict["Application"] = f"MAN-{str(uuid.uuid4())[:8]}"
        
        # Save to CSV
        save_to_csv(app_dict)
        
        # Process the application
        processing_results = AgentOrchestration.process_applications(STANDARD_CSV_PATH)
        
        # Clean output files after processing is complete
        cleanup_output_files()
        
        # Return the processing_results part
        return processing_results["processing_results"]
    except Exception as e:
        return {"status": "error", "message": f"Error processing manual application: {str(e)}"}

@credit_router.post("/process-application-images/")
async def process_application_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple application image files (CUST####-#.jpg format)")
):
    """
    Process multiple loan application images and extract field values.
    
    - Upload multiple images following the naming convention CUST####-#.jpg
    - Images with the same prefix (before the hyphen) are considered part of the same application
    - Returns a CSV file with one row per application
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    for file in files:
        # Validate filename format
        if not re.match(r'^[A-Za-z0-9]+-\d+\.(jpg|jpeg|png)$', file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid filename format: {file.filename}. Expected format: CUSTXXXX-#.jpg"
            )
    
    # Group files by application
    applications = Image_OcrProcess.group_images_by_application(files)
    
    if not applications:
        raise HTTPException(status_code=400, detail="No valid applications found")
    
    # Process all applications - now returns dict with csv_path and errors
    result = await Image_OcrProcess.process_all_applications(applications, background_tasks)
    
    # Extract CSV path from result
    csv_path = result["csv_path"]
    errors = result.get("errors")
    
    # Add this line to process the generated CSV
    processing_results = AgentOrchestration.process_applications(csv_path)
    
    # Add cleanup_output_files to background tasks after processing
    background_tasks.add_task(cleanup_output_files)
    
    # Include error information in the response
    response_content = processing_results["processing_results"]
    if errors:
        response_content["ocr_errors"] = errors
    
    # Return processing results with error information if any
    return JSONResponse(content=response_content)

@credit_router.get("/download-csv", response_class=FileResponse)
async def download_csv():
    """
    Download the processed loan applications CSV file
    
    Returns:
        FileResponse: The CSV file as a downloadable attachment
    """
    if not os.path.exists(STANDARD_CSV_PATH):
        raise HTTPException(status_code=404, detail="CSV file not found. Process applications first.")
    
    return FileResponse(
        path=STANDARD_CSV_PATH,
        filename=STANDARD_CSV_FILENAME,
        media_type="text/csv"
    )


