from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import datetime
from pathlib import Path
import logging
from typing import Optional

from .Spec_bot import (process_request, job_status, extract_document_content,
                      generate_infrastructure_recommendations, RESULTS_DIR, 
                      check_cache_status, clear_cache, cleanup_job)

router = APIRouter()

@router.post("/upload/")
async def upload_file(
    background_tasks: BackgroundTasks, 
    special_instructions: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Process a request for infrastructure recommendations.
    
    Can accept a document file (PDF, DOCX, DOC, CSV) and/or special instructions.
    At least one of the two inputs must be provided.
    
    Returns a job ID which can be used to check status and download results.
    """
    try:
        # Check that at least one input is provided
        if not file and not special_instructions:
            raise HTTPException(
                status_code=400, 
                detail="Either a file or special instructions must be provided"
            )
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status with default values
        job_status_data = {
            "status": "queued",
            "progress": 0,
            "special_instructions": special_instructions,
            "submitted_at": datetime.datetime.now().isoformat(),
            "is_text_only": file is None
        }
        
        file_path = None
        
        if file:
            # Process the file if one was uploaded
            file_path = await process_uploaded_file(file, job_id)
            
            # Add file information to job status
            job_status_data["file_name"] = file.filename
            job_status_data["file_path"] = str(file_path)
        
        # Store the job status
        job_status[job_id] = job_status_data
        
        # Process the document or instructions in the background
        background_tasks.add_task(process_request, file_path, job_id, special_instructions)
        
        return {"job_id": job_id, "status": "queued"}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_msg = f"Error processing upload: {str(e)}"
        logging.error(f"{error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Check the status of a processing job.
    
    Returns the current status, progress, and error (if any).
    """
    try:
        if job_id not in job_status:
            raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
        
        return job_status[job_id]
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_msg = f"Error checking job status: {str(e)}"
        logging.error(f"{error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/download/csv/{job_id}")
async def download_csv(job_id: str):
    """
    Download the CSV results for a completed job.
    
    Returns the CSV file as an attachment.
    """
    try:
        if job_id not in job_status:
            raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
        
        if job_status[job_id]["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job is not completed. Current status: {job_status[job_id]['status']}"
            )
        
        if "csv_path" not in job_status[job_id] or not job_status[job_id]["csv_path"]:
            raise HTTPException(status_code=404, detail="CSV result not found")
        
        csv_path = job_status[job_id]["csv_path"]
        
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="CSV file not found on server")
        
        return FileResponse(
            path=csv_path, 
            filename=f"infrastructure_recommendations_{job_id}.csv",
            media_type="text/csv"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_msg = f"Error downloading CSV: {str(e)}"
        logging.error(f"{error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/download/excel/{job_id}")
async def download_excel(job_id: str):
    """
    Download the styled Excel results for a completed job.
    
    Returns the Excel file as an attachment.
    """
    try:
        if job_id not in job_status:
            raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
        
        if job_status[job_id]["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job is not completed. Current status: {job_status[job_id]['status']}"
            )
        
        if "excel_path" not in job_status[job_id] or not job_status[job_id]["excel_path"]:
            raise HTTPException(status_code=404, detail="Excel result not found")
        
        excel_path = job_status[job_id]["excel_path"]
        
        if not os.path.exists(excel_path):
            raise HTTPException(status_code=404, detail="Excel file not found on server")
        
        return FileResponse(
            path=excel_path, 
            filename=f"infrastructure_recommendations_{job_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_msg = f"Error downloading Excel: {str(e)}"
        logging.error(f"{error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/check-cache")
async def check_cache_status_endpoint():
    """Get information about the current cache status"""
    return check_cache_status()

@router.delete("/clear-cache")
async def clear_cache_endpoint():
    """Clear the recommendation cache"""
    return clear_cache()

@router.delete("/cleanup/{job_id}")
async def cleanup_job_endpoint(job_id: str):
    """
    Clean up files and status for a specific job.
    """
    try:
        result = cleanup_job(job_id)  # Use the imported function from Spec_bot.py
        return result
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_msg = f"Error cleaning up job: {str(e)}"
        logging.error(f"{error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

async def process_uploaded_file(file, job_id):
    """Helper function to process uploaded files"""
    # Check file type
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.pdf', '.docx', '.doc', '.csv']:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_extension}. Supported formats: .pdf, .docx, .doc, .csv"
        )
    
    # Create a directory for this job
    upload_dir = Path("./uploads")
    job_dir = upload_dir / job_id
    job_dir.mkdir(exist_ok=True)
    
    # Save the uploaded file
    file_path = job_dir / file.filename
    
    try:
        import shutil
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        error_msg = f"Error saving uploaded file: {str(e)}"
        logging.error(f"{error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        file.file.close()
    
    return file_path
