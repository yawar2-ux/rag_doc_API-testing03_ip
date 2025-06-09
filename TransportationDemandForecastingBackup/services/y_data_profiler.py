#!/usr/bin/env python3
"""
FastAPI application to generate YData Profiling reports from uploaded CSV files.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from ydata_profiling import ProfileReport
import uvicorn

# Directory to store temporary files
TEMP_DIR = Path("temp_reports")
TEMP_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Clean up any existing temporary files
    for temp_file in TEMP_DIR.glob("*"):
        try:
            temp_file.unlink()
        except:
            pass
    yield
    # Shutdown: Clean up temporary files
    for temp_file in TEMP_DIR.glob("*"):
        try:
            temp_file.unlink()
        except:
            pass

app = FastAPI(
    title="YData Profiler API",
    description="Upload CSV files and get downloadable YData Profiling reports",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "YData Profiler API",
        "endpoints": {
            "upload": "/profile-csv/",
            "docs": "/docs"
        }
    }

@app.post("/profile-csv/")
async def create_profile_report(
    file: UploadFile = File(..., description="CSV file to profile"),
    title: Optional[str] = Form(None, description="Custom title for the report"),
    minimal: bool = Form(False, description="Generate minimal report for faster processing")
):
    """
    Upload a CSV file and get a downloadable YData Profiling report.
    
    - **file**: CSV file to analyze
    - **title**: Optional custom title for the report
    - **minimal**: Generate minimal report for large datasets (faster but less detailed)
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    temp_csv_path = TEMP_DIR / f"temp_{unique_id}.csv"
    output_html_path = TEMP_DIR / f"profile_report_{unique_id}.html"
    
    try:
        # Save uploaded file temporarily
        with open(temp_csv_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read CSV with pandas
        try:
            df = pd.read_csv(temp_csv_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Validate that CSV has data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Set report title
        report_title = title if title else f"Profiling Report - {file.filename}"
        
        # Configure profiling based on minimal flag
        if minimal:
            # Minimal configuration for faster processing
            profile = ProfileReport(
                df, 
                title=report_title,
                minimal=True,
                interactions=None,
                correlations=None,
                missing_diagrams=None
            )
        else:
            # Full profiling report
            profile = ProfileReport(df, title=report_title)
        
        # Generate HTML report
        profile.to_file(output_html_path)
        
        # Clean up temporary CSV file
        temp_csv_path.unlink()
        
        # Return the HTML file as download
        return FileResponse(
            path=output_html_path,
            filename=f"ydata_profile_report_{file.filename.replace('.csv', '')}.html",
            media_type="text/html",
            headers={"Content-Disposition": "attachment"}
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up files in case of error
        if temp_csv_path.exists():
            temp_csv_path.unlink()
        if output_html_path.exists():
            output_html_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Error generating profile report: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "YData Profiler API"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)