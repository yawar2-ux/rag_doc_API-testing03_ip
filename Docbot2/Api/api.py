"""
API Module: FastAPI endpoints for the RAG document bot.

This module provides:
- API endpoints for document uploading and processing
- Chat functionality with document context
- Input validation and error handling

Input: API requests with files or messages
Output: API responses with processing results
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import shutil
from pathlib import Path

from ..Services.RagAgent import RagAgent  # Corrected relative import
from ..Services.thumbsup import thumbsup_service # Import thumbsup service
from ..Services.thumbsdown import thumbsdown_service # Import thumbsdown service

# Create router
docbot_router = APIRouter()

# Create upload directory in temp folder
UPLOAD_DIR = Path("./temp/uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Initialize RagAgent
rag_agent = RagAgent()

class ChatRequest(BaseModel):
    """
    Request model for chat API.
    
    Attributes:
        user_id: Unique identifier for the user
        message: The user's query text
        hybrid_alpha: Optional weight for hybrid search (0-1)
        use_reranking: Optional flag to enable/disable reranking
        temperature: Optional LLM temperature
        max_tokens: Optional LLM max tokens
        top_p: Optional LLM top_p
        thumbsup_score_threshold: Optional threshold for thumbsup filtering
    """
    user_id: str
    message: str
    hybrid_alpha: Optional[float] = 0.7  # Default weight for semantic search vs BM25
    use_reranking: Optional[bool] = True  # Default to use reranking if available
    temperature: Optional[float] = 0.36
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 1.0
    thumbsup_score_threshold: Optional[float] = 0.78 # Default value for the new field

class ThumbsUpAddRequest(BaseModel):
    user_id: str
    content: str
    user_query: Optional[str] = None  # Add user_query field as optional
class ThumbsDownAddRequest(BaseModel):
    user_id: str
    content: str
    user_query: Optional[str] = None # Add user_query field

class ThumbsUpClearRequest(BaseModel):
    user_id: str

class ChatResponse(BaseModel):
    """
    Response model for chat API.
    
    Attributes:
        response: The generated text response
        sources: Optional list of document sources
        debug_info: Optional debug information
    """
    response: str
    sources: Optional[List[dict]] = None
    debug_info: Optional[dict] = None

@docbot_router.post("/upload", response_model=List[str])
async def upload_files( # Renamed from upload_pdfs
    files: List[UploadFile] = File(...), 
    user_id: str = Form(...), # user_id is present but not directly used in this version for collection naming
    advanced_extraction: bool = Form(False),
    perform_ocr: bool = Form(False) # New parameter for OCR
):
    """
    Upload and process multiple files.
    
    Args:
        files: List of files to upload
        user_id: Unique identifier for the user (for potential future use, e.g., user-specific storage)
        advanced_extraction: If True, use docling for advanced extraction for supported types (e.g. PDFs)
        perform_ocr: If True, perform OCR on images within PDFs
        
    Returns:
        List of processing result messages
        
    Raises:
        HTTPException: If no files provided or processing errors
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")
    
    file_paths_for_cleanup = [] # Keep track of all file paths for potential cleanup
    processed_file_paths_for_agent = []
    
    try:
        for file in files:
            original_filename = Path(file.filename)
            base_name = original_filename.stem
            extension = original_filename.suffix
            
            counter = 0
            unique_filename = original_filename.name
            file_path = UPLOAD_DIR / unique_filename
            
            # Handle filename conflicts by appending _1, _2, etc.
            while file_path.exists():
                counter += 1
                unique_filename = f"{base_name}_{counter}{extension}"
                file_path = UPLOAD_DIR / unique_filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            processed_file_paths_for_agent.append(str(file_path))
            file_paths_for_cleanup.append(file_path) 
                
        # Process the files
        results = rag_agent.process_files(
            processed_file_paths_for_agent, 
            advanced_extraction,
            perform_ocr=perform_ocr # Pass OCR flag
        )
        
        # Optionally, clean up files from UPLOAD_DIR after successful processing by RAG agent
        # if they are stored elsewhere by the RAG agent or if they are temporary.
        # For now, we'll leave them in UPLOAD_DIR as the /files/{filename} endpoint serves from there.

        return results
    except Exception as e:
        # Clean up any saved files in case of error during processing or upload
        for path in file_paths_for_cleanup:
            if Path(path).exists():
                os.remove(path)
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@docbot_router.post("/thumbsup/add", response_model=Dict[str, Any])
async def add_thumbsup(request: ThumbsUpAddRequest):
    """
    Add a piece of content to the user's thumbs_up collection.
    """
    try:
        collection_name = f"{request.user_id}_thumbsup"
        # Metadata can be simple, just indicating the source
        metadata = {"source": "thumbs_up_user_submission", "original_user_id": request.user_id}
        
        # Prepare data for thumbsup_service, including user_query if provided
        data_to_add = {"content": request.content}
        if request.user_query:
            data_to_add["user_query"] = request.user_query
            
        result = thumbsup_service.add_data(
            collection_name=collection_name,
            data=data_to_add,
            metadata=metadata
        )
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding thumbs up data: {str(e)}")

@docbot_router.post("/thumbsup/clear", response_model=Dict[str, Any])
async def clear_thumbsup(request: ThumbsUpClearRequest):
    """
    Clear the user's thumbs_up and thumbs_down collections.
    """
    try:
        # Clear thumbsup collection
        thumbsup_collection_name = f"{request.user_id}_thumbsup"
        thumbsup_result = thumbsup_service.clear_collection(thumbsup_collection_name)

        if thumbsup_result["status"] == "error":
            raise HTTPException(status_code=500, detail=f"Error clearing thumbs-up data: {thumbsup_result['message']}")

        # Clear thumbsdown collection
        thumbsdown_collection_name = f"{request.user_id}_thumbsdown"
        thumbsdown_result = thumbsdown_service.clear_collection(thumbsdown_collection_name)

        if thumbsdown_result["status"] == "error":
            # If thumbsup was cleared or partially cleared, this error for thumbsdown is still critical.
            raise HTTPException(status_code=500, detail=f"Error clearing thumbs-down data: {thumbsdown_result['message']}")

        # Determine overall status and combine messages
        messages = []
        overall_status = "success"

        # Prioritize "partial_success" if it occurs and no "error"
        if thumbsup_result["status"] == "partial_success" or thumbsdown_result["status"] == "partial_success":
            overall_status = "partial_success"
        
        messages.append(f"Thumbs-up collection: {thumbsup_result['message']}")
        messages.append(f"Thumbs-down collection: {thumbsdown_result['message']}")
        
        return {"status": overall_status, "message": " | ".join(messages)}

    except HTTPException as http_exc: # Re-raise HTTPException
        raise http_exc
    except Exception as e: # Catch any other unexpected exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while clearing feedback data: {str(e)}")

@docbot_router.post("/thumbsdown/add", response_model=Dict[str, Any])
async def add_thumbsdown(request: ThumbsDownAddRequest):
    """
    Add a piece of content to the user's thumbs_down collection.
    """
    try:
        collection_name = f"{request.user_id}_thumbsdown"
        # Metadata can be simple, just indicating the source
        metadata = {"source": "thumbs_down_user_submission", "original_user_id": request.user_id}
        result = thumbsdown_service.add_data(
            collection_name=collection_name,
            data={"content": request.content},
            metadata=metadata
        )
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding thumbs down data: {str(e)}")

@docbot_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message using RAG and LLM.
    
    Args:
        request: ChatRequest containing user_id and message
        
    Returns:
        ChatResponse with generated response and sources
        
    Raises:
        HTTPException: If error occurs during processing
    """
    try:
        response = rag_agent.generate_response(
            request.user_id, 
            request.message,
            hybrid_alpha=request.hybrid_alpha,
            use_reranking=request.use_reranking,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            thumbsup_score_threshold=request.thumbsup_score_threshold # Pass the new threshold
        )
        return ChatResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@docbot_router.post("/cleardata", response_model=dict)
async def clear_data():
    """
    Clear all data including uploaded files, extracted content,
    vector stores, and conversation memory.
    """
    try:
        result = rag_agent.clear_all_data()
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return {"message": "All data cleared successfully."} # Consistent success message
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")

@docbot_router.get("/files/{filename}")
async def get_file(filename: str):
    """
    Serve a file from the UPLOAD_DIR.
    """
    # Basic security: ensure filename does not try to escape the directory
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_location = UPLOAD_DIR / filename
    
    if not file_location.exists() or not file_location.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    
    # Ensure the resolved path is still within UPLOAD_DIR (stricter check)
    if UPLOAD_DIR.resolve() not in file_location.resolve().parents:
        raise HTTPException(status_code=403, detail="Access forbidden.")

    return FileResponse(path=str(file_location), media_type='application/pdf', filename=filename)