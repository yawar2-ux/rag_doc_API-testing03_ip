import os
import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, BackgroundTasks, APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware

# Define the default voice here to avoid import issues
DEFAULT_VOICE = "en-US-AriaNeural"

# Import our modules with simplified error handling
try:
    from TTS.LLM import generate_response, get_available_models
    from TTS.audio import text_to_speech, list_voices
    from TTS.Rag import (
        process_document,
        get_collection,
        list_collections,
        clear_collection
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise

# Create FastAPI app
router = APIRouter()



class VoiceResponse(BaseModel):
    ShortName: str
    Gender: str
    Locale: str

class QueryRequest(BaseModel):
    collection_name: str
    query: str
    k: Optional[int] = 3
    system_message: Optional[str] = "You are a helpful assistant."
    voice: Optional[str] = DEFAULT_VOICE
    model: Optional[str] = "llama-3.3-70b-versatile"  # Default model

@router.get("/voices", response_model=List[VoiceResponse])
async def get_voices():
    """Get list of available voices for text-to-speech"""
    try:
        voices = await list_voices()
        return voices
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing voices: {str(e)}")

@router.post("/upload-document/")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form(...)
):
    """Upload and process document (PDF/TXT) for a specific collection"""
    try:
        # Save uploaded file
        os.makedirs("documents", exist_ok=True)
        file_path = f"documents/{collection_name}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process document in background to avoid timeouts with large files
        background_tasks.add_task(
            process_document,
            file_path,
            collection_name
        )

        return {
            "message": f"Document uploaded and processing started for collection '{collection_name}'",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/list-collections/")
async def get_collections():
    """List all available collections"""
    try:
        collections = list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@router.delete("/clear-collections")
async def clear_all_collections_endpoint():
    """Clear all collections from the database and related document files"""
    try:
        # Import the function here to ensure we're using the latest version
        from TTS.Rag import clear_all_collections

        success = clear_all_collections()
        if success:
            return {"message": "All collections and associated documents cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear collections")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing collections: {str(e)}")
@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a specific collection by name"""
    try:
        success = clear_collection(collection_name)
        if success:
            return {"message": f"Collection '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found or could not be deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


@router.post("/chat-with-pdf/")
async def chat_with_pdf(query_data: QueryRequest):
    """Chat with PDF and get text and audio response"""
    try:
        # Get relevant documents from collection
        db = get_collection(query_data.collection_name)
        if not db:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found in collection '{query_data.collection_name}' or collection doesn't exist"
            )

        # Get context from retrieved documents
        results = db.similarity_search(query_data.query, k=query_data.k)
        context = "\n\n".join([doc.page_content for doc in results])

        # Create enhanced prompt with context
        enhanced_prompt = f"""Based on the following content, please answer the question.

Content:
{context}

Question: {query_data.query}"""

        # Generate response using LLM with context - now passing the model parameter
        text_response = generate_response(
            enhanced_prompt,
            query_data.system_message,
            query_data.model
        )

        # Convert response to speech
        os.makedirs("audio_outputs", exist_ok=True)
        audio_filename = f"audio_outputs/response_{hash(text_response)}.mp3"

        # Generate audio
        await text_to_speech(text_response, query_data.voice, audio_filename)

        # Return both text and audio file
        return {
            "text_response": text_response,
            "audio_url": f"/get-audio/{os.path.basename(audio_filename)}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.get("/get-audio/{filename}")
async def get_audio(filename: str):
    """Get audio file by filename"""
    audio_path = f"audio_outputs/{filename}"
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")  # Debug output
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")

    # Ensure proper MIME type and headers
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
    }
    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        headers=headers,
        filename=filename  # Ensure browser treats it as a downloadable file with name
    )

@router.get("/models")
async def get_models():
    """Get list of available LLM models"""
    try:
        models = get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

