from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field, field_validator  # Change validator to field_validator
from typing import List, Optional
import time
import tiktoken  # Make sure tiktoken is imported

from .LLM import generate_chat_completion, AVAILABLE_MODELS
from .Matrix import router as matrix_router

router = APIRouter()

class ModelConfig(BaseModel):
    model_name: str
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_tokens: int = Field(default=3000, ge=1, le=4096)
    
    # Update to V2 validator style
    @field_validator("model_name")
    @classmethod  # Required for V2 validators
    def validate_model(cls, v):
        if v not in AVAILABLE_MODELS:
            raise ValueError(f"Model '{v}' not available. Choose from {AVAILABLE_MODELS}")
        return v

class PromptRequest(BaseModel):
    prompt: str
    system_message: str = "You are a helpful assistant."
    model_configs: List[ModelConfig]
    use_case: Optional[str] = "General Text Generation"

class ModelResponse(BaseModel):
    model_name: str
    temperature: float
    response: str
    response_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int

class PromptResponse(BaseModel):
    prompt: str
    model_responses: List[ModelResponse]

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate counting if tiktoken fails
        return len(text.split()) * 1.3  # Rough approximation

@router.post("/generate", response_model=PromptResponse)
async def generate_responses(request: PromptRequest):
    model_responses = []
    
    for config in request.model_configs:
        start_time = time.time()
        
        try:
            # Count input tokens
            input_text = f"{request.system_message}\n{request.prompt}"
            input_tokens = count_tokens(input_text)
            
            # Generate response
            response = generate_chat_completion(
                system_message=request.system_message,
                prompt=request.prompt,
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens  # Pass max_tokens to the function
            )
            
            # Count output tokens
            output_tokens = count_tokens(response)
            total_tokens = input_tokens + output_tokens
            
            response_time = time.time() - start_time
            
            model_responses.append(
                ModelResponse(
                    model_name=config.model_name,
                    temperature=config.temperature,
                    response=response,
                    response_time=response_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens
                )
            )
        except Exception as e:
            # Instead of failing the whole request, add an error message as response
            model_responses.append(
                ModelResponse(
                    model_name=config.model_name,
                    temperature=config.temperature,
                    response=f"Error: {str(e)}",
                    response_time=time.time() - start_time,
                    input_tokens=count_tokens(input_text),
                    output_tokens=0,
                    total_tokens=count_tokens(input_text)
                )
            )
    
    return PromptResponse(
        prompt=request.prompt,
        model_responses=model_responses
    )

@router.get("/models")
async def get_available_models():
    return {"available_models": AVAILABLE_MODELS}

@router.get("/health")
async def health_check():
    """Simple health check endpoint for the API"""
    return {"status": "healthy", "available_models": AVAILABLE_MODELS}

# Include the matrix router
router.include_router(matrix_router, prefix="/matrix")