import os
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Load environment variables from .env file
load_dotenv()

# Define available models
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",  # Default
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen-qwq-32b",
    "gemma2-9b-it",
    

]

# Default model
DEFAULT_MODEL = "llama-3.3-70b-versatile"

def get_groq_client():
    """Get Groq client instance using API key from environment variables"""
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_response(prompt, system_message="You are a helpful assistant.", model=DEFAULT_MODEL):
    """
    Generate a response using the Groq LLM
    Args:
        prompt: User's input text
        system_message: System message to guide the model behavior
        model: Model to use for generation
    Returns:
        Generated text response
    """
    client = get_groq_client()
    
    # Validate model choice
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        temperature=1,
        max_completion_tokens=3000,
        top_p=1,
        stream=False,
        stop=None,
    )

    return chat_completion.choices[0].message.content

def generate_streaming_response(prompt, system_message="You are a helpful assistant.", model=DEFAULT_MODEL):
    """
    Generate a streaming response using the Groq LLM
    Args:
        prompt: User's input text
        system_message: System message to guide the model behavior
        model: Model to use for generation
    Returns:
        Generator yielding chunks of text as they're generated
    """
    client = get_groq_client()
    
    # Validate model choice
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL
    
    stream = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        temperature=1,
        max_completion_tokens=3000,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def get_available_models():
    """Return the list of available models"""
    return AVAILABLE_MODELS

app = FastAPI()

@app.get("/models")
async def get_models():
    """Get list of available LLM models"""
    try:
        models = get_available_models()
        return {
            "models": models,
            "default": DEFAULT_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

if __name__ == "__main__":
    # Example usage when script is run directly
    response = generate_response("Explain the importance of fast language models")
    print(response)