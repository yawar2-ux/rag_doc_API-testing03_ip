import os
from groq import Groq  # Correct import
from dotenv import load_dotenv

load_dotenv()

# Define available models
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",  # Default
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",  
    "meta-llama/llama-4-maverick-17b-128e-instruct",  
    "llama3-8b-8192",
    "llama3-70b-8192",
    "llama-3.1-8b-instant"
]

# Default model
DEFAULT_MODEL = "llama-3.3-70b-versatile"

def get_groq_client():
    """Get Groq client instance using API key from environment variables"""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")
    return Groq(api_key=api_key)

# Define a function to generate chat completion
def generate_chat_completion(system_message, prompt, model=DEFAULT_MODEL, temperature=0.7, max_tokens=3000):
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model}' is not available. Choose from {AVAILABLE_MODELS}.")
    
    client = get_groq_client()
    
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
        temperature=temperature,
        max_completion_tokens=max_tokens,  # Use the parameter
        top_p=1,
        stream=False,
        stop=None,
    )
    
    return chat_completion.choices[0].message.content

def generate_streaming_response(system_message, prompt, model=DEFAULT_MODEL, temperature=0.7):
    """
    Generate a streaming response using the Groq LLM
    Returns:
        Generator yielding chunks of text as they're generated
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model}' is not available. Choose from {AVAILABLE_MODELS}.")
    
    client = get_groq_client()
    
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
        temperature=temperature,
        max_completion_tokens=3000,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Example usage
if __name__ == "__main__":
    system_message = "You are a helpful assistant."
    prompt = "What is the capital of France?"
    
    # Test with default model
    try:
        print(f"Testing with default model...")
        response = generate_chat_completion(system_message, prompt)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error with default model: {e}")
    
    # Test with alternate model if available
    if len(AVAILABLE_MODELS) > 1:
        try:
            alt_model = AVAILABLE_MODELS[1]
            print(f"Testing with {alt_model}...")
            response = generate_chat_completion(system_message, prompt, model=alt_model, temperature=0.5)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error with {alt_model}: {e}")