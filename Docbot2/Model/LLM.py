"""
LLM Module: Initializes the large language model for the RAG system.

This module:
- Loads environment variables for API keys
- Initializes the Groq model for text generation
- Provides error handling and fallback options

Input: Environment variables (API keys)
Output: Initialized LLM instance for use in chains
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.llms import FakeListLLM # Updated import for fallback

# Load API keys from environment variables
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
print(f"GROQ API Key found: {'Yes' if api_key else 'No'}") # Useful for startup diagnostics

# Create LangChain-compatible Groq model
try:
    Groq_Model = ChatGroq(
        api_key=api_key, 
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct", 
        temperature=0.1, 
        max_tokens=4000
    )
    print("LLM loaded successfully")
except Exception as e:
    print(f"Error initializing Groq model: {str(e)}")
    Groq_Model = FakeListLLM(responses=["Error: Groq model failed to initialize. Check your API key."])
    print("Using fallback model due to initialization error")

def get_llm_instance(temperature: float, max_tokens: int, top_p: float) -> ChatGroq:
    """
    Creates and returns a ChatGroq instance configured with the specified parameters.
    """
    current_api_key = os.getenv('GROQ_API_KEY')
    if not current_api_key:
        print("Warning: GROQ_API_KEY not found for get_llm_instance. Using fallback if Groq_Model is also fallback.")
        if isinstance(Groq_Model, FakeListLLM):
            return FakeListLLM(responses=["Error: Groq model failed to initialize in get_llm_instance. Check API key."])

    model_name_to_use = "meta-llama/llama-4-maverick-17b-128e-instruct"
    if isinstance(Groq_Model, ChatGroq) and hasattr(Groq_Model, 'model_name'):
        model_name_to_use = Groq_Model.model_name
    
    try:
        llm = ChatGroq(
            api_key=current_api_key,
            model_name=model_name_to_use,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens > 0 else None,
            model_kwargs={"top_p": top_p}
        )
        return llm
    except Exception as e:
        print(f"Error initializing ChatGroq instance in get_llm_instance: {str(e)}")
        return FakeListLLM(responses=[f"Error: Groq model failed to initialize with custom params: {str(e)}"])