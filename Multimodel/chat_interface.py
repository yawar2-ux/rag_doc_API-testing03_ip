"""
Chat interface for document-based conversations using LLM and vector search
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from Multimodel.vector_store import EnhancedVectorStore
import base64
from pathlib import Path
import json
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama Configuration from env
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
MODEL = os.getenv('OLLAMA_MODEL')
TEXTMODEL = os.getenv('OLLAMA_TEXT_MODEL')

@dataclass
class ChatResponse:
    """Structure for chat response"""
    bot_response: str
    image: Optional[str] = None  # base64 encoded image

class DocumentChat:
    """Handles document-based chat interactions"""

    def __init__(self, vector_store: EnhancedVectorStore, similarity_threshold: float = 0.5):
        """Initialize chat interface"""
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    def _get_chunk_image(self, chunk_metadata: Dict) -> Optional[str]:
        """Extract and encode image from chunk if available"""
        try:
            source_dir = Path(chunk_metadata.get('source_dir', ''))
            if source_dir.exists():
                image_path = source_dir / 'image.png'
                if image_path.exists():
                    with open(image_path, 'rb') as img_file:
                        return base64.b64encode(img_file.read()).decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Error extracting image: {e}")
            return None

    def _send_ollama_request(self, prompt: str) -> str:
        """Send request to Ollama API"""
        payload = {
            "model": TEXTMODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 500,
            }
        }

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                verify=False
            )

            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.text}")
                return "I apologize, but I encountered an error processing your message."

            result = response.json()
            return result.get('response', '')

        except Exception as e:
            logger.error(f"Error in Ollama request: {e}")
            return "I apologize, but I encountered an error processing your message."

    def chat(self, user_message: str) -> ChatResponse:
        """Process user message and generate response"""
        try:
            # Get relevant context
            results = self.vector_store.search_similar(
                query=user_message,
                top_k=1,  # Get only the most relevant chunk
                threshold=self.similarity_threshold
            )

            if not results:
                return ChatResponse(
                    bot_response="I couldn't find any relevant information to answer your question."
                )

            most_relevant = results[0]

            # Format prompt with context
            prompt = f"""Based on the following context, provide a natural and concise response to the user's question.

Context:
{most_relevant['content']}

User question: {user_message}

Provide a natural conversational response without mentioning the context or using phrases like 'based on the provided context'."""

            # Generate response using Ollama
            response = self._send_ollama_request(prompt)

            # Get image if available
            image = self._get_chunk_image(most_relevant['metadata'])

            return ChatResponse(
                bot_response=response,
                image=image
            )

        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return ChatResponse(
                bot_response="I apologize, but I encountered an error processing your message. Please try again."
            )
