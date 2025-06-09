from typing import List, Optional, Dict, Tuple
import face_recognition
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import logging
from dataclasses import dataclass
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import requests
from dotenv import load_dotenv
import os
from Multimodel.vector_store import EnhancedVectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FaceSearchResult:
    """Data class for storing face search results"""
    chunk_id: str
    confidence: float
    face_location: tuple
    matched_image: Optional[str] = None  # base64 encoded image

class ImageSearchService:
    """Service for handling face recognition and image search functionality"""
    
    def __init__(self, vector_store: EnhancedVectorStore, model="hog", tolerance=0.5):
        """Initialize the image search service
        
        Args:
            vector_store: Vector store instance for document chunks
            model (str): Face detection model ('hog' or 'cnn')
            tolerance (float): Face matching tolerance (0-1)
        """
        self.vector_store = vector_store
        self.model = model
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
        
        # Ollama configuration
        self.ollama_url = f"{os.getenv('OLLAMA_BASE_URL')}/api/generate"
        self.ollama_model = os.getenv('OLLAMA_MODEL')
        
        if not self.ollama_url or not self.ollama_model:
            raise ValueError("Missing required environment variables for Ollama")

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _send_ollama_request(self, prompt: str, image_data: str = None) -> str:
        """Send request to Ollama API with retries
        
        Args:
            prompt (str): Text prompt for the model
            image_data (str, optional): Base64 encoded image data
            
        Returns:
            str: Model response
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500,
                }
            }
            
            if image_data:
                payload["images"] = [image_data]
            
            self.logger.info("Sending request to Ollama API")
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                verify=False
            )
            
            response.raise_for_status()
            
            result = response.json()
            if not result or 'response' not in result:
                raise ValueError("Invalid response structure from Ollama API")
                
            return result['response']
                    
        except requests.exceptions.RequestException:
            self.logger.error("Request to Ollama API failed", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in Ollama request: {str(e)}")
            raise

    def _process_input_image(self, image_data: bytes) -> Tuple[np.ndarray, list, list]:
        """Process input image and extract face encodings
        
        Args:
            image_data (bytes): Raw image data
            
        Returns:
            Tuple containing:
            - np.ndarray: Image array
            - list: Face locations
            - list: Face encodings
            
        Raises:
            ValueError: If no faces found in image
        """
        image = face_recognition.load_image_file(io.BytesIO(image_data))
        
        # Detect faces
        face_locations = face_recognition.face_locations(image, model=self.model)
        if not face_locations:
            raise ValueError("No faces found in the input image")
            
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings:
            raise ValueError("Failed to encode faces in the image")
            
        return image, face_locations, face_encodings

    def _find_matching_face(self, face_encoding: list) -> Optional[FaceSearchResult]:
        """Find the best matching face in stored chunks
        
        Args:
            face_encoding (list): Encoding of the face to match
            
        Returns:
            Optional[FaceSearchResult]: Best matching face result or None
        """
        best_match = None
        best_confidence = 0

        # Get all chunks
        chunks = self.vector_store.collection.get()
        
        for chunk_id in chunks['ids']:
            chunk_dir = Path(self.vector_store.collection.get(ids=[chunk_id])['metadatas'][0]['source_dir'])
            image_path = chunk_dir / 'image.png'
            
            if not image_path.exists():
                continue

            try:
                # Load and process chunk image
                chunk_image = face_recognition.load_image_file(image_path)
                chunk_face_locations = face_recognition.face_locations(chunk_image, model=self.model)
                
                if not chunk_face_locations:
                    continue
                    
                chunk_encodings = face_recognition.face_encodings(chunk_image, chunk_face_locations)
                
                # Compare faces
                for idx, chunk_encoding in enumerate(chunk_encodings):
                    distance = face_recognition.face_distance([chunk_encoding], face_encoding)[0]
                    confidence = 1 - distance
                    
                    if confidence > best_confidence and confidence >= self.tolerance:
                        # Convert matched image to base64
                        img = Image.fromarray(chunk_image)
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        best_match = FaceSearchResult(
                            chunk_id=chunk_id,
                            confidence=confidence,
                            face_location=chunk_face_locations[idx],
                            matched_image=img_str
                        )
                        best_confidence = confidence
                        
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_id}: {e}")
                continue

        return best_match

    def _get_chunk_content(self, chunk_id: str) -> Dict:
        """Get all content associated with a chunk
        
        Args:
            chunk_id (str): ID of the chunk
            
        Returns:
            Dict containing text content, image descriptions, and table data
        """
        chunk_dir = Path(self.vector_store.collection.get(ids=[chunk_id])['metadatas'][0]['source_dir'])
        content = {}

        # Get main text content
        text_path = chunk_dir / 'text.txt'
        if text_path.exists():
            content['text'] = text_path.read_text()

        # Get image description
        img_desc_path = chunk_dir / 'image_description.txt'
        if img_desc_path.exists():
            content['image_description'] = img_desc_path.read_text()

        # Get tables and their descriptions
        tables = []
        for table_path in chunk_dir.glob('table_*.csv'):
            table_num = table_path.stem.split('_')[1]
            table_desc_path = chunk_dir / f'table_{table_num}_description.txt'
            
            if table_desc_path.exists():
                tables.append({
                    'table_content': table_path.read_text(),
                    'table_description': table_desc_path.read_text()
                })
        
        if tables:
            content['tables'] = tables

        return content

    def search_and_respond(self, image_data: bytes) -> Dict:
        """Main method to process image search and generate response
        
        Args:
            image_data (bytes): Raw image data to process
            
        Returns:
            Dict containing:
            - message: Analysis response
            - confidence: Match confidence score
            - matched_image: Base64 encoded matched image
        """
        try:
            # Process input image
            self.logger.info("Starting image processing")
            image, face_locations, face_encodings = self._process_input_image(image_data)
            self.logger.info(f"Found {len(face_locations)} faces in input image")
            
            # Find best matching face
            best_match = None
            for face_encoding in face_encodings:
                match = self._find_matching_face(face_encoding)
                if match and (not best_match or match.confidence > best_match.confidence):
                    best_match = match

            if not best_match:
                self.logger.info("No matching faces found")
                return {
                    "message": "No matching faces found in the documents",
                    "confidence": 0,
                    "matched_image": None
                }

            self.logger.info(f"Found match with confidence: {best_match.confidence:.2%}")

            # Get associated content
            chunk_content = self._get_chunk_content(best_match.chunk_id)
            
            # Prepare prompt for analysis
            prompt = f"""Analyze the provided image along with the following information:

Confidence Score: {best_match.confidence:.2%}

Image Description:
{chunk_content.get('image_description', 'No image description available')}

Context:
{chunk_content.get('text', 'No text content available')}"""

            if chunk_content.get('tables'):
                prompt += "\nRelated Table Information:\n"
                for table in chunk_content['tables']:
                    prompt += f"\nTable Description: {table['table_description']}\n"
                    prompt += f"Table Content: {table['table_content']}\n"

            self.logger.info("Sending request to Ollama for analysis")
            
            # Get analysis from Ollama
            llm_response = self._send_ollama_request(
                prompt=prompt,
                image_data=best_match.matched_image
            )

            return {
                "message": llm_response,
                "confidence": best_match.confidence,
                "matched_image": best_match.matched_image
            }

        except ValueError as e:
            self.logger.warning(f"Value error in image search: {str(e)}")
            return {
                "message": str(e),
                "confidence": 0,
                "matched_image": None
            }
        except Exception as e:
            self.logger.error("Error in image search", exc_info=True)
            raise
