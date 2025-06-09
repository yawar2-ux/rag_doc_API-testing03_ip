"""
RAG Agent Service: Interface for the RAG system.
"""

from typing import List, Dict, Any, Optional
from ..Agents.AgentRAG import AgentRAG  # Corrected relative import
from ..Agents.MemoryAgent import MemoryAgent # Corrected relative import
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import shutil
from pathlib import Path
from docling.document_converter import DocumentConverter

class RagAgent:
    """
    Service to interact with the RAG system.
    """
    
    def __init__(self):
        """Initialize the underlying RAG agent."""
        self.agent_rag = AgentRAG()
        
    def process_files(self, file_paths: List[str], advanced_extraction: bool = False, perform_ocr: bool = False) -> List[str]:
        """
        Process files (e.g., PDF, text) using the RAG system.
        
        Args:
            file_paths: List of paths to files
            advanced_extraction: If True, use docling for advanced extraction for PDFs
            perform_ocr: If True, perform OCR on images within PDFs
            
        Returns:
            List of processing result messages
        """
        return self.agent_rag.process_files(file_paths, advanced_extraction, perform_ocr=perform_ocr)
    
    def clear_all_data(self) -> Dict[str, str]:
        """
        Clear all stored data including memory, RAG state, and temporary files.
        """
        try:
            self.agent_rag.rag_service.reset_state()
            self.agent_rag.memory_agent.clear_memory()
            
            temp_dirs_to_clear = [
                Path("temp/uploads"),
                Path("temp/extracted_content")
            ]
            
            for temp_dir in temp_dirs_to_clear:
                if temp_dir.exists() and temp_dir.is_dir():
                    for item in temp_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    print(f"Cleared temporary directory: {temp_dir}")
                else:
                    print(f"Temporary directory not found or not a dir: {temp_dir}")
            
            return {"status": "success", "message": "All data cleared successfully."}
            
        except Exception as e:
            print(f"Error during data clearing: {str(e)}")
            return {"status": "error", "message": f"Error clearing data: {str(e)}"}

    # Keep for backward compatibility
    def process_pdfs(self, file_paths: List[str], advanced_extraction: bool = False) -> List[str]:
        """Legacy method that calls process_files."""
        # Note: This legacy method does not expose perform_ocr. 
        # Consider updating or deprecating if OCR is needed through this path.
        return self.process_files(file_paths, advanced_extraction, perform_ocr=False)
    
    def generate_response(self, user_id: str, message: str, hybrid_alpha: float = 0.7, use_reranking: bool = True, temperature: float = 0.36, max_tokens: int = 1024, top_p: float = 1.0, thumbsup_score_threshold: float = 0.78) -> Dict[str, Any]:
        """
        Generate a response to a user message.
        
        Args:
            user_id: Unique identifier for the user
            message: The user's query text
            hybrid_alpha: Weight for semantic search in hybrid retrieval (0-1)
            use_reranking: Whether to use cross-encoder reranking
            temperature: LLM temperature
            max_tokens: LLM max tokens
            top_p: LLM top_p
            thumbsup_score_threshold: Threshold for thumbs up score
            
        Returns:
            Dictionary with response text and source references
        """
        return self.agent_rag.process_query(user_id, message, hybrid_alpha, use_reranking, temperature, max_tokens, top_p, thumbsup_score_threshold)
