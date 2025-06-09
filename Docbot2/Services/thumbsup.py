from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path
import shutil # Added for rmtree
import time # Added for retry delay
import gc # Added for garbage collection
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np

class ThumbsUpService:
    """Service for managing persistent Chroma DB collections."""
    
    def __init__(self, base_dir: str = "chroma_collections"):
        """Initialize the ThumbsUp service.
        
        Args:
            base_dir: Base directory for storing Chroma collections
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.active_collections = {}
    
    def get_collection(self, collection_name: str) -> Chroma:
        """Get or create a persistent Chroma collection.
        
        Args:
            collection_name: Name of the collection to access
            
        Returns:
            Chroma instance for the requested collection
        """
        if collection_name in self.active_collections:
            return self.active_collections[collection_name]
        
        # Create collection directory if it doesn't exist
        collection_dir = self.base_dir / collection_name
        collection_dir.mkdir(exist_ok=True)
        
        # Initialize persistent Chroma DB
        chroma_collection = Chroma(
            persist_directory=str(collection_dir),
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        # Store in active collections
        self.active_collections[collection_name] = chroma_collection
        return chroma_collection
    
    def add_data(self, collection_name: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add data to a collection.
        
        Args:
            collection_name: Name of the collection
            data: Dictionary containing the data (must have a 'content' field, can have others like 'user_query')
            metadata: Optional metadata for the document
            
        Returns:
            Result dictionary with status information
        """
        if not data.get('content'):
            return {"status": "error", "message": "Data must contain a 'content' field"}
        
        # Basic sanitization for collection_name to be a valid directory name
        # This replaces common problematic characters with underscores.
        sane_collection_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in collection_name)
        if not sane_collection_name: # Handle empty or all-invalid char names
            return {"status": "error", "message": "Invalid collection name after sanitization."}

        try:
            # Get collection using the sanitized name
            collection = self.get_collection(sane_collection_name)
            
            # Create document
            content = data['content']
            doc_metadata = metadata or {}
            
            # Add additional metadata from the data if available
            # This part will automatically pick up 'user_query' if it's in the 'data' dict
            # and not 'content', and add it to doc_metadata.
            if isinstance(data, dict):
                for key, value in data.items():
                    if key != 'content' and key not in doc_metadata:
                        doc_metadata[key] = value
            
            document = Document(page_content=content, metadata=doc_metadata)
            
            # Add to collection
            collection.add_documents([document])
            
            return {
                "status": "success", 
                "message": f"Data added to collection '{sane_collection_name}'",
                "collection_name": sane_collection_name
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error adding data: {str(e)}"}
    
    def list_collections(self) -> List[str]:
        """List all available collections.
        
        Returns:
            List of collection names
        """
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of items in a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of items in the collection, or 0 if collection doesn't exist/is empty.
        """
        sane_collection_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in collection_name)
        collection_dir = self.base_dir / sane_collection_name
        if not collection_dir.exists():
            return 0
        
        try:
            collection = self.get_collection(sane_collection_name) # This will load it if it exists
            return collection._collection.count() # Access underlying collection object for count
        except Exception: # Broad exception if collection is corrupted or cannot be loaded
            # If collection cannot be loaded, it might be corrupted or empty in a way that count fails
            print(f"Could not get count for collection {sane_collection_name}, assuming 0 or error state.")
            return 0

    def query_collection(self, collection_name: str, query_text: str, k: int = 1, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Query a collection for similar documents.
        
        Args:
            collection_name: Name of the collection
            query_text: Text to search for
            k: Number of results to return
            score_threshold: L2 distance threshold (lower is more similar)
            
        Returns:
            List of matching documents with content, score, and metadata.
        """
        sane_collection_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in collection_name)
        collection_dir = self.base_dir / sane_collection_name
        if not collection_dir.exists():
            print(f"ThumbsUp Query: Collection directory {collection_dir} not found.")
            return []

        try:
            collection = self.get_collection(sane_collection_name)
            if self.get_collection_count(sane_collection_name) == 0: # Use the robust count method
                print(f"ThumbsUp Query: Collection {sane_collection_name} is empty.")
                return []

            results_with_scores = collection.similarity_search_with_score(query_text, k=k)
            
            filtered_results = []
            for doc, score in results_with_scores:
                if score <= score_threshold: # Lower L2 distance score is better
                    filtered_results.append({
                        "content": doc.page_content,
                        "score": float(score), # Ensure score is float
                        "metadata": doc.metadata
                    })
            return filtered_results
        except Exception as e:
            print(f"Error querying collection {sane_collection_name}: {str(e)}")
            return []

    def _build_bm25_index(self, collection_name: str) -> Optional[Tuple[BM25Okapi, List[str]]]:
        """Build or rebuild the BM25 index for a specific collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Tuple of (BM25Okapi index, list of document texts) or None if failed
        """
        sane_collection_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in collection_name)
        collection = self.get_collection(sane_collection_name)
        
        if not collection:
            return None
            
        try:
            # Get all documents from the collection
            results = collection.get()
            
            if not results or not results.get('documents'):
                return None
                
            texts = results['documents']
            
            # Tokenize the documents
            tokenized_corpus = [doc.lower().split() for doc in texts]
            
            # Create BM25 index
            return BM25Okapi(tokenized_corpus), texts
        except Exception as e:
            print(f"Error building BM25 index for collection {sane_collection_name}: {str(e)}")
            return None

    def hybrid_query_collection(self, collection_name: str, query_text: str, k: int = 1, 
                            score_threshold: float = 0.5, hybrid_alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Query a collection using hybrid search (vector similarity + BM25).
        
        Args:
            collection_name: Name of the collection
            query_text: Text to search for
            k: Number of results to return
            score_threshold: L2 distance threshold for vector search (lower is more similar)
            hybrid_alpha: Weight for vector search (1-hybrid_alpha is weight for BM25)
            
        Returns:
            List of matching documents with content, score, and metadata.
        """
        sane_collection_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in collection_name)
        collection_dir = self.base_dir / sane_collection_name
        if not collection_dir.exists():
            print(f"ThumbsUp Hybrid Query: Collection directory {collection_dir} not found.")
            return []

        try:
            collection = self.get_collection(sane_collection_name)
            if self.get_collection_count(sane_collection_name) == 0:
                print(f"ThumbsUp Hybrid Query: Collection {sane_collection_name} is empty.")
                return []
                
            # Get more candidates than requested for hybrid search
            k_candidates = max(k * 3, 15)
            
            # Step 1: Vector search
            results_with_scores = collection.similarity_search_with_score(query_text, k=k_candidates)
            
            semantic_results = []
            for doc, score in results_with_scores:
                semantic_results.append({
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                    "source": "vector"
                })
            
            # Step 2: BM25 search
            bm25_results = []
            bm25_data = self._build_bm25_index(sane_collection_name)
            
            if bm25_data:
                bm25_index, texts = bm25_data
                
                # Tokenize the query
                tokenized_query = query_text.lower().split()
                
                # Get BM25 scores
                bm25_scores = bm25_index.get_scores(tokenized_query)
                
                # Get top k_candidates document indices by BM25 score
                top_bm25_indices = np.argsort(bm25_scores)[::-1][:k_candidates]
                
                # Get collection data for metadata
                results = collection.get()
                metadatas = results.get('metadatas', [])
                documents = results.get('documents', [])
                
                # Collect BM25 results
                for idx in top_bm25_indices:
                    if idx < len(documents) and idx < len(metadatas):
                        content = documents[idx]
                        metadata = metadatas[idx]
                        bm25_results.append({
                            "content": content,
                            "score": float(bm25_scores[idx]),
                            "metadata": metadata,
                            "source": "bm25"
                        })
            
            # Step 3: Combine results with hybrid scoring
            combined_results = {}
            
            # Add semantic results with normalized scores
            valid_semantic_scores = [r['score'] for r in semantic_results if r['score'] > 0]
            max_semantic_score = max(valid_semantic_scores) if valid_semantic_scores else 1.0
            
            for result in semantic_results:
                content = result['content']
                # Normalize score: 1 - (score / max_score) for distance scores to make higher better
                normalized_score = 1.0 - (result['score'] / max_semantic_score) if max_semantic_score > 0 else 0.0
                
                if content not in combined_results:
                    combined_results[content] = {
                        'content': content,
                        'metadata': result['metadata'],
                        'hybrid_score': hybrid_alpha * normalized_score,
                        'vector_score': result['score']
                    }
            
            # Add BM25 results with normalized scores
            valid_bm25_scores = [r['score'] for r in bm25_results if r['score'] > 0]
            max_bm25_score = max(valid_bm25_scores) if valid_bm25_scores else 1.0
            
            for result in bm25_results:
                content = result['content']
                normalized_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0.0
                
                if content in combined_results:
                    combined_results[content]['bm25_score'] = result['score']
                    combined_results[content]['hybrid_score'] += (1 - hybrid_alpha) * normalized_score
                else:
                    combined_results[content] = {
                        'content': content,
                        'metadata': result['metadata'],
                        'hybrid_score': (1 - hybrid_alpha) * normalized_score,
                        'bm25_score': result['score']
                    }
            
            # Sort by hybrid score
            results = list(combined_results.values())
            results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            # Return top k results
            top_results = []
            for i, result in enumerate(results[:k]):
                score_to_report = result.get('hybrid_score', 0.0)
                top_results.append({
                    'content': result['content'],
                    'score': score_to_report,
                    'metadata': result['metadata']
                })
                
            return top_results
            
        except Exception as e:
            print(f"Error performing hybrid search on collection {sane_collection_name}: {str(e)}")
            return []

    def clear_collection(self, collection_name: str) -> Dict[str, Any]:
        """Clear a specific collection by deleting its data.
        
        Args:
            collection_name: Name of the collection to clear.
            
        Returns:
            Result dictionary with status information.
        """
        sane_collection_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in collection_name)
        collection_dir = self.base_dir / sane_collection_name
        
        collection_cleared_by_chroma = False
        if sane_collection_name in self.active_collections:
            try:
                # Get reference to collection before removing from active_collections
                collection_instance = self.active_collections[sane_collection_name]
                
                # Forcefully close any existing connections
                if hasattr(collection_instance, '_client'):
                    print(f"Attempting to delete collection '{sane_collection_name}' via Chroma client.")
                    try:
                        # First try to delete the collection from ChromaDB
                        if hasattr(collection_instance._client, 'delete_collection'):
                            collection_instance._client.delete_collection(name=sane_collection_name)
                            collection_cleared_by_chroma = True
                            print(f"Collection '{sane_collection_name}' deleted by Chroma client.")
                        
                        # Then try to close any client connections
                        if hasattr(collection_instance._client, '_producer'):
                            if hasattr(collection_instance._client._producer, 'close'):
                                collection_instance._client._producer.close()
                        
                        # Try to clean up any other resources
                        if hasattr(collection_instance, 'persist'):
                            collection_instance.persist()
                    except Exception as e:
                        print(f"Warning during Chroma cleanup: {str(e)}")
                
                # Remove from active collections
                del self.active_collections[sane_collection_name]
                # Encourage garbage collection of any resources
                del collection_instance
                gc.collect()
            except Exception as e:
                print(f"Error during collection cleanup: {str(e)}")
                # Continue with deletion attempt even if this part fails
    
        # Check if directory exists before attempting deletion
        if collection_dir.exists() and collection_dir.is_dir():
            max_retries = 5
            retry_delay = 1  # seconds
            
            # Try renaming the directory first (often works when direct deletion fails)
            try:
                temp_name = f"{sane_collection_name}_deleting_{int(time.time())}"
                temp_dir = self.base_dir / temp_name
                os.rename(collection_dir, temp_dir)
                collection_dir = temp_dir  # Update path for deletion attempts
                print(f"Successfully renamed directory to {temp_dir}")
            except Exception as e:
                print(f"Failed to rename directory {collection_dir}: {str(e)}")
                # Continue with original path if rename fails
            
            # Now try to delete the directory
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(collection_dir)
                    print(f"Successfully removed directory {collection_dir} on attempt {attempt + 1}.")
                    
                    message = f"Collection '{sane_collection_name}' and its directory cleared."
                    if collection_cleared_by_chroma:
                        message = f"Collection '{sane_collection_name}' deleted by Chroma client and directory removed."
                    return {"status": "success", "message": message}
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} to remove {collection_dir} failed with PermissionError: {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        # On final attempt, create a marker file for future cleanup
                        try:
                            marker_file = collection_dir / "_TO_DELETE"
                            with open(marker_file, "w") as f:
                                f.write(f"Marked for deletion at {time.ctime()}")
                            print(f"Created deletion marker file at {marker_file}")
                            return {
                                "status": "partial_success", 
                                "message": f"Collection '{sane_collection_name}' removed from memory but directory could not be fully deleted. Marked for later cleanup."
                            }
                        except:
                            # If even the marker file can't be created, just return an error
                            print(f"Final attempt to remove {collection_dir} failed with PermissionError: {e}.")
                            return {"status": "error", "message": f"Error clearing collection directory '{sane_collection_name}' after multiple retries: {str(e)}"}
                except Exception as e:
                    print(f"Error removing directory {collection_dir}: {str(e)}")
                    return {"status": "error", "message": f"Error clearing collection directory '{sane_collection_name}': {str(e)}"}
            
            # This part should ideally not be reached
            return {"status": "error", "message": f"Failed to clear collection '{sane_collection_name}' directory after {max_retries} retries."}
        
        elif collection_cleared_by_chroma:
            return {"status": "success", "message": f"Collection '{sane_collection_name}' deleted by Chroma client. Directory was not present."}
        else:
            return {"status": "not_found", "message": f"Collection '{sane_collection_name}' not found (neither in active memory nor on disk), nothing to clear."}

# Create a singleton instance
thumbsup_service = ThumbsUpService()