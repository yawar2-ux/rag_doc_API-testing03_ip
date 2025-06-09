from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import logging
import json
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class ChunkContent:
    """Data class for chunk content"""
    chunk_id: str
    text_content: str
    files: List[str]
    metadata: Dict
    created_at: str

class EnhancedVectorStore:
    """Enhanced vector store for document chunks with improved organization and error handling"""

    def __init__(self, collection_name: str = "records"):
        """Initialize vector store with offline embedding model"""
        try:
            # Initialize embedding model from local path
            model_path = Path(os.path.join(base_path, "sentence_transformer"))
            if not model_path.exists():
                raise Exception(
                    "Model directory not found. Please run download_models.py first."
                )

            self.model = SentenceTransformer(str(model_path))
            logger.info("Successfully loaded local embedding model")

            # Initialize ChromaDB with persistent storage
            persist_dir = Path("chroma_store")
            persist_dir.mkdir(exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=chromadb.Settings(
                    anonymized_telemetry=False  # Disable telemetry
                )
            )

            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"Successfully initialized vector store with collection: {collection_name}")

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def process_text_content(self, chunk_dir: Path) -> Optional[ChunkContent]:
        """Process content from a chunk directory"""
        try:
            chunk_content = []
            files_processed = []

            # Process main text content
            text_file = chunk_dir / "text.txt"
            if text_file.exists():
                content = text_file.read_text(encoding='utf-8')
                chunk_content.append(content)
                files_processed.append(text_file.name)

            # Process image descriptions
            img_desc_file = chunk_dir / "image_description.txt"
            if img_desc_file.exists():
                content = img_desc_file.read_text(encoding='utf-8')
                chunk_content.append(f"Image Description: {content}")
                files_processed.append(img_desc_file.name)

            # Process tables
            for table_file in chunk_dir.glob("table_*.csv"):
                try:
                    df = pd.read_csv(table_file)
                    table_text = [f"Table {table_file.stem}:"]
                    table_text.append(f"Columns: {', '.join(df.columns)}")
                    for _, row in df.iterrows():
                        table_text.append(" | ".join(f"{col}: {val}" for col, val in row.items()))
                    chunk_content.append("\n".join(table_text))
                    files_processed.append(table_file.name)
                except Exception as e:
                    logger.warning(f"Error processing table {table_file}: {e}")

            if not chunk_content:
                return None

            return ChunkContent(
                chunk_id=chunk_dir.name,
                text_content="\n\n".join(chunk_content),
                files=files_processed,
                metadata={
                    "has_image": img_desc_file.exists(),
                    "table_count": len(list(chunk_dir.glob("table_*.csv"))),
                    "source_dir": str(chunk_dir)
                },
                created_at=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Error processing chunk directory {chunk_dir}: {e}")
            return None

    def add_chunks_to_db(self, chunks_dir: Path) -> Dict:
        """Process and add chunks to vector store"""
        try:
            chunks_info = {"total_chunks": 0, "processed_chunks": 0, "errors": 0}
            chunk_dirs = list(chunks_dir.glob("chunk_*"))
            chunks_info["total_chunks"] = len(chunk_dirs)

            for chunk_dir in chunk_dirs:
                try:
                    chunk_data = self.process_text_content(chunk_dir)
                    if not chunk_data:
                        continue

                    # Generate embedding
                    embedding = self.model.encode(chunk_data.text_content,
                                               show_progress_bar=False,  # Disable progress bar
                                               convert_to_tensor=False,  # Return numpy array
                                               normalize_embeddings=True  # Normalize for cosine similarity
                                               ).tolist()

                    # Add or update in collection
                    self.collection.upsert(
                        ids=[chunk_data.chunk_id],
                        embeddings=[embedding],
                        documents=[chunk_data.text_content],
                        metadatas=[{
                            **chunk_data.metadata,
                            "files": json.dumps(chunk_data.files),
                            "created_at": chunk_data.created_at
                        }]
                    )

                    chunks_info["processed_chunks"] += 1
                    logger.info(f"Processed chunk: {chunk_data.chunk_id}")

                except Exception as e:
                    chunks_info["errors"] += 1
                    logger.error(f"Error processing chunk {chunk_dir}: {e}")

            return chunks_info

        except Exception as e:
            logger.error(f"Error processing chunks directory: {e}")
            raise

    def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """Search for similar chunks with threshold filtering"""
        try:
            # Encode query
            query_embedding = self.model.encode(query,
                                             show_progress_bar=False,
                                             convert_to_tensor=False,
                                             normalize_embeddings=True
                                             ).tolist()

            # Search with higher k to allow for filtering
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 20),
                include=["documents", "metadatas", "distances"]
            )

            # Process and filter results
            processed_results = []
            for idx in range(len(results['ids'][0])):
                similarity = 1 - results['distances'][0][idx]
                if similarity >= threshold:
                    processed_results.append({
                        'chunk_id': results['ids'][0][idx],
                        'content': results['documents'][0][idx],
                        'metadata': results['metadatas'][0][idx],
                        'similarity_score': similarity
                    })

            # Sort by similarity and limit to top_k
            processed_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return processed_results[:top_k]

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def get_total_chunks(self) -> int:
        """Get total number of chunks in store"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0

    def cleanup(self) -> bool:
        """Clean up vector store resources"""
        try:
            self.collection.delete()
            self.chroma_client.reset()
            return True
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False