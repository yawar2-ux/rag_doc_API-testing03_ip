#!/usr/bin/env python3
"""
CSV Chatbot service using ChromaDB and Groq for fleet data analysis.
Replaces the old Chatagent.py functionality.
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import groq
import io
import uuid
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVChatbot:
    def __init__(self, groq_api_key: str = None, chroma_db_path: str = "./fleet_chroma_db"):
        """Initialize the CSV Chatbot with ChromaDB and Groq"""
        
        # Initialize ChromaDB client
        self.chroma_db_path = Path(chroma_db_path)
        self.chroma_db_path.mkdir(exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        
        # Initialize Groq client
        self.groq_api_key = groq_api_key or "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"
        self.groq_client = groq.Groq(api_key=self.groq_api_key)
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Default collection name
        self.collection_name = "fleet_csv_data"
        
        logger.info("CSV Chatbot initialized successfully")
    
    def estimate_tokens(self, text: str) -> int:
        """Fast token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def truncate_context_fast(self, context_docs: List[str], max_chars: int = 32000) -> str:
        """Fast context truncation using character limits"""
        context = ""
        current_chars = 0
        
        for doc in context_docs:
            doc_length = len(doc)
            if current_chars + doc_length > max_chars:
                # Add partial document if there's significant space
                remaining_chars = max_chars - current_chars
                if remaining_chars > 500:  # Only add if meaningful space left
                    context += doc[:remaining_chars] + "...\n\n"
                break
            
            context += doc + "\n\n"
            current_chars += doc_length
        
        return context.strip()
    
    def parse_reference_data(self, context_docs: List[str]) -> List[Dict[str, Any]]:
        """Parse context documents into structured data for frontend tables"""
        structured_data = []
        
        for doc in context_docs:
            # Split document into rows
            rows = doc.split('\n')
            
            for row in rows:
                if not row.strip() or row.startswith('Row '):
                    continue
                    
                # Parse key-value pairs from each row
                row_data = {}
                pairs = row.split('; ')
                
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        row_data[key.strip()] = value.strip()
                
                if row_data:  # Only add if we found data
                    structured_data.append(row_data)
        
        return structured_data[:20]  # Limit to 20 rows for frontend performance
    
    def process_csv_data(self, df: pd.DataFrame, chunk_size: int = 200) -> List[Dict[str, Any]]:
        """Process CSV data efficiently into chunks for vector storage"""
        documents = []
        
        # Process in chunks for better performance with large CSVs
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            
            # Convert chunk to text
            chunk_text = ""
            for idx, row in chunk_df.iterrows():
                row_text = ""
                for col, val in row.items():
                    if pd.notna(val):
                        row_text += f"{col}: {val}; "
                chunk_text += f"Row {idx}: {row_text.strip()}\n"
            
            documents.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text.strip(),
                "metadata": {
                    "chunk_start": i,
                    "chunk_end": min(i + chunk_size, len(df)),
                    "source": "csv_upload"
                }
            })
        
        return documents
    
    def upload_csv_data(self, csv_content: bytes, filename: str = "fleet_data.csv") -> Dict[str, Any]:
        """Upload and process CSV file into ChromaDB (overwrites existing data)"""
        
        try:
            # Read CSV file
            df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
            
            logger.info(f"Processing CSV '{filename}' with {len(df)} rows and {len(df.columns)} columns")
            
            # Process CSV data into documents
            documents = self.process_csv_data(df)
            
            # Delete existing collection if it exists
            try:
                existing_collections = [col.name for col in self.chroma_client.list_collections()]
                if self.collection_name in existing_collections:
                    self.chroma_client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted existing collection '{self.collection_name}'")
            except Exception as e:
                logger.warning(f"Could not delete existing collection: {e}")
            
            # Create fresh collection
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Add documents to collection
            texts = [doc["text"] for doc in documents]
            ids = [doc["id"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully processed {len(documents)} documents into collection")
            
            return {
                "message": "CSV uploaded and processed successfully",
                "documents_processed": len(documents),
                "rows": len(df),
                "columns": len(df.columns),
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise Exception(f"Error processing CSV: {str(e)}")
    
    def query_csv_data(self, query: str) -> Dict[str, Any]:
        """Query the CSV data using natural language"""
        
        try:
            # Get the collection
            collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Query for relevant documents
            results = collection.query(
                query_texts=[query],
                n_results=5
            )
            
            if not results['documents'] or not results['documents'][0]:
                raise Exception("No relevant data found. Please upload a CSV first.")
            
            # Fast context truncation using character limits
            relevant_docs = results['documents'][0]
            context = self.truncate_context_fast(relevant_docs, max_chars=30000)
            
            # Parse reference data for frontend tabulation
            structured_reference = self.parse_reference_data(relevant_docs)
            
            # Generate response using Groq with fleet maintenance focus
            prompt = f"""You are a fleet maintenance and transport demand forecasting expert that helps analyzing and predicting vehicle maintenance needs and ridership patterns. Based on the following context from fleet data, answer the user's question accurately and helpfully.

Provide insights about:
- Vehicle maintenance patterns and predictions
- Component failure analysis
- Fleet utilization and efficiency
- Cost optimization recommendations
- Preventive maintenance scheduling

Context from Fleet Data:
{context}

User Question: {query}

Please provide a clear, accurate answer based on the fleet data provided. If you need to make calculations or summarize data, please do so. Focus on actionable insights for fleet management.

Answer:"""

            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = completion.choices[0].message.content
            
            return {
                "answer": answer,
                "reference_data": structured_reference,
                "raw_context": relevant_docs[:3],
                "query": query
            }
            
        except Exception as e:
            if "does not exist" in str(e).lower():
                raise Exception("No CSV data found. Please upload a CSV file first.")
            
            logger.error(f"Error querying data: {e}")
            raise Exception(f"Error querying data: {str(e)}")
    
    def has_data(self) -> bool:
        """Check if there's active CSV data"""
        try:
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            return self.collection_name in collection_names
        except Exception:
            return False
    
    def clear_data(self) -> Dict[str, str]:
        """Clear all CSV data"""
        try:
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            if self.collection_name in existing_collections:
                self.chroma_client.delete_collection(name=self.collection_name)
                return {"message": "CSV data cleared successfully"}
            else:
                return {"message": "No data to clear"}
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            raise Exception(f"Error clearing data: {str(e)}")

# Global chatbot instance
csv_chatbot = CSVChatbot()