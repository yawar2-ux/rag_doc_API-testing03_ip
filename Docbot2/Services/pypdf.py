from typing import List, Dict, Any, Tuple  # Added Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import os  # Added import for os.path.basename

def extract_pdf_content(file_path: str, document_metadata: Dict, raw_documents: Dict) -> Tuple[List[Document], int]:  # Corrected type hint
    """Extract content from PDF files using PyPDFLoader."""
    filename = os.path.basename(file_path)  # Use os.path.basename for robustness
    
    loader = PyPDFLoader(file_path)
    pdf_documents = loader.load()
    
    documents = []
    for doc in pdf_documents:
        page_num = doc.metadata.get('page', 0) + 1  # PyPDF is 0-indexed
        
        # Update metadata
        doc.metadata['page'] = page_num
        doc.metadata['source'] = filename
        doc.metadata['filename'] = filename
        
        documents.append(doc)
        
        # Store metadata and raw text
        key = f"{filename}_page_{page_num}"
        document_metadata[key] = {
            'filename': filename,
            'page_number': page_num
        }
        raw_documents[key] = doc.page_content
    
    return documents, len(pdf_documents)