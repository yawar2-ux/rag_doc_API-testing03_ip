import os
import shutil
from typing import Optional, List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# Create persistent directory for Chroma
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../chroma_db")
DOCUMENTS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../documents")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(DOCUMENTS_DIRECTORY, exist_ok=True)

# Embedding model
def get_embeddings():
    """Initialize embeddings with local model or remote model"""
    # You can replace with a remote model or local one based on your needs
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def process_document(file_path: str, collection_name: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Chroma:
    """Process document (PDF or TXT) and add to collection in vector database"""
    print(f"\n=== Processing document for collection: {collection_name} ===")

    # Determine file type and use appropriate loader
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide PDF or TXT file.")

    # Load and split the document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Loaded and split into {len(texts)} chunks")

    # Get embeddings
    embeddings = get_embeddings()

    # Create/update collection in vector database
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=collection_name
    )

    print(f"Updated collection '{collection_name}' with {len(texts)} documents")
    return db

def get_collection(collection_name: str) -> Optional[Chroma]:
    """Get collection if it exists"""
    embeddings = get_embeddings()

    try:
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        # Check if collection exists and has documents
        if db._collection.count() > 0:
            return db
    except Exception as e:
        print(f"Error accessing collection: {e}")

    return None

def clear_collection(collection_name: str) -> bool:
    """Clear all documents from a collection"""
    embeddings = get_embeddings()
    try:
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        db.delete_collection()

        # Also delete the collection directory if it exists
        collection_dir = os.path.join(PERSIST_DIRECTORY, collection_name)
        if os.path.exists(collection_dir):
            shutil.rmtree(collection_dir)

        print(f"Collection '{collection_name}' cleared successfully")
        return True
    except Exception as e:
        print(f"Error clearing collection: {e}")
        return False

def list_collections() -> List[str]:
    """List all available collections, filtering out langchain internal collections"""
    embeddings = get_embeddings()
    try:
        client = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        all_collections = client._client.list_collections()

        # Extract collection names
        user_collections = [c.name for c in all_collections
                            if not c.name.lower().startswith('langchain')
                            and 'langchain' not in c.name.lower()]

        return user_collections
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []

def clear_all_collections() -> bool:
    """Clear all collections and related document files"""
    try:
        # Get list of all non-langchain collections
        collections = list_collections()

        # Clear each individual collection
        for collection_name in collections:
            clear_collection(collection_name)

        # Remove all files from the documents directory
        if os.path.exists(DOCUMENTS_DIRECTORY):
            for filename in os.listdir(DOCUMENTS_DIRECTORY):
                file_path = os.path.join(DOCUMENTS_DIRECTORY, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        # Clear the audio outputs directory as well
        audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../audio_outputs")
        if os.path.exists(audio_dir):
            for filename in os.listdir(audio_dir):
                file_path = os.path.join(audio_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        print(f"Error deleting audio file {file_path}: {e}")

        print("All collections and associated documents cleared successfully")
        return True
    except Exception as e:
        print(f"Error in clear_all_collections: {e}")
        return False

def search_collection(collection_name: str, query: str, k: int = 3) -> List[Document]:
    """Search collection for documents similar to query"""
    db = get_collection(collection_name)
    if not db:
        return []

    results = db.similarity_search(query, k=k)
    return results
