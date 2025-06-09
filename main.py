import base64
import io
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple, Union



from bs4 import BeautifulSoup
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import (APIRouter, Depends, FastAPI, File, Form, Path, HTTPException,Query, UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from keycloak import KeycloakOpenID
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from wordcloud import WordCloud
from models.Auth.loginmodel import Login
from models.Auth.logoutmodel import Logout
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings
from PIL import Image
import pandas as pd
import requests
import urllib3
import validators
import fitz
import pickle
import logging

logger = logging.getLogger(__name__)

from Recommendation_system.recommendation import ChatAssistant
from Recommendation_system.ml.script import DynamicClassification
from FinBot.selfRag import EnhancedSelfRAG
from Multimodel.chat_interface import ChatResponse, DocumentChat
from Multimodel.image_search import ImageSearchService
from Multimodel.pdf_processor import process_multiple_pdfs
from Multimodel.vector_store import EnhancedVectorStore


from Automated_email_response.email import router as email_router

from tempfile import NamedTemporaryFile
# Credit Underwriting imports
#from credit_underwriting.utils import setup_logging
#from credit_underwriting.function import process_loan_application
#from credit_underwriting.api import router as credit_router
#from credit_underwriting.ocr import OCRProcessor
from FraudDetection.ML.function import AutoFraudDetection
from FraudDetection.GenAI.fraudcore import FraudDetectionAssistant

from pathlib import Path




#--------Financial_Statement-------------------------
from Financial_Statements_Analysis.image_agent import ImageAnalyzerAgent
from Financial_Statements_Analysis.csv_agent import CSVAnalyzerAgent
#---------------------------------------------------

#---credit 2.0----------
from Credit.api.api import credit_router

#----docbot2.0----------
from Docbot2.Api.api import docbot_router
#---------------------------------------------------Data Engineering------------------------

from DataEngineering.DataEngineeringAPI import router as dataengineering_router

#---------------------------------------------------
from DigitalMarketing.DigitalMarketingAPI import router as digitalmarketing_router

from Customer360.Customerapi import router as customer_router
from SentimentInvestmentAdvisor.Sentiment import router as sentiment_router
# from DocBot.docbot import router as docbot_router
from FraudDetection.Network.network import router as network_router
from TTS.tts import router as tts_router
from PromptFusion.Api import router as promptfusion_router

#---------------------------------------------------ComplianceAndSecurity------------------------
from ComplianceAndSecurity.complianceApi import router as compliance_router
#---------spec bot----------------
from spec_dot.api import router as specbot_router

#---------------------------------------------------Transportation Demand Forecasting Bot------------------------
from TransportationDemandForecasting.Api.api import TransportationDemandForecasting_router

# Fleet Management API import





from Fleet.Api.api import Fleet_router

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv(".env")
logging.basicConfig(level=logging.DEBUG)

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

vector_store = None
chat_interface = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles initialization and cleanup of services.
    """
    # Initialize services
    global vector_store, chat_interface
    try:
        vector_store = EnhancedVectorStore()
        chat_interface = DocumentChat(vector_store)
        logger.info("Successfully initialized services")
        yield
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise
    finally:
        # Cleanup services
        if vector_store:
            try:
                vector_store.cleanup()
                logger.info("Successfully cleaned up vector store")
            except Exception as e:
                logger.error(f"Error cleaning up vector store: {e}")
        logger.info("Application shutdown complete")

app = FastAPI(
    title="RAG Agent API",
    description="API for GenAI operations",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#recommendation
chat_assistant = ChatAssistant()

#finbot
static_directory = "static"

# Check if the directory exists, and create it if not
if not os.path.exists(static_directory):
    os.makedirs(static_directory)

# Mount the static files directory
app.mount("/static", StaticFiles(directory=static_directory), name="static")

#credit underwriting
#setup_logging()


#  fraud detection ML  #######################################
MODEL_PATH = "fraud_model.pkl"
# def load_model():
#     if os.path.exists(MODEL_PATH):
#         with open(MODEL_PATH, 'rb') as f:
#             return pickle.load(f)
#     return None

def save_model(model):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

# model = load_model()

class TransactionData(BaseModel):
    transaction_id: int
    amount: float
    merchant: str
    transaction_type: str
    location: str
    timestamp: str
    customer_age: int
    customer_risk_score: float
    device_type: str
    ip_location_match: int
    transaction_frequency: int

    class Config:
        schema_extra = {
            "example": {
                "transaction_id": 469,
                "amount": 50.36739,
                "merchant": "Apple",
                "transaction_type": "mobile",
                "location": "Miami",
                "timestamp": "1/2/2023 21:38",
                "customer_age": 73,
                "customer_risk_score": 68.0,
                "device_type": "tablet",
                "ip_location_match": 0,
                "transaction_frequency": 8
            }
        }

##########################################################fraud Genai

fraud_assistant = FraudDetectionAssistant()

class FileNameRequest(BaseModel):
    file_name: str

class ChatRequest(BaseModel):
    question: str

#####################################################################

self_rag = EnhancedSelfRAG()
ALLOWED_EXTENSIONS = {'pdf'}
generated_strategies_cache = {}
source_tracking = {
    'documents': {},
    'urls': {},
}

class URLInput(BaseModel):
    url: str
    sector: str = "general"

class StrategyInput(BaseModel):
    sector: str
    parameters: Dict[str, Any] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "sector": "credit_card",
                "parameters": {
                    "credit_score": 750,
                    "annual_income": 75000,
                    "monthly_spending": 3000,
                    "preferred_rewards": "travel",  # Options: travel, cashback, points, business
                    "card_type": "rewards"  # Options: rewards, business, secured, student
                }
            }
        }



class MoreInfoInput(BaseModel):
    strategy: Dict[str, Any]
    urls: List[str]

class SourceInput(BaseModel):
    source: str

#finbot

class KeycloakConfig(BaseSettings):
    server_url: str = os.getenv('KEYCLOAK_SERVER_URL')
    client_id: str = os.getenv('KEYCLOAK_CLIENT_ID')
    realm_name: str = os.getenv('KEYCLOAK_REALM_NAME')
    client_secret_key: str = os.getenv('KEYCLOAK_CLIENT_SECRET_KEY')


config = KeycloakConfig()
keycloak_openid = KeycloakOpenID(
    server_url=config.server_url,
    client_id=config.client_id,
    realm_name=config.realm_name,
    client_secret_key=config.client_secret_key,
    verify=False
)


DOCUMENT_DIR = "docs"
Path(DOCUMENT_DIR).mkdir(parents=True, exist_ok=True)

# Directories and Globals for Doc-Bot
IMAGEDIR1 = "DOC"
os.makedirs(IMAGEDIR1, exist_ok=True)
os.makedirs("./data", exist_ok=True)
client = Client(Settings(persist_directory="./data", is_persistent=True))
# rag_Client = Client(Settings(persist_directory="./db", is_persistent=True))
model_path = Path(os.path.join(base_path, "sentence_transformer"))
embedding_model = SentenceTransformer(str(model_path))
ollama_host = os.getenv("OLLAMA_BASE_URL")
OLLAMA_EMBEDDING_URL = f"{ollama_host}/api/embeddings"
OLLAMA_GENERATE_URL = f"{ollama_host}/api/generate"

global rag_agent
rag_agent = None
IMAGEDIR = "images/brightness"
os.makedirs(IMAGEDIR, exist_ok=True)

def is_string_an_url(url_string: str) -> bool:
    result = validators.url(url_string)
    print(result)
    return result


router = APIRouter(prefix="/rag_doc")

@router.get("/")
async def read_root():
    message = f"Hello world! From for Rag and docBot Agent."
    return {"message": message}

# ------------------------------------AUTHENTICATION API's -----------------------------------------------

@router.post("/login")
async def login(login: Login):
    try:
        token = keycloak_openid.token(login.username, login.password)
        return JSONResponse(content=token)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")

@router.post("/logout")
async def logout(logout: Logout):
    try:
        logout = keycloak_openid.logout(logout.refresh_token)
        return JSONResponse(content={"message": "User logged out successfully."}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Logout failed: {str(e)}")

def reset_user_collection(client: Client, username: str):
    """Delete existing collection for the user if it exists and create a new one."""
    collection_name = f"{username}_collection"
    collections = client.list_collections()
    collection_names = [collection.name for collection in collections]

    if collection_name in collection_names:
        print(f"Collection '{collection_name}' exists. Deleting it.")
        client.delete_collection(name=collection_name)

    return client.create_collection(name=collection_name)

async def get_ollama_embedding(text: str) -> List[float] | None:
     return embedding_model.encode(text).tolist()
    # """Fetch text embedding from the Ollama API."""
    # try:
    #     response = requests.post(
    #         OLLAMA_EMBEDDING_URL,
    #         json={"model": "llama3.1", "prompt": text},
    #         verify=False
    #     )
    #     response.raise_for_status()
    #     return response.json().get("embedding")
    # except requests.RequestException as e:
    #     print(f"Error fetching embedding: {e}")
    #     return None

async def generate_ollama_response(docs: str, prompt: str) -> str | None:
    """Generate a response from the Ollama API."""

    restricted_topics = {
        "harmful", "impersonate", "forget rules", "explicit", "abuse",
        "sensitive", "personal information", "execute code", "system prompt",
        "garbled", "stupid", "idiot", "dumb", "useless", "fool", "moron"
    }

    prompt_lower = prompt.lower()

    # If the prompt contains any restricted word, return a predefined message
    if any(word in prompt_lower for word in restricted_topics):
        return "Please, Enter a valid Question."

    try:
        llm_prompt = (
            f"Using this data: {docs} as a reference.\n"
            f"Based on this data, answer this question: {prompt}.\n"
            f"Make sure the answer is relevant to the data."
        )

        response = requests.post(
            OLLAMA_GENERATE_URL,  # Ensure this is correctly set in the environment
            json={"model": os.getenv("OLLAMA_TEXT_MODEL"), "stream": False, "prompt": llm_prompt},
            verify=False
        )
        response.raise_for_status()
        return response.json().get("response")

    except requests.RequestException as e:
        print(f"Error generating response: {e}")
        return None

async def generate_ollama_response_doc(prompt: str ) -> str | None:
    """Generate a response from the Ollama API."""
    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": "llama3.2:latest", "stream": False, "prompt": prompt},
            verify=False
        )
        response.raise_for_status()
        return response.json().get("response")
    except requests.RequestException as e:
        print(f"Error generating response: {e}")
        return None



def construct_prompt(data: str, question: str) -> str:
    """Construct a prompt for Ollama based on the retrieved data and question."""
    prompt = f"""Using this data: {data} as a reference.
            Based upon this data, answer this question: {question}.
            Make sure the answer is relevant to the data and do not answer with your own knowledge base.
                """
    return prompt

def reset_user_data_collection(client: Client, username: str):
    collection_name = f"{username}_data_collection"
    collections = client.list_collections()
    if collection_name in [col.name for col in collections]:
        client.delete_collection(name=collection_name)
    return client.create_collection(name=collection_name)

async def is_query_relevant(query: str, collection, threshold: float = 0.7) -> Tuple[bool, List[str]]:
    """Check if a query is relevant to the stored collection."""
    embedding = await get_ollama_embedding(query)
    results = collection.query(query_embeddings=[embedding], n_results=3)
    distances = results["distances"]
    if isinstance(distances[0], list):
        distances = [score for sublist in distances for score in sublist]

    relevant_docs = [
        doc for doc, score in zip(results["documents"], distances) if score <= threshold
    ]
    relevant_docs = [str(doc) for doc in relevant_docs]  # Convert each item to string

    if relevant_docs:
        return True, relevant_docs
    return False, []

async def handle_ambiguous_query(query: str, collection, threshold: float = 1.3) -> dict:
    """Retrieve results from both the collection and web search for ambiguous queries."""
    embedding = await get_ollama_embedding(query)
    results = collection.query(query_embeddings=[embedding], n_results=3)

    distances = results["distances"]

    if isinstance(distances[0], list):
        distances = [score for sublist in distances for score in sublist]

    relevant_docs = [
        doc for doc, score in zip(results["documents"], distances) if score <= threshold
    ]

    relevant_docs = [str(doc) for doc in relevant_docs]

    web_results = perform_web_search(query)

    return {
        "documents": relevant_docs if relevant_docs else [],
        "web_results": web_results
    }



def perform_web_search(query: str) -> str:
    """Perform a web search and return top results in a structured format."""
    search_url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for result in soup.find_all('div', class_='result__body', limit=5):
            title = result.find('a', class_='result__a').text.strip()
            snippet = result.find('a', class_='result__snippet').text.strip()
            results.append(f"{title}: {snippet}")

        return "\n\n".join(results) if results else "No relevant information found."
    except requests.RequestException as e:
        print(f"Error during web search: {e}")
        return "Web search is unavailable."
async def generate_combined_response(prompt: str, docs: List[str], web_results: Union[str, List[str]]) -> str:
    """Combine documents and web search results for a response."""
    if isinstance(web_results, list):
        web_results = "\n".join(web_results)  # Convert list to a single string

    combined_context = "\n".join(docs) + "\n\nWeb Results:\n" + web_results

    restricted_topics = {
        "harmful", "impersonate", "forget rules", "explicit", "abuse",
        "sensitive", "personal information", "execute code", "system prompt",
        "garbled", "stupid", "idiot", "dumb", "useless", "fool", "moron"
    }

    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in restricted_topics):
        return "Please, Enter a valid Question."

    try:
        llm_prompt = (
            f"Using this data: {combined_context} as a reference.\n"
            f"Based on this data, answer this question: {prompt}.\n"
            f"Make sure the answer is relevant to the data."
        )

        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": os.getenv("OLLAMA_TEXT_MODEL"), "stream": False, "prompt": llm_prompt},
            verify=False
        )
        response.raise_for_status()
        return response.json().get("response")

    except requests.RequestException as e:
        print(f"Error generating response: {e}")
        return None


async def process_and_extract_texts_from_urls(urls: List[str]) -> List[str]:
    """
    Fetch content from URLs, extract text, split into chunks, and return the chunks.

    Args:
        urls (List[str]): A list of URLs to fetch and process.

    Returns:
        List[str]: A list of text chunks extracted from the URLs.
    """
    docs_list = []
    for url in urls:
        try:
            documents = WebBaseLoader(url).load()
            docs_list.extend(documents)
        except Exception as e:
            print(f"Error loading URL {url}: {e}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs_list)

    return [doc.page_content for doc in split_docs]

async def process_and_extract_texts_from_files(files: List[UploadFile]) -> List[str]:
    """Process files and extract text chunks."""
    docs_list = []
    loader_mapping = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
    }

    for uploaded_file in files:
        try:
            file_ext = uploaded_file.filename.split(".")[-1]
            timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            file_name = f"{uploaded_file.filename.split('.')[0]}_{timestamp}.{file_ext}"
            file_path = os.path.join(DOCUMENT_DIR, file_name)

            contents = await uploaded_file.read()
            with open(file_path, "wb") as f:
                f.write(contents)

            loader_class = loader_mapping.get(f".{file_ext.lower()}")
            if loader_class:
                documents = loader_class(file_path).load()
                docs_list.extend(documents)
            else:
                print(f"Unsupported file type: {uploaded_file.filename}")
        except Exception as e:
            print(f"Error processing file {uploaded_file.filename}: {e}")
            continue

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs_list)
    return [doc.page_content for doc in split_docs]


@router.post("/upload_url_doc/")
async def upload_data(
    username: str,
    files: List[UploadFile] = None,
    urls: List[str] = None
):
    """Upload files/URLs and store in ChromaDB collection."""
    data_collection = reset_user_data_collection(client, username)
    if files:
        doc_texts = await process_and_extract_texts_from_files(files)
        if doc_texts:
            doc_embeddings = await asyncio.gather(*[get_ollama_embedding(text) for text in doc_texts])
            doc_embeddings = [emb for emb in doc_embeddings if emb]
            if doc_embeddings:
                data_collection.add(
                    ids=[f"doc_{i}" for i in range(len(doc_embeddings))],
                    embeddings=doc_embeddings,
                    documents=doc_texts,
                )

    if urls:
        url_texts = await process_and_extract_texts_from_urls(urls)
        if url_texts:
            url_embeddings = await asyncio.gather(*[get_ollama_embedding(text) for text in url_texts])
            url_embeddings = [emb for emb in url_embeddings if emb]
            if url_embeddings:
                data_collection.add(
                    ids=[f"url_{i}" for i in range(len(url_embeddings))],
                    embeddings=url_embeddings,
                    documents=url_texts,
                )

    return {"message": "Data processed and stored successfully"}

@router.post("/get_prompts")
async def query_rag_system(prompt: str, username: str):
    """Handle user queries using Corrective RAG."""
    try:
        collection_name = f"{username}_data_collection"
        collections = client.list_collections()

        if collection_name not in {col.name for col in collections}:  # Use set for efficiency
            raise HTTPException(status_code=404, detail="User's collection not found.")

        collection = client.get_collection(name=collection_name)

        relevant, docs = await is_query_relevant(prompt, collection)

        if relevant:
            response = await generate_ollama_response("\n".join(docs), prompt)
            return {"response": response}

        # Handle ambiguous queries
        ambiguous_results = await handle_ambiguous_query(prompt, collection)

        if ambiguous_results:
            docs = ambiguous_results.get("documents", [])
            web_results = ambiguous_results.get("web_results", [])

            response = await generate_combined_response(prompt, docs, web_results)
            return {"response": response}

        return {"response": "I'm not sure how to answer this. Can you rephrase your question?"}

    except HTTPException as http_err:
        raise http_err  # Preserve FastAPI HTTP exceptions

    except Exception as e:
        print(f"Error in query_rag_system: {e}")  # Log the error
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@router.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/rag_doc/openapi.json", title="RAG Agent API Docs"
    )


@router.get("/openapi.json", include_in_schema=False)
async def custom_openapi():
    return app.openapi()

#----------------------------------------------------------------Finbot----------------------------------------------------------------

@router.post("/upload_documents")
async def upload_documents(documents: List[UploadFile] = File(...)):
    try:
        uploaded_files = []

        # Iterate over each file and validate PDF type
        for document in documents:
            if not document.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"Invalid file format: {document.filename}. Please upload PDF files only.")

            document_id = str(uuid.uuid4())  # Generate a unique ID for each document
            filename = f"{document_id}_{document.filename}"
            permanent_filepath = os.path.join(self_rag.pdf_dir, filename)

            # Save the file
            os.makedirs(self_rag.pdf_dir, exist_ok=True)
            with open(permanent_filepath, "wb") as buffer:
                shutil.copyfileobj(document.file, buffer)

            uploaded_files.append({
                "document_id": document_id,
                "filename": document.filename,
                "is_indexed": False
            })

        return {
            "status": "success",
            "message": f"{len(uploaded_files)} document(s) successfully uploaded",
            "uploaded_files": uploaded_files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/process_url")
async def process_url(
    urls: str = Form(...)  # Accept comma-separated URLs
    # sector: str = Form(...)
):
    """
    Process multiple URLs submitted as a form.
    Handles both repeated fields and comma-separated strings.
    """
    try:
        # Split URLs by comma and strip spaces
        url_list = [url.strip() for url in urls.split(",") if url.strip()]

        # Check if URLs are provided
        if not url_list:
            raise HTTPException(status_code=400, detail="No valid URLs provided")

        # Build details list
        details = [{"url": url} for url in url_list]

        return {
            "status": "success",
            "message": f"{len(url_list)} URL(s) successfully processed",
            "details": details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_strategies")
async def generate_strategies(input_data: StrategyInput):
    try:
        sector = input_data.sector
        parameters = input_data.parameters or {}

        if sector not in generated_strategies_cache:
            generated_strategies_cache[sector] = set()

        # Store and validate parameters first
        validated_params = self_rag.store_sector_parameters(sector, input_data.parameters)



        result = self_rag.query(
            question=f"Generate strategies for {sector} sector",
            sector=sector
        )


        # Rest of your existing endpoint code...

        return {
            "strategies": result.get('strategies', []),
            "metadata": {
                "sector": sector,
                "parameters_used": validated_params,
                "total_generated": len(generated_strategies_cache[sector])
            }
        }

    except Exception as e:
        logging.error(f"Strategy generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view_pdf_page")
async def view_pdf_page(path: str, page: int = 1, dpi: int = 200):
    try:
        # Extract both filename and page reference from path
        parts = path.split(':')
        pdf_filename = parts[0].strip()

        # Handle UUID prefix in filename
        pdf_files = [f for f in os.listdir(self_rag.pdf_dir) if f.lower().endswith('.pdf')]
        target_pdf = next((pdf for pdf in pdf_files if pdf_filename.lower() in pdf.lower()), None)

        if not target_pdf:
            # Search by UUID if direct filename match fails
            target_pdf = next((pdf for pdf in pdf_files if pdf.startswith(pdf_filename)), None)

        if not target_pdf:
            raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_filename}")

        # Load the PDF file
        pdf_path = os.path.join(self_rag.pdf_dir, target_pdf)
        doc = fitz.open(pdf_path)

        # Check if the page exists
        if page < 1 or page > doc.page_count:
            raise HTTPException(status_code=400, detail=f"Page {page} is out of range for {target_pdf}.")

        # Get the specified page
        pdf_page = doc.load_page(page - 1)

        # Render the page to a Pixmap
        zoom = dpi / 72  # 72 DPI is the default resolution in PDFs
        mat = fitz.Matrix(zoom, zoom)  # Scale by zoom factor
        pix = pdf_page.get_pixmap(matrix=mat, alpha=False)

        # Convert Pixmap to PIL Image for format conversion
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert to JPEG format in memory
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG", quality=85)
        img_byte_arr.seek(0)

        # Convert to base64
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Return JSON response with base64 string
        return {
            "filename": f"{pdf_filename}_page_{page}.jpg",
            "content_type": "image/jpeg",
            "base64_content": base64_image
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get_more_info")
async def get_more_info(input_data: MoreInfoInput):
    try:
        # Assuming `self_rag.get_more_info` is your method that returns data
        result = self_rag.get_more_info(
            strategy=input_data.strategy,
            urls=input_data.urls
        )

        # Sanitize the result to clean NaN and infinite values
        clean_result = sanitize_response(result)

        # Return the sanitized response
        return clean_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import numpy as np

# Function to clean the response data
def sanitize_response(data):
    if isinstance(data, dict):
        return {k: sanitize_response(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_response(item) for item in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None  # Replace with a default value (e.g., None)
    return data



# --------------------------------------------------MultiModel-------------------------------------------------------------------------------------------

# Create directories for file storage
UPLOAD_DIR = Path("temp_uploads")
OUTPUT_DIR = Path("pdf_chunks")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)




class ChatRequest(BaseModel):
    """Chat request model"""
    message: str

def format_table_data(table_content: str) -> str:
    """Format table data with pipe operators"""
    try:
        lines = table_content.strip().split('\n')
        formatted_rows = []

        for line in lines:
            if '|' in line:
                formatted_rows.append(line.strip())
            elif ':' in line:
                key, value = line.split(':', 1)
                formatted_rows.append(f"{key.strip()} | {value.strip()}")

        return '\n'.join(formatted_rows)
    except Exception as e:
        return table_content

@router.post("/multimodel/process-documents/")
async def process_documents(files: List[UploadFile]):
    """Process uploaded documents and store in vector database"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved_files = []
    try:
        # Save uploaded files
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")

            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(str(file_path))
            logger.info(f"Saved file: {file.filename}")

        # Process PDFs and generate chunks
        processing_result = await process_multiple_pdfs(saved_files, OUTPUT_DIR)

        # Add chunks to vector store
        vector_store_result = vector_store.add_chunks_to_db(OUTPUT_DIR)

        return JSONResponse(content={
            "message": "Documents processed successfully",
            "processing_details": processing_result,
            "vector_store_details": vector_store_result
        })

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup uploaded files
        for file_path in saved_files:
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {str(e)}")

@router.post("/multimodel/chat/")
async def chat(request: ChatRequest):
    """Chat endpoint for document-based conversations"""
    if not chat_interface:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    try:
        response = chat_interface.chat(request.message)

        if any(indicator in response.bot_response.lower() for indicator in
               ['| value', '| data', 'statistic |', 'column:', 'columns:']):
            response.bot_response = format_table_data(response.bot_response)

        return response

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multimodel/image-search/")
async def image_search(image: UploadFile = File(...)):
    """Search for matching faces and generate contextual response"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await image.read()
        image_service = ImageSearchService(vector_store=vector_store)
        result = image_service.search_and_respond(image_data)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing image search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/multimodel/chunks/{chunk_id}/{file_type}")
async def get_chunk_file(chunk_id: int, file_type: str):
    """Retrieve a specific file from a chunk"""
    chunk_dir = OUTPUT_DIR / f"chunk_{chunk_id}"

    if not chunk_dir.exists():
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")

    file_mapping = {
        "image": "image.png",
        "text": "text.txt",
        "image_description": "image_description.txt"
    }

    if file_type.startswith("table"):
        try:
            table_num = int(file_type.split("_")[1])
            file_path = chunk_dir / f'table_{table_num}.csv'

            if file_path.exists():
                df = pd.read_csv(file_path)
                formatted_table = df.to_string(index=False).replace('  ', ' | ')
                return JSONResponse(content={"table_data": formatted_table})

        except:
            raise HTTPException(status_code=400, detail="Invalid table number")
    else:
        if file_type not in file_mapping:
            raise HTTPException(status_code=400, detail="Invalid file type")
        file_path = chunk_dir / file_mapping[file_type]

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {file_type} not found in chunk {chunk_id}")

    return FileResponse(file_path)

@router.get("/multimodel/chunks/list")
async def list_chunks():
    """List all available chunks and their contents"""
    chunks = []
    for chunk_dir in sorted(OUTPUT_DIR.glob("chunk_*")):
        chunk_num = int(chunk_dir.name.split("_")[1])
        contents = {
            "chunk_id": chunk_num,
            "files": [f.name for f in chunk_dir.iterdir()]
        }
        chunks.append(contents)
    return chunks

@router.get("/health")
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store is not None,
        "chat_interface_initialized": chat_interface is not None,
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "output_dir_exists": OUTPUT_DIR.exists(),
        "upload_dir_path": str(UPLOAD_DIR.absolute()),
        "output_dir_path": str(OUTPUT_DIR.absolute())
    }

#-------------------------------------------recommendation api's-------------------------------------------
class VisualizationRequest(BaseModel):
    x_axis: str
    y_axis: Optional[str] = None

class ChatRequest(BaseModel):
    question: str

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a dataset file
    Returns preview and dataset information
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    if not chat_assistant.validate_file_size(len(content)):
        raise HTTPException(status_code=400, detail="File size exceeds maximum limit")

    success, message = chat_assistant.process_file(file.filename, content)
    if not success:
        raise HTTPException(status_code=400, detail=message)

    df = chat_assistant.uploaded_files_data[file.filename]

    return {
        "preview": df.head().to_dict(),
        "dataset_info": chat_assistant.get_dataset_info(df),
        "message": "File processed successfully"
    }

@router.post("/visualize")
async def create_visualization(request: VisualizationRequest):
    """
    Generate visualizations based on specified columns
    Returns base64 encoded distribution and scatter plots
    """
    if not chat_assistant.current_dataset:
        raise HTTPException(status_code=400, detail="No dataset loaded")

    df = chat_assistant.uploaded_files_data[chat_assistant.current_dataset]

    try:
        plots = chat_assistant.generate_visualizations(df, request.x_axis, request.y_axis)
        return plots
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from pydantic import BaseModel

class FileNameRequest(BaseModel):
    file_name: str

@router.post("/profile-report")
async def generate_profile(request: FileNameRequest):
    """
    Generate YData Profiling report for the dataset using a file name
    Returns HTML report
    """
    file_name = request.file_name

    # Check if the file has been uploaded and processed
    if file_name not in chat_assistant.uploaded_files_data:
        raise HTTPException(status_code=400, detail="File not found or not processed")

    # Retrieve the DataFrame for the specified file
    df = chat_assistant.uploaded_files_data[file_name]

    try:
        # Generate the profile report
        profile_html = chat_assistant.generate_profile_report(df)
        return {
            "profile_report": profile_html
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_with_assistant(request: ChatRequest):
    """
    Process questions about the loaded dataset
    Returns AI-generated response
    """
    success, response = chat_assistant.process_question(request.question)

    if not success:
        raise HTTPException(status_code=400, detail=response)

    return {
        "answer": response
    }

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return HTTPException(
            status_code=500,
            detail=str(e)
        )


# -------------------------------------------------------Credit underwriting --------------------------------

#@router.post("/process-loan-applications")
#async def process_loan_applications(file: UploadFile = File(...)):
#    if not file.filename.endswith('.csv'):
#        raise HTTPException(400, "Only CSV files are accepted")
#
#    try:
#        # Create the 'shaukat' directory if it doesn't exist
#        shaukat_dir = os.path.join(os.getcwd(), "shaukat")
#        os.makedirs(shaukat_dir, exist_ok=True)
#
#        # Save the file in the 'shaukat' directory
#        file_path = os.path.join(shaukat_dir, file.filename)
#        with open(file_path, "wb") as f:
#            shutil.copyfileobj(file.file, f)
#
#        return StreamingResponse(
#            process_loan_application(file_path),
#            media_type='text/event-stream'
#        )
#
#    finally:
#        file.file.close()
#
## Initialize OCR processor
#ocr_processor = OCRProcessor()
#
## Move these endpoints inside the router section
#@router.post("/process-to-csv")
#async def process_to_csv(files: List[UploadFile] = File(...)):
#    temp_paths = []
#    try:
#        # Validate file types
#        for file in files:
#            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
#                raise HTTPException(
#                    status_code=400,
#                    detail=f"Invalid file format: {file.filename}. Only image files are supported."
#                )
#
#        # Process files
#        base64_images = []
#        for file in files:
#            temp_path = os.path.join(ocr_processor.temp_dir, file.filename)
#            temp_paths.append(temp_path)
#
#            with open(temp_path, "wb") as buffer:
#                shutil.copyfileobj(file.file, buffer)
#
#            with open(temp_path, "rb") as image_file:
#                base64_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
#
#        # Process images and get CSV file
#        csv_file, _ = ocr_processor.process_images(base64_images)
#
#        if not os.path.exists(csv_file):
#            raise HTTPException(status_code=500, detail="Failed to generate CSV file")
#
#        # Process the CSV file through loan application
#        return StreamingResponse(
#            process_loan_application(csv_file),
#            media_type='text/event-stream'
#        )
#
#    except Exception as e:
#        logger.error(f"Error in process_to_csv: {str(e)}")
#        raise HTTPException(status_code=500, detail=str(e))
#
#    finally:
#        # Cleanup temporary files
#        for temp_path in temp_paths:
#            try:
#                if os.path.exists(temp_path):
#                    os.remove(temp_path)
#            except Exception as e:
#                logger.error(f"Error removing temporary file {temp_path}: {str(e)}")
#

#--------------------------------- financial statements analysis ------------------------------------------
image_agent = ImageAnalyzerAgent()

@router.post("/analyze-image")
async def analyze_image(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        if not image.content_type in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type")

        image_bytes = await image.read()
        analysis = await image_agent.analyze_image(image_bytes, prompt)
        return {"analysis": analysis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

csv_analyzer = CSVAnalyzerAgent()



@router.post("/analyze-csv")
async def analyze_csv(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file type. Must be CSV")

        # Save uploaded file temporarily
        # Create directories for CSV analysis
        CSV_DIR = "temp_csv"
        os.makedirs(CSV_DIR, exist_ok=True)

        # Create unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(CSV_DIR, unique_filename)

        try:
            # Save the file
            content = await file.read()
            with open(temp_path, "wb") as buffer:
                buffer.write(content)

            # Analyze the CSV
            analysis = await csv_analyzer.analyze_csv(temp_path, prompt)

            return {"analysis": analysis}

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#-----------------------------------Fraud Detection ML APIS -----------------------------------------------------------

fraudModel = None

class TrainingMetricsML(BaseModel):
    accuracy: float
    f1_score: float
    confusion_matrix: List[List[int]]
    message: str
    target_column: str

class ManualPredictionInputML(BaseModel):
    data: Dict[str, Any]


@router.post("/fraudDetection/ml/train", response_model=TrainingMetricsML)
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    global fraudModel
    temp_file = f"temp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        # Save uploaded file
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Read CSV to validate target column
        df = pd.read_csv(temp_file)

        # Validate target column exists
        if target_column not in df.columns:
            os.remove(temp_file)
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")

        # Initialize and train model
        fraudModel = AutoFraudDetection()
        fraudModel.dataset_path = temp_file
        fraudModel.target_column = target_column
        metrics = fraudModel.train_model()

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return TrainingMetricsML(
            accuracy=metrics['accuracy'],
            f1_score=metrics['f1_score'],
            confusion_matrix=metrics['confusion_matrix'],
            message="Model trained successfully",
            target_column=target_column
        )

    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/fraudDetection/ml/predict/manual")
async def manual_predict(input_data: ManualPredictionInputML):
    if fraudModel is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train model first.")

    try:
        validated_data = {}
        for column, value in input_data.data.items():
            if column not in fraudModel.original_columns:
                raise HTTPException(status_code=400, detail=f"Unknown column: {column}")

            try:
                if column in fraudModel.datetime_columns:
                    pd.to_datetime(value)
                    validated_data[column] = value
                elif fraudModel.column_types.get(column) in ['int64', 'int32']:
                    validated_data[column] = int(value)
                elif fraudModel.column_types.get(column) in ['float64', 'float32']:
                    validated_data[column] = float(value)
                else:
                    validated_data[column] = value

                if column in fraudModel.one_hot_columns and value not in fraudModel.one_hot_columns[column]:
                    raise HTTPException(status_code=400, detail=f"Invalid value for {column}")

            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid value for {column}")

        missing_cols = set(fraudModel.original_columns) - set(validated_data.keys())
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        result = fraudModel.predict_single(validated_data)
        if result is None:
            raise HTTPException(status_code=500, detail="Prediction failed")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fraudDetection/ml/predict/bulk")
async def bulk_predict(file: UploadFile = File(...)):
    if fraudModel is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train model first.")

    temp_file = f"temp_predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    predictions_file = temp_file.replace('.csv', '_predictions.csv')

    try:
        # Save uploaded file
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Generate predictions
        fraudModel.predict_bulk(temp_file)

        # Verify predictions file exists
        if not os.path.exists(predictions_file):
            raise HTTPException(status_code=500, detail="Failed to generate predictions file")

        # Create response object
        response = FileResponse(
            path=predictions_file,
            media_type="text/csv",
            filename="predictions.csv"
        )

        # Clean up input file only
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Return response with predictions file
        return response

    except Exception as e:
        # Clean up both files on error
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(predictions_file):
            os.remove(predictions_file)
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/fraudDetection/ml/model/status")
async def model_status():
    return {
        "model_loaded": fraudModel is not None,
        "model_path": MODEL_PATH if fraudModel is not None else None
    }


@router.get("/fraudDetection/ml/model/required_columns")
async def get_required_columns():
    if fraudModel is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train model first.")

    try:
        return {
            "required_columns": fraudModel.original_columns,
            "column_types": fraudModel.column_types,
            "datetime_columns": fraudModel.datetime_columns,
            "categorical_columns": list(fraudModel.one_hot_columns.keys()) if fraudModel.one_hot_columns else []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@router.post("/fraudUpload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a fraud detection dataset
    Returns preview and dataset information
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Check file size
    file_size = 0
    contents = await file.read()
    file_size = len(contents)

    if not fraud_assistant.validate_file_size(file_size):
        raise HTTPException(status_code=400, detail="File size exceeds maximum limit of 50MB")

    success, message = fraud_assistant.process_file(file.filename, contents)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    df = fraud_assistant.uploaded_files_data[file.filename]

    # Get preview and basic statistics
    preview = df.head().to_dict()
    stats = {
        "total_transactions": len(df),
        "fraudulent_transactions": int(df['is_fraudulent'].sum()),
        "fraud_percentage": float((df['is_fraudulent'].sum() / len(df)) * 100),
        "total_amount": float(df['amount'].sum()),
        "average_transaction": float(df['amount'].mean()),
        "unique_merchants": int(df['merchant'].nunique())
    }

    return {
        "success": True,
        "preview": preview,
        "statistics": stats,
        "message": message
    }

@router.post("/fraudVisualize")
async def create_visualization(file: UploadFile = File(...)):
    """
    Generate fraud detection visualizations from uploaded file
    Returns two base64 encoded images:
    1. Transaction Amount Distribution by Fraud Status
    2. Top 10 Merchants by Fraud Count
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Process the uploaded file
    contents = await file.read()
    success, message = fraud_assistant.process_file(file.filename, contents)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    df = fraud_assistant.uploaded_files_data[file.filename]

    try:
        # Generate visualizations using the fraud assistant's method
        # Note: The x_axis and y_axis parameters will be ignored as we're using predefined visualizations
        visualizations = fraud_assistant.generate_visualizations(df, x_axis='amount', y_axis='is_fraudulent')

        return {
            "success": True,
            "message": "Visualizations generated successfully",
            "visualizations": {
                "amount_distribution": visualizations['amount_distribution'],  # Base64 encoded image
                "top_merchants": visualizations['top_merchants']  # Base64 encoded image
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")

@router.post("/fraudProfileReport")
async def generate_profile(request: FileNameRequest):
    """
    Generate YData Profiling report for the dataset using a file name
    Returns HTML report
    """
    file_name = request.file_name

    # Check if the file has been uploaded and processed
    if file_name not in fraud_assistant.uploaded_files_data:
        raise HTTPException(status_code=400, detail="File not found or not processed")

    # Retrieve the DataFrame for the specified file
    df = fraud_assistant.uploaded_files_data[file_name]

    try:
        # Generate the profile report
        profile = fraud_assistant.generate_profile_report(df)

        return {
            "success": True,
            "profile_report": profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fraudChat")
async def chat_with_assistant(request: ChatRequest):
    """
    Process questions about fraud detection patterns
    """
    try:
        success, response = fraud_assistant.process_question(request.question)

        if not success:
            raise HTTPException(status_code=400, detail=response)

        return {
            "success": True,
            "answer": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#-----------------------------Call Agent APIS -------------------------------------------------------------
from CallAgent.streamlit import CallCenterAssistant
from CallAgent.db import get_db_connection, CallCenterDB, validate_mobile_no, generate_cust_id
from CallAgent.pdf import PDFGenerator

import speech_recognition as sr


assistant = CallCenterAssistant()
sessions = {}
db = get_db_connection()

class CallSession:
    def __init__(self, assistant: CallCenterAssistant, db: CallCenterDB, mobile_no: str):
        self.assistant = assistant
        self.db = db
        self.transcript = []  # Stores structured conversation history
        self.sentiment_scores = []
        self.wordcloud_data = {}
        self.start_time = datetime.now()
        self.cust_id = generate_cust_id()
        self.agent_id = db.select_available_agent()
        self.mobile_no = mobile_no
        self.previous_summary = db.get_call_history(mobile_no)
        self.sales_opportunities = []

    def update_wordcloud(self, customer_text):
        words = customer_text.split()
        for word in words:
            self.wordcloud_data[word] = self.wordcloud_data.get(word, 0) + 1

    def compute_average_sentiment(self):
        return sum(self.sentiment_scores) / len(self.sentiment_scores) if self.sentiment_scores else 50.0

    def save_wordcloud(self):
        if not self.wordcloud_data:
            return None
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(self.wordcloud_data)
        output = io.BytesIO()
        wordcloud.to_image().save(output, format="PNG")
        base64_image = base64.b64encode(output.getvalue()).decode()
        return base64_image

    def save_call_record(self, summary):
        final_sentiment = self.compute_average_sentiment()
        self.db.save_call_record(
            start_time=self.start_time,
            cust_id=self.cust_id,
            agent_id=self.agent_id,
            mobile_no=self.mobile_no,
            call_transcript="\n".join([entry['message'] for entry in self.transcript]),
            call_summary=summary,
            avg_sentiment=final_sentiment
        )

    def generate_call_summary(self):
        structured_conversation = "".join(
            f"{entry['type'].title()}: {entry['message']}\n" for entry in self.transcript
        )
        prompt = (
            "You are an AI assistant summarizing customer service calls. Summarize the following conversation into key points,"
            " highlighting customer issues, concerns, resolutions provided, and important details discussed. If there are any "
            "actionable items, include them in the summary."
        )
        response = self.assistant.azure_client.chat.completions.create(
            model=self.assistant.deployment_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": structured_conversation}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content

    def update_sales_opportunities(self):
        """Update sales opportunities dynamically."""
        new_opportunity = self.assistant.analyze_sales_opportunity(self.transcript)
        self.sales_opportunities.append(new_opportunity)

    def categorize_final_sentiment(self,final_sentiment):
        """
        Categorizes the final sentiment score and provides an explanation.
        Sentiment score ranges:
        - 0 to 40: Negative
        - 41 to 60: Neutral
        - 61 to 100: Positive
        """
        try:
            if final_sentiment > 60:
                sentiment_category = "Positive"
                explanation = (
                    f"The final sentiment score is {final_sentiment}, which falls into the Positive range. "
                    "This indicates that the overall tone of the conversation was optimistic or favorable."
                )
            elif final_sentiment < 40:
                sentiment_category = "Negative"
                explanation = (
                    f"The final sentiment score is {final_sentiment}, which falls into the Negative range. "
                    "This suggests that the overall tone of the conversation was dissatisfied or unfavorable."
                )
            else:
                sentiment_category = "Neutral"
                explanation = (
                    f"The final sentiment score is {final_sentiment}, which falls into the Neutral range. "
                    "This indicates a balanced or neutral tone with no strong emotional cues."
                )

            return {
                "sentiment_score": final_sentiment,
                "sentiment_category": sentiment_category,
                "explanation": explanation
            }
        except Exception as e:
            # Handle any unexpected issues
            return {
                "sentiment_score": final_sentiment,
                "sentiment_category": "Unknown",
                "explanation": f"An error occurred during categorization: {str(e)}"
            }

@router.post("/callAgent/upload_multiple_documents")
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    logging.info(f"Received {len(files)} files")

    try:
        result = await assistant.process_document(files)

        return JSONResponse(content={
            "status": result['status'],
            "message": "Documents processed" if result['status'] == 'success' else "Some documents failed to process",
            "results": result['doc_ids']
        })

    except Exception as e:
        logging.error(f"Error processing files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing documents"
        )

@router.post("/callAgent/offline_query")
async def offline_query(query: str):
    try:
        # Step 1: Document search using vector similarity
        doc_match = assistant.search_documents(query)
        if doc_match:
            response = assistant.generate_ai_response(
                f"Based on the document content: {doc_match}\n\nQuery: {query}"
            )
            source = "document"
        else:
            # Step 2: ChromaDB history search
            try:
                history_results = assistant.qa_collection.query(
                    query_texts=[query],
                    n_results=1
                )

                if (history_results and
                    history_results.get('metadatas') and
                    len(history_results['metadatas']) > 0 and
                    history_results.get('distances') and
                    len(history_results['distances']) > 0 and
                    history_results['distances'][0][0] < 0.3):

                    response = history_results['metadatas'][0][0]['response']
                    source = "history"
                else:
                    # Step 3: LLM fallback
                    response = assistant.generate_ai_response(query)
                    source = "llm"

                    # Store new Q&A in ChromaDB
                    assistant.qa_collection.add(
                        documents=[query],
                        metadatas=[{
                            "response": response,
                            "source": source,
                            "timestamp": datetime.now().isoformat()
                        }],
                        ids=[str(uuid.uuid4())]
                    )

            except Exception as chroma_error:
                logging.error(f"ChromaDB error: {str(chroma_error)}")
                # Fallback to LLM if ChromaDB fails
                response = assistant.generate_ai_response(query)
                source = "llm_fallback"

        return JSONResponse({
            "status": "success",
            "response": response,
            "source": source,
            "document_match": bool(doc_match),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Error in offline query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
#------------------ml recommendation api's--------------------------------------------
# Initialize global model and target column
model = None
current_target_column = None

class TrainingRequest(BaseModel):
    target_column: str

class TrainingMetrics(BaseModel):
    accuracy: float
    confusion_matrix: List[List[int]]
    best_parameters: Dict[str, Any]
    message: str
    target_column: str

class ManualPredictionInput(BaseModel):
    data: Dict[str, Any]

@router.post("/Recommendation/train_model", response_model=TrainingMetrics)
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    global model, current_target_column
    temp_file = f"temp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    try:
        # Save uploaded file
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Validate target column exists in dataset
        df = pd.read_csv(temp_file)
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")

        # Initialize and train model
        model = DynamicClassification(dataset_path=temp_file, target_column=target_column)
        metrics = model.train_model()

        # Store current target column
        current_target_column = target_column

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return TrainingMetrics(
            accuracy=metrics['accuracy'],
            confusion_matrix=metrics['confusion_matrix'],
            best_parameters=metrics['best_parameters'],
            message="Model trained successfully",
            target_column=target_column
        )
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/Recommendation/bulk_upload")
async def bulk_predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")

    temp_file = f"temp_predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    predictions_file = temp_file.replace('.csv', f'_{current_target_column}_predictions.csv')

    try:
        # Save uploaded file
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Generate predictions
        model.predict_bulk(temp_file)

        # Verify predictions file exists
        if not os.path.exists(predictions_file):
            raise HTTPException(status_code=500, detail="Failed to generate predictions file")

        # Create response object
        response = FileResponse(
            path=predictions_file,
            media_type="text/csv",
            filename=f"{current_target_column}_predictions.csv"
        )

        # Clean up input file only
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Return response with predictions file
        return response

    except Exception as e:
        # Clean up both files on error
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(predictions_file):
            os.remove(predictions_file)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/Recommendation/form_predict")
async def manual_predict(input_data: ManualPredictionInput):
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")

    try:
        result = model.predict_single(input_data.data)
        if result is None:
            raise HTTPException(status_code=500, detail="Prediction failed")

        return {
            f"predicted_{current_target_column}": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/Recommendation/required_columns")
async def get_required_columns():
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")

    try:
        return {
            "required_columns": model.original_columns,
            "column_types": model.column_types,
            "datetime_columns": model.datetime_columns,
            "categorical_columns": list(model.one_hot_columns.keys()) if model.one_hot_columns else [],
            "target_column": current_target_column
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/callAgent/online/text")
async def query_endpoint(mobile_no: str, query: str):
    """
    Handles text-based queries for the call session. Creates a new session if none exists
    and answers the query using document search first, then LLM if no relevant content is found.
    Automatically generates a PDF report when the session ends and includes the PDF path in the response.
    """
    try:
        # Validate mobile number
        mobile_no = validate_mobile_no(mobile_no)

        # Initialize new session if none exists
        if mobile_no not in sessions:
            session = CallSession(assistant, db, mobile_no)  # Use the global `assistant`
            sessions[mobile_no] = session
            logging.info(f"New session created for mobile number: {mobile_no}")

        session = sessions[mobile_no]

        # Check if this is a session end request
        if query.lower().strip() == "thanks":
            # Generate final call summary
            call_summary = session.generate_call_summary()
            sales_opportunity = session.assistant.analyze_sales_opportunity(session.transcript)
            session.save_call_record(call_summary)

            # Categorize the final sentiment
            final_sentiment = session.compute_average_sentiment()
            sentiment_analysis = session.categorize_final_sentiment(final_sentiment)

            # Ensure reports directory exists
            reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
            os.makedirs(reports_dir, exist_ok=True)

            # Generate and save wordcloud
            wordcloud_path = None
            if session.wordcloud_data:
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white'
                ).generate_from_frequencies(session.wordcloud_data)

                wordcloud_path = os.path.join(
                    reports_dir,
                    f'wordcloud_{mobile_no}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                )
                wordcloud.to_file(wordcloud_path)

            # Generate PDF report
            pdf_path = PDFGenerator.generate_call_report(
                mobile_no=mobile_no,
                conversation_history=session.transcript,
                call_summary=call_summary,
                sale_opportunity=sales_opportunity,
                chat_history=session.transcript,
                wordcloud_path=wordcloud_path,
                avg_sentiment=final_sentiment
            )

            pdf_file_name = os.path.basename(pdf_path)

            # Clean up wordcloud file after it's been included in the PDF
            if wordcloud_path and os.path.exists(wordcloud_path):
                try:
                    os.remove(wordcloud_path)
                except Exception as e:
                    logging.warning(f"Failed to clean up wordcloud file: {str(e)}")

            # Prepare final response
            final_response = {
                "status": "conversation_ended",
                "call_summary": call_summary,
                "chat_history": session.transcript,
                "relevant_summary": db.find_similar_in_customer_history(query, mobile_no),
                "all_summary": db.get_all_call_summaries(mobile_no),
                "final_wordcloud": session.save_wordcloud(),
                "sales_opportunity": sales_opportunity,
                "final_sentiment": sentiment_analysis["sentiment_score"],
                "sentiment_category": sentiment_analysis["sentiment_category"],
                "explanation": sentiment_analysis["explanation"],
                "pdf_report_file_name": pdf_file_name  # Include the PDF path in the response
            }

            # Clean up session
            del sessions[mobile_no]
            logging.info(f"Session ended for mobile number: {mobile_no}")

            return final_response

        # Process the query normally
        logging.info(f"Processing query for mobile number {mobile_no}: {query}")

        # Step 1: Search for a relevant document match
        document_match = assistant.search_documents(query)  # Use global `assistant`
        if document_match:
            ai_response = assistant.generate_ai_response(
                f"Based on the document content: {document_match}\n\nQuery: {query}"
            )
            source = "document"
        else:
            # Step 2: Fall back to the LLM
            ai_response = assistant.generate_ai_response(query)
            source = "llm"

        # Analyze sentiment
        sentiment_analysis = json.loads(assistant.analyze_sentiment(query))

        # Update session data
        session.transcript.append({"type": "customer", "message": query})
        session.transcript.append({"type": "assistant", "message": ai_response})
        session.sentiment_scores.append(sentiment_analysis.get("sentiment_score", 50.0))
        session.update_wordcloud(query)

        # Return response for ongoing conversation
        return {
            "status": "ongoing",
            "source": source,
            "ai_response": ai_response,
            "sentiment": sentiment_analysis,
            "chat_history": session.transcript,
            "relevant_summary": db.find_similar_in_customer_history(query, mobile_no),
            "all_summary": db.get_all_call_summaries(mobile_no)
        }

    except HTTPException as http_err:
        logging.error(f"HTTP error: {http_err.detail}")
        raise http_err
    except Exception as e:
        logging.error(f"Unexpected error in query_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/callAgent/online/text/getAllSummaries")
async def get_all_summaries(mobile_no: str):
    """
    Fetches all call summaries for a given mobile number.
    """
    try:
        # Validate mobile number
        mobile_no = validate_mobile_no(mobile_no)

        # Fetch all call summaries from the database
        all_summaries = db.get_all_call_summaries(mobile_no)

        if not all_summaries:
            raise HTTPException(status_code=404, detail="No call summaries found for this number")

        return {"mobile_no": mobile_no, "all_summaries": all_summaries}

    except HTTPException as http_err:
        logging.error(f"HTTP error: {http_err.detail}")
        raise http_err
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/callAgent/online/speech")
async def single_test(mobile_no: str):
    """
    Handles speech input for the call session.
    """
    try:
        # Validate mobile number
        mobile_no = validate_mobile_no(mobile_no)

        # Initialize CallCenterAssistant and CallSession
        if mobile_no not in sessions:
            assistant = CallCenterAssistant()
            session = CallSession(assistant, db, mobile_no)
            sessions[mobile_no] = session

        session = sessions[mobile_no]

        # Process speech input
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            logging.info("Listening for customer query...")
            try:
                audio = recognizer.listen(source, timeout=20, phrase_time_limit=20)
                query = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                logging.error("Speech recognition could not understand the audio.")
                raise HTTPException(status_code=400, detail="Speech recognition could not understand the audio.")
            except sr.RequestError as e:
                logging.error(f"Speech recognition service error: {e}")
                raise HTTPException(status_code=500, detail="Speech recognition service error.")

        # Pass the query to the /query endpoint for processing
        response = await query_endpoint(mobile_no=mobile_no, query=query)
        return response

    except HTTPException as http_err:
        logging.error(f"HTTP error: {http_err.detail}")
        raise http_err
    except Exception as e:
        logging.error(f"Unexpected error in single_test endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/callAgent/online/audio")
async def process_audio_file(mobile_no: str = Form(...), audio_file: UploadFile = None):
    """
    Handles audio file input for the call session.
    """
    try:
        # Validate mobile number
        mobile_no = validate_mobile_no(mobile_no)

        # Initialize CallCenterAssistant and CallSession
        if mobile_no not in sessions:
            assistant = CallCenterAssistant()
            session = CallSession(assistant, db, mobile_no)
            sessions[mobile_no] = session

        session = sessions[mobile_no]

        # Ensure an audio file is provided
        if not audio_file:
            raise HTTPException(status_code=400, detail="Audio file is required.")

        # Process the uploaded audio file
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_file.file) as source:
                logging.info("Processing uploaded audio file...")
                audio = recognizer.record(source)
                query = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            logging.error("Speech recognition could not understand the audio.")
            raise HTTPException(status_code=400, detail="Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            logging.error(f"Speech recognition service error: {e}")
            raise HTTPException(status_code=500, detail="Speech recognition service error.")

        # Pass the query to the /query endpoint for processing
        response = await query_endpoint(mobile_no=mobile_no, query=query)
        return response

    except HTTPException as http_err:
        logging.error(f"HTTP error: {http_err.detail}")
        raise http_err
    except Exception as e:
        logging.error(f"Unexpected error in process_audio_file endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/callAgent/downloadReport/{report_name}")
async def download_report(report_name: str):
    """
    This endpoint allows the user to download the PDF report by providing the report name.
    The report is assumed to be in the 'reports' directory.
    """
    try:
        # print(os.path.join(os.path.dirname(__file__)))
        reports_dir = os.path.join(os.path.dirname(__file__), 'CallAgent', 'reports')

        pdf_path = os.path.join(reports_dir, report_name)

        if not os.path.exists(pdf_path):
                       raise HTTPException(status_code=404, detail="Report not found")

        return FileResponse(pdf_path, media_type='application/pdf', filename=report_name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in downloading the report: {str(e)}")



#-----------Fetching the files from NFS PATH --------


NFS_PATH = os.getenv("NFS_PATH")
if not NFS_PATH:
    raise RuntimeError("Environment variable NFS_PATH is not set.")

@router.get("/cdc/files/list", response_model=List[str])
async def list_files():
    """
    Get all file names from the NFS path
    Returns:
        List[str]: List of file names
    """
    try:
        # Get all files from the directory
        files = [f for f in os.listdir(NFS_PATH) if os.path.isfile(os.path.join(NFS_PATH, f))]
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing directory: {str(e)}")

@router.get("/cdc/files/download/{filename}")
async def download_file(filename: str):
    """
    Download a specific file from the NFS path
    Args:
        filename (str): Name of the file to download
    Returns:
        FileResponse: File content for download
    """
    try:
        file_path = os.path.join(NFS_PATH, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Check if path is safe (prevents directory traversal)
        if not Path(file_path).resolve().is_relative_to(Path(NFS_PATH)):
            raise HTTPException(status_code=400, detail="Invalid file path")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")



app.include_router(router)
app.include_router(email_router, prefix="/rag_doc/Automated_email_response", tags=["Automated Email Response"])
#app.include_router(docbot_router, prefix="/rag_doc", tags=["DocBot"])
app.include_router(dataengineering_router, prefix="/rag_doc/dataengineering", tags=["Data Engineering"])
app.include_router(customer_router, prefix="/rag_doc/customer", tags=["Customer Support"])
app.include_router(sentiment_router, prefix="/rag_doc/sentiment", tags=["Sentiment Analysis"])
app.include_router(digitalmarketing_router, prefix="/rag_doc/digital_marketing", tags=["Digital Marketing"])
app.include_router(network_router, prefix="/rag_doc/fraudDetection", tags=["Fraud Detection Network"])
app.include_router(tts_router, prefix="/rag_doc/tts", tags=["TTS"])
app.include_router(promptfusion_router, prefix="/rag_doc/PromptFusion", tags=["Prompt Fusion"])
app.include_router(compliance_router, prefix="/rag_doc/compliance", tags=["Compliance"])
#app.include_router(credit_router, prefix="/rag_doc/credit_underwriting", tags=["Credit Underwriting"])
app.include_router(specbot_router, prefix="/rag_doc/spec_bot", tags=["Spec Bot"])
app.include_router(credit_router, prefix="/rag_doc/credit_underwriting", tags=["Credit Underwriting"])
app.include_router(docbot_router, prefix="/rag_doc/v2/docbot", tags=["docbot2.0"])
app.include_router(TransportationDemandForecasting_router, prefix="/rag_doc/TransportationDemandForecasting", tags=["Transportation Demand Forecasting"])
app.include_router(Fleet_router, prefix="/rag_doc/Fleet", tags=["Fleet2.1"])