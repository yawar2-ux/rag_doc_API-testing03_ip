import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import uuid
from datetime import datetime
import logging
from dotenv import load_dotenv

# Add tabulate to imports
try:
    import tabulate
except ImportError:
    logging.warning("Tabulate is not installed. Tables may not render correctly.")

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatSession(BaseModel):
    session_id: str
    files: List[str] = []
    dataframes: Dict[str, pd.DataFrame] = {}
    # Store JSON data as structured dicts for initial processing
    raw_json_data: Dict[str, Any] = {}
    # Store Chroma collection names for each JSON file
    json_vector_stores: Dict[str, Chroma] = Field(default_factory=dict)
    memory: Optional[Any] = None
    agent: Optional[Any] = None
    created_at: datetime
    last_activity: datetime

    class Config:
        arbitrary_types_allowed = True

class FleetChatAgent:
    """
    Chat agent that can interact with CSV and JSON data using LangChain.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the chat agent.
        
        Args:
            api_key: OpenAI API key or OpenRouter API key
            base_url: Base URL for the API (for OpenRouter or custom endpoints)
        """
        # Get API key with explicit fallback to value in .env
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("No OpenRouter API key found. Set OPENROUTER_API_KEY in environment or .env file.")
            self.api_key = "<OPENROUTER_API_KEY>" # Fallback placeholder
        else:
            logger.info(f"API key loaded successfully (first 5 chars: {self.api_key[:5]})")
            
        # Get base URL with fallback
        self.base_url = base_url or os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
        self.sessions: Dict[str, ChatSession] = {}
        
        # Initialize embeddings model and text splitter
        try:
            # Replace OpenAI embeddings with HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Popular v2 sentence embedding model
                model_kwargs={"device": "cpu"},  # Use CPU by default, can change to "cuda" for GPU
                encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better similarity search
            )
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            logger.info("HuggingFace embeddings and text splitter initialized.")
        except Exception as e:
            logger.error(f"Error initializing embeddings/text_splitter: {str(e)}")
            # Decide if this is a fatal error or if the app can run without embeddings
            # For now, let it raise if critical components fail
            raise

        # Initialize the LLM
        try:
            self.llm = ChatOpenAI(
                temperature=0.1,
                model=os.getenv("DEFAULT_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free"),
                openai_api_key=self.api_key,
                openai_api_base=self.base_url
            )
            logger.info(f"ChatOpenAI client initialized with base URL: {self.base_url}")
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAI client: {str(e)}")
            raise
    
    def create_session(self) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        session = ChatSession(
            session_id=session_id,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        self.sessions[session_id] = session
        return session_id
    
    def upload_files(self, session_id: str, csv_files: List[Path] = None, json_files: List[Path] = None) -> Dict[str, Any]:
        """
        Upload and process CSV and JSON files for a session.
        
        Args:
            session_id: The session ID
            csv_files: List of CSV file paths
            json_files: List of JSON file paths
            
        Returns:
            Dictionary with upload results
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        results = {"csv_files": [], "json_files": [], "errors": []}
        
        # Process CSV files
        if csv_files:
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    file_name = csv_file.stem.replace(" ", "_").replace("-", "_") # Sanitize name
                    session.dataframes[file_name] = df
                    session.files.append(str(csv_file))
                    
                    results["csv_files"].append({
                        "name": file_name,
                        "rows": len(df),
                        "columns": list(df.columns),
                    })
                except Exception as e:
                    error_msg = f"Error processing CSV {csv_file}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
        
        # Process JSON files
        if json_files:
            for json_file_path in json_files:
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Remove "visualizations" key if it exists
                    if isinstance(data, dict) and "visualizations" in data:
                        del data["visualizations"]
                        logger.info(f"Removed 'visualizations' key from JSON data in {json_file_path.name}")

                    file_name = json_file_path.stem.replace(" ", "_").replace("-", "_") # Sanitize name
                    session.raw_json_data[file_name] = data # Store raw data if needed later
                    session.files.append(str(json_file_path))

                    # Convert JSON to text and create documents
                    json_text = json.dumps(data, indent=2)
                    docs = [Document(page_content=json_text, metadata={"source": file_name})]
                    
                    # Split documents
                    split_docs = self.text_splitter.split_documents(docs)
                    
                    # Create in-memory Chroma vector store for this JSON file
                    # Collection name must be unique and follow Chroma's naming rules
                    collection_name = f"session_{session_id.replace('-', '_')}_json_{file_name}"
                    
                    # Ensure collection_name is valid (basic check)
                    collection_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in collection_name)
                    if not (3 <= len(collection_name) <= 63): # Chroma length constraints
                        collection_name = f"col_{uuid.uuid4().hex[:50]}" # Fallback unique name

                    logger.info(f"Creating Chroma collection: {collection_name} for {file_name}")
                    vector_store = Chroma.from_documents(
                        documents=split_docs,
                        embedding=self.embeddings,
                        collection_name=collection_name,
                        # persist_directory=None # For in-memory
                    )
                    session.json_vector_stores[file_name] = vector_store
                    
                    results["json_files"].append({
                        "name": file_name,
                        "type": type(data).__name__,
                        "vector_store_collection": collection_name
                    })
                except Exception as e:
                    error_msg = f"Error processing JSON {json_file_path}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    results["errors"].append(error_msg)
        
        # Create or update the agent with new data
        self._create_agent(session_id)
        session.last_activity = datetime.now()
        
        return results
    
    def _create_agent(self, session_id: str):
        """Create or update the agent for a session."""
        session = self.sessions[session_id]
        session.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        tools = []

        # Add pandas dataframe tools for CSVs
        for name, df in session.dataframes.items():
            sanitized_name = name.replace(" ", "_").replace("-", "_")
            tool_name = f"query_csv_dataset_{sanitized_name}"
            tool_description = (
                f"Use this tool to answer questions about the '{sanitized_name}' CSV dataset. "
                f"Input should be a clear, natural language question about the data in this specific CSV. "
                f"The dataset has columns: {', '.join(df.columns)}. "
                f"Example questions: 'What is the average value of column X in {sanitized_name}?', "
                f"'Show me rows where column Y is Z in {sanitized_name}?'"
            )
            
            # Closure to capture df and df_name_for_agent correctly
            def create_query_func(dataframe, df_name_for_agent):
                def query_dataframe(query: str) -> str:
                    try:
                        logger.info(f"Pandas agent for '{df_name_for_agent}' received query: {query}")
                        pandas_agent = create_pandas_dataframe_agent(
                            llm=self.llm,
                            df=dataframe, # Pass the actual DataFrame
                            verbose=True, # Enable verbose for debugging agent steps
                            agent_type=AgentType.OPENAI_TOOLS, # Changed from OPENAI_FUNCTIONS
                            handle_parsing_errors=True,
                            allow_dangerous_code=True, # Be cautious with this in production
                            # You might want to add a prefix or suffix to the prompt if needed
                            # prefix="You are working with a pandas dataframe in Python. The name of the dataframe is `df`."
                        )
                        # response = pandas_agent.run(query) # Deprecated
                        response_dict = pandas_agent.invoke({"input": query})
                        response = response_dict.get("output", str(response_dict)) # Extract output
                        return f"Result from CSV '{df_name_for_agent}': {response}"
                    except Exception as e:
                        logger.error(f"Error querying CSV '{df_name_for_agent}': {str(e)}", exc_info=True)
                        return f"Error querying CSV '{df_name_for_agent}': {str(e)}. Please try rephrasing your question or ask about available columns."
                return query_dataframe

            tools.append(Tool(
                name=tool_name,
                description=tool_description,
                func=create_query_func(df, sanitized_name) # Pass df and its name
            ))

        # Add ChromaDB query tools for JSONs
        for file_name, vector_store in session.json_vector_stores.items():
            sanitized_name = file_name.replace(" ", "_").replace("-", "_")
            tool_name = f"query_json_document_{sanitized_name}"
            tool_description = (
                f"Use this tool to answer questions based on the content of the JSON document named '{sanitized_name}'. "
                f"Input should be a specific question about the information contained in this JSON document. "
                f"For example: 'What is the value of key X in {sanitized_name}?' or 'Summarize the section Y of {sanitized_name}?'"
            )

            def create_json_query_func(vs, vs_name):
                def query_vector_store(query: str) -> str:
                    try:
                        logger.info(f"Querying JSON vector store '{vs_name}' with: {query}")
                        # Perform similarity search
                        docs = vs.similarity_search(query, k=3) # Get top 3 relevant chunks
                        if not docs:
                            return f"No relevant information found in JSON document '{vs_name}' for your query."
                        
                        context = "\n\n".join([doc.page_content for doc in docs])
                        # Optionally, could use an LLM to synthesize an answer from context
                        # For now, return the retrieved context
                        return f"Relevant information from JSON document '{vs_name}':\n{context}"
                    except Exception as e:
                        logger.error(f"Error querying JSON vector store '{vs_name}': {str(e)}", exc_info=True)
                        return f"Error querying JSON document '{vs_name}': {str(e)}"
                return query_vector_store

            tools.append(Tool(
                name=tool_name,
                description=tool_description,
                func=create_json_query_func(vector_store, sanitized_name)
            ))
        
        csv_file_names = list(session.dataframes.keys())
        json_doc_names = list(session.json_vector_stores.keys())

        system_message_template = f"""You are a helpful data analysis assistant.
You have access to the following data sources:
- CSV Files: {', '.join(csv_file_names) if csv_file_names else 'None'}
- JSON Documents: {', '.join(json_doc_names) if json_doc_names else 'None'}

When a user asks a question:
1. Determine if the question pertains to a CSV file or a JSON document.
2. If it's about a CSV file, use the appropriate 'query_csv_dataset_<csv_file_name>' tool.
3. If it's about a JSON document, use the appropriate 'query_json_document_<json_file_name>' tool.
4. Provide clear, concise answers based on the tool's output.
5. If a tool returns an error, inform the user and suggest rephrasing or asking a different question.
6. Do not make up information. If you cannot answer, say so.

Chat History: {{chat_history}}
User Input: {{input}}
Agent Scratchpad: {{agent_scratchpad}}"""
        
        prompt = PromptTemplate(
            input_variables=["chat_history", "input", "agent_scratchpad"],
            template=system_message_template
        )

        try:
            agent_executor = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=session.memory,
                verbose=True, # Enable verbose for debugging agent steps
                max_iterations=7, # Increased slightly
                early_stopping_method="generate",
                handle_parsing_errors="Check your output and make sure it conforms to the expected format. If the error persists, try simplifying your response or thinking step by step.",
                # agent_kwargs={"system_message": system_message_template} # For some agent types
            )
            # For CONVERSATIONAL_REACT_DESCRIPTION, the prompt is usually part of the agent's internal setup,
            # but we can try to influence it via the tool descriptions and overall guidance.
            # The main prompt for this agent type is often constructed from tool descriptions.
            # We can also try to set a custom prompt for the LLMChain within the agent if needed,
            # but initialize_agent abstracts much of this.
            # The `prompt` variable created above is more for constructing the system message if the agent type supports it directly.
            # For Conversational React, the system message is implicitly built.
            # Let's ensure the tool descriptions are very clear.

            # A more direct way to influence the system prompt for some agents:
            if hasattr(agent_executor.agent, 'llm_chain') and hasattr(agent_executor.agent.llm_chain, 'prompt'):
                 # This is a bit of a hack and might not always work depending on agent structure
                try:
                    # Create a new prompt or modify existing. This is complex.
                    # For now, relying on tool descriptions and the general agent behavior.
                    # The template above is more of a guideline for how the agent *should* behave.
                    pass
                except Exception as e:
                    logger.warning(f"Could not directly set system prompt on agent's LLM chain: {e}")


            session.agent = agent_executor
            logger.info(f"Agent created for session {session_id} with {len(tools)} tools.")

        except Exception as e:
            logger.error(f"Error initializing agent for session {session_id}: {str(e)}", exc_info=True)
            session.agent = None # Ensure agent is None if initialization fails
            # Optionally, re-raise or handle this to inform the user
    
    def chat(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Chat with the agent about the uploaded data.
        
        Args:
            session_id: The session ID
            message: User message
            
        Returns:
            Dictionary with response and metadata
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        if not session.agent:
            return {
                "response": "No data uploaded yet. Please upload CSV or JSON files first.",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "error": "No agent available"
            }
        
        try:
            # Get response from agent
            # response = session.agent.run(input=message) # Deprecated
            response_dict = session.agent.invoke({"input": message})
            response = response_dict.get("output", str(response_dict)) # Extract output
            session.last_activity = datetime.now()
            
            return {
                "response": response,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "data_sources": {
                    "csv_files": list(session.dataframes.keys()),
                    "json_files": list(session.json_vector_stores.keys())
                }
            }
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "files_count": len(session.files),
            "csv_datasets": list(session.dataframes.keys()),
            "json_datasets": list(session.json_vector_stores.keys()),
            "has_agent": session.agent is not None
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [self.get_session_info(session_id) for session_id in self.sessions.keys()]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

# Global agent instance
fleet_chat_agent = FleetChatAgent()
