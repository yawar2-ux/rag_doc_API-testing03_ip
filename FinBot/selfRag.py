import os
import logging
import json
import re
import uuid
import pickle
import hashlib
from typing import Dict, Any, List, Set
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import AzureOpenAI
from PyPDF2 import PdfReader
from langchain.schema import Document
import pdfplumber
import pandas as pd
from datetime import datetime
import shutil
import logging
import traceback
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from time import sleep
import random
# import faiss

from sklearn.metrics.pairwise import cosine_similarity


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('self_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategyTracker:
    def __init__(self):
        self.stored_strategies = {}  # Store strategies by sector
        self.first_run = True


    def store_strategies(self, sector: str, strategies: List[Dict[str, str]]):
        if self.first_run:
            self.stored_strategies[sector] = strategies
            self.first_run = False
            logger.info(f"Stored {len(strategies)} initial strategies for {sector}")

    def get_stored_strategies(self, sector: str) -> List[Dict[str, str]]:
        return self.stored_strategies.get(sector, [])


class GradeDocuments:
    def __init__(self):
        self.binary_score = None

    @classmethod
    def from_dict(cls, data):
        instance = cls()
        instance.binary_score = data.get("binary_score", "no")
        return instance

class EnhancedSelfRAG:
    def __init__(self, persist_dir: str = "./faiss_index"):
        # Initialize directories
        self.persist_dir = persist_dir
        self.pdf_dir = os.path.join(persist_dir, "pdfs")
        os.makedirs(self.pdf_dir, exist_ok=True)

        os.environ['AZURE_OPENAI_API_KEY'] = '8b1d436e85d1452bbcbfd5905921efa6'

        # Initialize Azure OpenAI client
        self.azure_client = AzureOpenAI(
        azure_endpoint="https://rahul.openai.azure.com/",
        api_key='8b1d436e85d1452bbcbfd5905921efa6',  # Direct API key assignment
        api_version="2024-02-15-preview"
    )

        self.pdf_cache = {}
        self.strategy_cache = {}
        self.strategy_attempts = {}
        self.max_attempts = 20  # Maximum attempts to generate new strategies

    #     # Load FAISS index
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModel.from_pretrained("gpt2")
        self.model.eval()


        # Clear the PDF directory
        if os.path.exists(self.pdf_dir):
            shutil.rmtree(self.pdf_dir)
            os.makedirs(self.pdf_dir, exist_ok=True)
        #




        # Add these new attributes
        self.visited_urls = set()
        self.doc_splits = []

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize models and components
        self.initialize_components()

        # Clear storage and initialize empty state
        self.clear_storage()

        self.logger.info("Initialized fresh EnhancedSelfRAG instance with empty storage")

    def initialize_components(self):
        self.sentence_transformer = SentenceTransformer("yiyanghkust/finbert-tone")
        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")
        self.model.eval()

        # Initialize storage components
        self.documents = []
        self.document_count = 0
        self.vectorstore = None

         # Initialize Azure OpenAI client
        self.llm = AzureOpenAI(
        azure_endpoint="https://rahul.openai.azure.com/",
        api_key='8b1d436e85d1452bbcbfd5905921efa6',  # Direct API key assignment
        api_version="2024-02-15-preview"
    )

        # Initialize caches and clients
        self.pdf_cache = {}
        self.azure_client = self.initialize_azure_client()

        # Add Azure OpenAI configuration
        os.environ['AZURE_OPENAI_API_KEY'] = '8b1d436e85d1452bbcbfd5905921efa6'



        self.retriever = None
        self.retrieval_grader = None
        self.rag_chain = None
        self.initialize_retrieval_components()

    def clear_storage(self):
        """Clear all previous documents and reset storage"""
    # Clear PDF directory
        if os.path.exists(self.pdf_dir):
            for item in os.listdir(self.pdf_dir):
                item_path = os.path.join(self.pdf_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)

        # Reset all storage components
        self.documents = []
        self.document_count = 0
        self.vectorstore = None
        self.pdf_cache = {}

        # Clear strategy cache
        self.strategy_cache = {}
        self.strategy_attempts = {}

        # Recreate necessary directories
        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(self.pdf_dir, exist_ok=True)

        logger.info("Cleared all previous documents and reset storage")


        # Initialize sector parameters configuration
        self.sector_params = {
        "investment": {
            "investment_amount": int,
            "risk_tolerance": int,
            "investment_horizon": int,
            "current_portfolio_value": int
        },
        "retail": {
            "loan_amount": int,
            "credit_score": int,
            "monthly_income": int,
            "debt_to_income_ratio": int
        },
        "taxation": {
            "annual_income": int,
            "deductions": int,
            "tax_credits": int,
            "capital_gains": int
        },
        "trading": {
            "trading_capital": int,
            "risk_per_trade": int,
            "trading_experience": int,
            "preferred_assets": int
        },
        "credit_card": {
            "credit_score": int,
            "annual_income": int,
            "monthly_spending": int,
            "preferred_rewards": str,
            "card_type": str
        }
    }

    def initialize_azure_client(self) -> AzureOpenAI:
        os.environ['AZURE_OPENAI_API_KEY'] = '8b1d436e85d1452bbcbfd5905921efa6'
        return AzureOpenAI(
            azure_endpoint="https://rahul.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview"
        )


    def initialize_retrieval_components(self):
    # Create Azure OpenAI chat completion function
        def azure_chat_completion(messages):
            response = self.azure_client.chat.completions.create(
                model="RAG",
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content

        # Create grading prompt template
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized financial document grader evaluating content relevance for strategy generation.
#             Score documents based on their relevance to financial strategy development."""),
            ("human", "Document: {document}\nQuestion: {question}")
        ])

        # Create retrieval grader chain
        self.retrieval_grader = grade_prompt | azure_chat_completion

        rag_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert financial assistant retrieving information for generating strategies.

Task:
1. Extract key financial insights from the provided documents.
2. Focus on details relevant to the sector and input parameters.

Sector-specific parameters:
- **Investment**: Investment Amount, Risk Tolerance, Investment Horizon, Current Portfolio Value.
- **Retail**: Loan Amount, Credit Score, Monthly Income, Debt-to-Income Ratio.
- **Taxation**: Annual Income, Deductions, Tax Credits, Capital Gains.
- **Trading**: Trading Capital, Risk per Trade, Trading Experience, Preferred Assets.
- **Credit Card**: Credit Score, Annual Income, Monthly Spending, Preferred Rewards, Card Type.

For Credit Card recommendations:
1. Match cards based on credit score requirements
2. Consider annual income qualifications
3. Analyze reward structures against spending patterns
4. Compare annual fees vs rewards value
5. Evaluate additional card benefits and perks

Output requirements:
- Summarize actionable insights related to the parameters.
- Reference source documents with page numbers for each insight.
"""),
        ("human", "Content: {context}\nQuestion: {question}")
    ])

        # Create RAG chain
        self.rag_chain = rag_prompt | azure_chat_completion


    def generate_bert_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:

        """Generate embeddings using FinBERT with advanced pooling strategies"""
        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                try:
                    batch = texts[i:i + batch_size]
                    logger.debug(f"Processing batch {i//batch_size + 1}")

                    # Enhanced tokenization with special token handling
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                        return_attention_mask=True,
                        return_token_type_ids=True
                    )

                    # Get model outputs with all hidden states
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        token_type_ids=inputs.token_type_ids,
                        output_hidden_states=True
                    )

                    # Get last 4 hidden layers for better representation
                    last_hidden_states = outputs.hidden_states[-4:]

                    # Concatenate last 4 layers
                    concatenated_states = torch.cat(
                        [hidden_state for hidden_state in last_hidden_states],
                        dim=-1
                    )

                    # Advanced pooling strategy
                    attention_mask = inputs.attention_mask.unsqueeze(-1)

                    # Weighted mean pooling
                    token_embeddings = concatenated_states * attention_mask
                    sum_embeddings = token_embeddings.sum(dim=1)
                    sum_mask = attention_mask.sum(dim=1)

                    # Avoid division by zero
                    sum_mask = torch.clamp(sum_mask, min=1e-9)

                    # Calculate pooled embeddings
                    pooled_embeddings = sum_embeddings / sum_mask

                    # Normalize embeddings
                    normalized_embeddings = torch.nn.functional.normalize(pooled_embeddings, p=2, dim=1)

                    # Convert to numpy and extend list
                    batch_embeddings = normalized_embeddings.numpy()
                    embeddings.extend(batch_embeddings)

                    logger.info(f"Successfully processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                    logger.debug(f"Batch embedding shape: {batch_embeddings.shape}")

                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                    # Return empty embedding vector of correct shape for failed batches
                    failed_batch_size = len(batch)
                    empty_embeddings = np.zeros((failed_batch_size, concatenated_states.shape[-1]))
                    embeddings.extend(empty_embeddings)
                    continue

        final_embeddings = np.array(embeddings)
        logger.info(f"Completed embedding generation. Final shape: {final_embeddings.shape}")
        return final_embeddings


    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Replace OpenAI embeddings with DistilBERT embeddings"""
        return self.generate_bert_embeddings(texts, batch_size)



    def process_documents(self, docs: List[Any]) -> List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        processed_docs = []
        for doc in docs:
            if doc.page_content.strip():
                splits = text_splitter.split_documents([doc])
                processed_docs.extend(splits)
                print(f"Processed {len(splits)} chunks from document")

        return processed_docs

    def filter_relevant_documents(self, query_embedding, doc_embeddings, documents, threshold=0.6):
        relevant_docs = []
        for i, embedding in enumerate(doc_embeddings):
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            if similarity > threshold:
                relevant_docs.append(documents[i])
        return relevant_docs

    def store_sector_parameters(self, sector: str, input_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Store and validate user input parameters for a specific sector"""
        logger.info(f"Storing parameters for sector: {sector}")

        # Initialize parameter storage if not exists
        if not hasattr(self, 'user_parameters'):
            self.user_parameters = {}

        # Map input parameters to sector parameters
        validated_params = {}
        sector_param_template = self.sector_params.get(sector, {})

        for param_name, param_type in sector_param_template.items():
            if param_name in input_parameters:
                try:
                    validated_params[param_name] = param_type(input_parameters[param_name])
                    logger.info(f"Validated parameter {param_name}: {validated_params[param_name]}")
                except ValueError:
                    logger.warning(f"Invalid value for {param_name}, using default")

        # Store validated parameters
        self.user_parameters[sector] = validated_params
        logger.info(f"Stored parameters for {sector}: {validated_params}")

        return validated_params



    # def generate_strategies(self, context: str, sector: str, sector_params: Dict[str, Any]):
    #     logger.info(f"Generating strategies for sector: {sector}")
    #     logger.info(f"Context length: {len(context)} chars")
    #     logger.info(f"Using parameters: {sector_params}")

    #     if not hasattr(self, 'strategy_history'):
    #         self.strategy_history = {sector: set() for sector in ["investment", "retail", "taxation", "trading", "credit_card"]}

    #     # Handle credit card sector separately
    #     if sector == "credit_card":
    #         formatted_prompt = f"""
    #         Based on the provided documents, recommend credit cards that best match these criteria:
    #         - Credit Score: {sector_params.get('credit_score', 'Not specified')}
    #         - Annual Income: ${sector_params.get('annual_income', 'Not specified')}
    #         - Monthly Spending: ${sector_params.get('monthly_spending', 'Not specified')}
    #         - Preferred Rewards: {sector_params.get('preferred_rewards', 'Not specified')}
    #         - Card Type: {sector_params.get('card_type', 'Not specified')}

    #         For each recommended card:
    #         1. Specify the card name and issuer
    #         2. List key benefits and rewards
    #         3. Detail qualification requirements
    #         4. Calculate potential rewards value based on spending
    #         5. Compare with similar cards
    #         6. Include any special offers or promotions
    #         7. Reference specific page numbers from source documents

    #         Context:
    #         {context}
    #         """
    #         strategy_prompt = formatted_prompt
    #     else:
    #         # Original parameter templates and sector prompts for other sectors
    #         parameter_templates = {
    #             "investment": [
    #                 f"For an investment amount of ${sector_params.get('investment_amount', 0)}, specifically allocate...",
    #                 f"Given the {sector_params.get('risk_tolerance', 0)}% risk tolerance, utilize...",
    #                 f"With a {sector_params.get('investment_horizon', 0)} year horizon, implement..."
    #             ],
    #             "retail": [
    #                 f"For a loan amount of ${sector_params.get('loan_amount', 0)}, specifically structure...",
    #                 f"Given the credit score of {sector_params.get('credit_score', 0)}, recommend...",
    #                 f"With monthly income of ${sector_params.get('monthly_income', 0)} and DTI of {sector_params.get('debt_to_income_ratio', 0)}%, design..."
    #             ],
    #             "taxation": [
    #                 f"For annual income of ${sector_params.get('annual_income', 0)}, optimize...",
    #                 f"Given deductions of ${sector_params.get('deductions', 0)} and credits of ${sector_params.get('tax_credits', 0)}, structure...",
    #                 f"With capital gains of ${sector_params.get('capital_gains', 0)}, implement..."
    #             ],
    #             "trading": [
    #                 f"For trading capital of ${sector_params.get('trading_capital', 0)}, position...",
    #                 f"Given {sector_params.get('trading_experience', 0)} years experience and {sector_params.get('risk_per_trade', 0)}% risk per trade, execute...",
    #                 f"With focus on {sector_params.get('preferred_assets', 0)} assets, develop..."
    #             ]
    #         }

    #         sector_prompts = {
    #             "investment": """
    #                 Based on the provided content, generate 3-4 specific investment strategies that:
    #                 1. Focus on investment product features and opportunities
    #                 2. Consider the investment parameters:
    #                 - Investment amount: {investment_amount}
    #                 - Risk tolerance: {risk_tolerance}
    #                 - Investment horizon: {investment_horizon} years
    #                 - Current portfolio value: {current_portfolio_value}
    #             """,
    #         "retail": """
    #             Based on the provided content, generate 3-4 specific retail banking strategies that:
    #             1. Focus on retail banking products and services
    #             2. Consider the retail parameters:
    #             - Loan amount: {loan_amount}
    #             - Credit score: {credit_score}
    #             - Monthly income: {monthly_income}
    #             - Debt-to-income ratio: {debt_to_income_ratio}
    #         """,
    #         "taxation": """
    #             Based on the provided content, generate 3-4 specific tax planning strategies that:
    #             1. Focus on tax optimization and compliance
    #             2. Consider the taxation parameters:
    #             - Annual income: {annual_income}
    #             - Deductions: {deductions}
    #             - Tax credits: {tax_credits}
    #             - Capital gains: {capital_gains}
    #         """,
    #         "trading": """
    #             Based on the provided content, generate 3-4 specific trading strategies that:
    #             1. Focus on market trading approaches and risk management
    #             2. Consider the trading parameters:
    #             - Trading capital: {trading_capital}
    #             - Risk per trade: {risk_per_trade}
    #             - Trading experience: {trading_experience}
    #             - Preferred assets: {preferred_assets}
    #         """
    #     }

    #         sector_prompt = sector_prompts.get(sector, "").format(**sector_params)

    #         strategy_prompt = f"""
    #         You are an expert financial advisor analyzing the provided PDF documents for the {sector} sector.

    #         {sector_prompt}

    #         IMPORTANT:
    #         1. Generate strategies ONLY using specific information found in the provided document content.
    #         2. DO NOT generate generic strategies. Each strategy must reference specific products, numbers, or methods mentioned in the documents.
    #         3. Generate 4-5 UNIQUE strategies that have NOT been suggested before
    #         4. Each strategy must be completely different in approach and products used
    #         5. Generate strategies ONLY using specific information from the provided document content
    #         6. Ensure NO overlap with these previously suggested strategies:
    #             {list(self.strategy_history[sector])}

    #         Document Content:
    #         {context}

    #         Strategy Generation Rules:
    #         1. Start each strategy with direct parameter application using these exact formats:
    #         {parameter_templates[sector][0]}
    #         {parameter_templates[sector][1]}
    #         {parameter_templates[sector][2]}

    #         2. Include exact product details:
    #         - Quote specific rates, returns, or fees from documents
    #         - Reference exact product names and features
    #         - Use actual numbers and percentages from the text

    #         3. Link recommendations to parameters:
    #         - Match product risk levels to stated risk tolerance
    #         - Align investment terms with specified time horizons
    #         - Scale suggestions to match given amounts

    #         4. Cite Evidence:
    #         - Quote specific text from documents
    #         - Reference exact page numbers
    #         - Include specific performance metrics

    #         Format each strategy as:
    #         TITLE: [Product/Strategy Name from Document] - [Parameter-Specific Application]
    #         DESCRIPTION: [Detailed strategy linking parameters to specific document content]
    #         SOURCE: [PDF name and exact page number with relevant quote]
    #         """

    #     message_text = [
    #         {"role": "system", "content": f"You are an expert financial advisor generating detailed {sector} strategies."},
    #         {"role": "user", "content": strategy_prompt}
    #     ]

    #     completion = self.llm.chat.completions.create(
    #         model="RAG",
    #         messages=message_text,
    #         temperature=0.1,
    #         max_tokens=1000,
    #         top_p=0.85,
    #         frequency_penalty=0.7,
    #         presence_penalty=0.7
    #     )

    #     response = completion.choices[0].message.content
    #     new_strategies = self.parse_strategies(response)

    #     for strategy in new_strategies:
    #         self.strategy_history[sector].add(strategy['title'])

    #     if not new_strategies:
    #         logger.warning("No strategies generated from response")
    #         return {
    #             "strategies": [],
    #             "metadata": {
    #                 "sector": sector,
    #                 "message": f"Could not generate {sector} strategies from document content"
    #             }
    #         }

    #     return {
    #         "strategies": new_strategies,
    #         "metadata": {
    #             "sector": sector,
    #             "source": "generated",
    #             "strategies_generated": len(new_strategies),
    #             "total_unique_strategies": len(self.strategy_history[sector])
    #         }
    #     }



    def generate_strategies(self, context: str, sector: str, sector_params: Dict[str, Any]):
        logger.info(f"Generating strategies for sector: {sector}")
        logger.info(f"Context length: {len(context)} chars")
        logger.info(f"Using parameters: {sector_params}")

        if not hasattr(self, 'strategy_history'):
            self.strategy_history = {sector: set() for sector in ["investment", "retail", "taxation", "trading", "credit_card"]}

        # Handle credit card sector
        if sector == "credit_card":
            parameter_templates = [
                f"For a credit score of {sector_params.get('credit_score', 0)} and annual income of ${sector_params.get('annual_income', 0)}...",
                f"With monthly spending of ${sector_params.get('monthly_spending', 0)} focusing on {sector_params.get('preferred_rewards', 'general')} rewards...",
                f"Looking for {sector_params.get('card_type', 'general')} cards with optimal benefits..."
            ]

            strategy_prompt = f"""
            You are a credit card recommendation expert analyzing credit card options.

            Generate 3-4 specific credit card recommendations based on:
            - Credit Score: {sector_params.get('credit_score', 'Not specified')}
            - Annual Income: ${sector_params.get('annual_income', 'Not specified')}
            - Monthly Spending: ${sector_params.get('monthly_spending', 'Not specified')}
            - Preferred Rewards: {sector_params.get('preferred_rewards', 'Not specified')}
            - Card Type: {sector_params.get('card_type', 'Not specified')}

            IMPORTANT:
            1. Use ONLY cards mentioned in the provided documents
            2. Each recommendation must cite specific page numbers and quotes
            3. Ensure recommendations are appropriate for the given credit score
            4. Calculate exact rewards value based on monthly spending
            5. Do not overlap with previous recommendations:
            {list(self.strategy_history[sector])}

            For each card recommendation:
            1. Start with parameter application using these formats:
            {parameter_templates[0]}
            {parameter_templates[1]}
            {parameter_templates[2]}

            2. Include exact card details:
            - Annual fee and APR from documents
            - Specific reward rates and categories
            - Sign-up bonus details
            - Additional card benefits

            3. Format as:
            TITLE: [Card Name] - [Primary Benefit Focus]
            DESCRIPTION: [Detailed analysis with reward calculations]
            SOURCE: [Document name, page number, and relevant quote]

            Context:
            {context}
            """
        else:
            # Original parameter templates and sector prompts for other sectors
            parameter_templates = {
                "investment": [
                    f"For an investment amount of ${sector_params.get('investment_amount', 0)}, specifically allocate...",
                    f"Given the {sector_params.get('risk_tolerance', 0)}% risk tolerance, utilize...",
                    f"With a {sector_params.get('investment_horizon', 0)} year horizon, implement..."
                ],
                "retail": [
                    f"For a loan amount of ${sector_params.get('loan_amount', 0)}, specifically structure...",
                    f"Given the credit score of {sector_params.get('credit_score', 0)}, recommend...",
                    f"With monthly income of ${sector_params.get('monthly_income', 0)} and DTI of {sector_params.get('debt_to_income_ratio', 0)}%, design..."
                ],
                "taxation": [
                    f"For annual income of ${sector_params.get('annual_income', 0)}, optimize...",
                    f"Given deductions of ${sector_params.get('deductions', 0)} and credits of ${sector_params.get('tax_credits', 0)}, structure...",
                    f"With capital gains of ${sector_params.get('capital_gains', 0)}, implement..."
                ],
                "trading": [
                    f"For trading capital of ${sector_params.get('trading_capital', 0)}, position...",
                    f"Given {sector_params.get('trading_experience', 0)} years experience and {sector_params.get('risk_per_trade', 0)}% risk per trade, execute...",
                    f"With focus on {sector_params.get('preferred_assets', 0)} assets, develop..."
                ]
            }

            sector_prompts = {
                "investment": """
                    Based on the provided content, generate 3-4 specific investment strategies that:
                    1. Focus on investment product features and opportunities
                    2. Consider the investment parameters:
                    - Investment amount: {investment_amount}
                    - Risk tolerance: {risk_tolerance}
                    - Investment horizon: {investment_horizon} years
                    - Current portfolio value: {current_portfolio_value}
                """,
            "retail": """
                Based on the provided content, generate 3-4 specific retail banking strategies that:
                1. Focus on retail banking products and services
                2. Consider the retail parameters:
                - Loan amount: {loan_amount}
                - Credit score: {credit_score}
                - Monthly income: {monthly_income}
                - Debt-to-income ratio: {debt_to_income_ratio}
            """,
            "taxation": """
                Based on the provided content, generate 3-4 specific tax planning strategies that:
                1. Focus on tax optimization and compliance
                2. Consider the taxation parameters:
                - Annual income: {annual_income}
                - Deductions: {deductions}
                - Tax credits: {tax_credits}
                - Capital gains: {capital_gains}
            """,
            "trading": """
                Based on the provided content, generate 3-4 specific trading strategies that:
                1. Focus on market trading approaches and risk management
                2. Consider the trading parameters:
                - Trading capital: {trading_capital}
                - Risk per trade: {risk_per_trade}
                - Trading experience: {trading_experience}
                - Preferred assets: {preferred_assets}
            """
        }

            sector_prompt = sector_prompts.get(sector, "").format(**sector_params)

            strategy_prompt = f"""
            You are an expert financial advisor analyzing the provided PDF documents for the {sector} sector.

            {sector_prompt}

            IMPORTANT:
            1. Generate strategies ONLY using specific information found in the provided document content.
            2. DO NOT generate generic strategies. Each strategy must reference specific products, numbers, or methods mentioned in the documents.
            3. Generate 4-5 UNIQUE strategies that have NOT been suggested before
            4. Each strategy must be completely different in approach and products used
            5. Generate strategies ONLY using specific information from the provided document content
            6. Ensure NO overlap with these previously suggested strategies:
                {list(self.strategy_history[sector])}

            Document Content:
            {context}

            Strategy Generation Rules:
            1. Start each strategy with direct parameter application using these exact formats:
            {parameter_templates[sector][0]}
            {parameter_templates[sector][1]}
            {parameter_templates[sector][2]}

            2. Include exact product details:
            - Quote specific rates, returns, or fees from documents
            - Reference exact product names and features
            - Use actual numbers and percentages from the text

            3. Link recommendations to parameters:
            - Match product risk levels to stated risk tolerance
            - Align investment terms with specified time horizons
            - Scale suggestions to match given amounts

            4. Cite Evidence:
            - Quote specific text from documents
            - Reference exact page numbers
            - Include specific performance metrics
            ... existing code for other sectors ...

            Format each strategy as:
            TITLE: [Product/Strategy Name from Document] - [Parameter-Specific Application]
            DESCRIPTION: [Detailed strategy linking parameters to specific document content]
            SOURCE: [PDF name and exact page number with relevant quote]
            """

        message_text = [
            {"role": "system", "content": f"You are an expert financial advisor generating detailed {sector} strategies."},
            {"role": "user", "content": strategy_prompt}
        ]

        logger.info(f"Sending prompt to Azure OpenAI for {sector} sector")
        try:
            completion = self.llm.chat.completions.create(
                model="RAG",
                messages=message_text,
                temperature=0.1,
                max_tokens=1000,
                top_p=0.85,
                frequency_penalty=0.7,
                presence_penalty=0.7
            )

            response = completion.choices[0].message.content
            logger.info(f"Received response from Azure OpenAI: {response[:200]}...")

            new_strategies = self.parse_strategies(response)
            logger.info(f"Parsed {len(new_strategies)} strategies from response")

            for strategy in new_strategies:
                self.strategy_history[sector].add(strategy['title'])

            if not new_strategies:
                logger.warning("No strategies generated from response")
                return {
                    "strategies": [],
                    "metadata": {
                        "sector": sector,
                        "message": f"Could not generate {sector} strategies from document content"
                    }
                }

            return {
                "strategies": new_strategies,
                "metadata": {
                    "sector": sector,
                    "source": "generated",
                    "strategies_generated": len(new_strategies),
                    "total_unique_strategies": len(self.strategy_history[sector])
                }
            }

        except Exception as e:
            logger.error(f"Error generating strategies: {str(e)}")
            raise





    def filter_unique_strategies(self, new_strategies: List[Dict], existing_strategies: List[Dict]) -> List[Dict]:
        unique_strategies = []
        existing_titles = {s['title'].lower() for s in existing_strategies}
        existing_descriptions = {s['description'].lower() for s in existing_strategies}

        for strategy in new_strategies:
            title_lower = strategy['title'].lower()
            desc_lower = strategy['description'].lower()

            if (title_lower not in existing_titles and
                desc_lower not in existing_descriptions):
                unique_strategies.append(strategy)
                existing_titles.add(title_lower)
                existing_descriptions.add(desc_lower)

        return unique_strategies





    def parse_strategies(self, response: str) -> List[Dict[str, Any]]:
        strategies = []
        sections = response.split("TITLE:")

        for section in sections[1:]:
            try:
                # Skip empty or malformed sections
                if not section.strip():
                    logger.warning("Empty section encountered during parsing.")
                    continue

                # Split section into description and source
                parts = section.split("DESCRIPTION:")
                if len(parts) != 2:
                    logger.warning("Section missing DESCRIPTION part.")
                    continue

                title_part = parts[0].strip()
                desc_source_parts = parts[1].split("SOURCE:")

                # Extract source and page information
                source = desc_source_parts[1].strip() if len(desc_source_parts) > 1 else "PDF Document"

                # Enhanced page number extraction
                page_num = None
                page_patterns = [
                    r'[Pp]age[s]?\s*[:#]?\s*(\d+)',
                    r'[Pp]\.\s*(\d+)',
                    r'\(p\.?\s*(\d+)\)',
                    r'\[p\.?\s*(\d+)\]',
                    r'@\s*(\d+)',
                    r'pg\.?\s*(\d+)',
                ]
                for pattern in page_patterns:
                    match = re.search(pattern, source, re.IGNORECASE)
                    if match:
                        page_num = match.group(1)
                        break

                # Extract PDF filename
                pdf_name = None
                if ".pdf" in source.lower():
                    filename_matches = re.finditer(r'([^/\\:\s]+\.pdf)', source, re.IGNORECASE)
                    for match in filename_matches:
                        pdf_name = re.sub(r'^[a-f0-9-]+_', '', match.group(1))  # Remove UUID prefixes
                        break

                # Fallback for page number
                try:
                    page_num = int(page_num) if page_num else 1
                except ValueError:
                    page_num = 1  # Default to page 1 if conversion fails

                # Format the source
                formatted_source = f"{pdf_name or 'PDF Document'}, page {page_num}"

                # Construct the strategy object
                strategy = {
                    "title": title_part,
                    "description": desc_source_parts[0].strip(),
                    "source": formatted_source,
                    "page_number": page_num,
                }

                strategies.append(strategy)
                logger.info(f"Parsed strategy: {strategy}")

            except Exception as e:
                logger.warning(f"Error parsing strategy section: {str(e)}")
                continue

        return strategies


    class SimpleVectorStore:
        def __init__(self, dimension):
            self.vectors = None
            self.dimension = dimension

        def add(self, vectors):
            vectors = np.array(vectors)
            if self.vectors is None:
                self.vectors = vectors
            else:
                self.vectors = np.vstack([self.vectors, vectors])

        def search(self, query_vector, k):
            similarities = cosine_similarity(query_vector.reshape(1, -1), self.vectors)[0]
            indices = np.argsort(similarities)[-k:][::-1]
            distances = 1 - similarities[indices]
            return distances, indices




    def query(self, question: str, sector: str) -> Dict[str, Any]:
        try:
            stored_params = getattr(self, 'user_parameters', {}).get(sector, {})
            logger.info(f"Using stored parameters for {sector}: {stored_params}")

            processed_query = self.preprocess_query(question)

            # Get and sort PDFs by creation time (most recent first)
            pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
            if not pdf_files:
                return {"strategies": [], "message": "No PDF documents available"}

            pdf_files.sort(key=lambda x: os.path.getctime(os.path.join(self.pdf_dir, x)), reverse=True)

            # Reset document processing
            all_documents = []
            processed_paths = set()
            self.pdf_cache = {}  # Clear cache to force reprocessing
            batch_size = 5

            # Process only the most recent PDFs
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.pdf_dir, pdf_file)
                file_hash = hashlib.md5(open(pdf_path,'rb').read()).hexdigest()

                if file_hash not in processed_paths:
                    processed_paths.add(file_hash)
                    # Force new processing by bypassing cache
                    pdf_info = self.process_pdf(pdf_path)

                    if pdf_info and pdf_info["chunks"]:
                        for chunk in pdf_info["chunks"]:
                            sub_chunks = self.chunk_document(chunk, chunk_size=4000)

                            for sub_chunk in sub_chunks:
                                doc = Document(
                                    page_content=sub_chunk,
                                    metadata={
                                        "source": pdf_file,
                                        "sector": sector,
                                        "page": pdf_info["page_numbers"].get(chunk, 1),
                                        "doc_id": pdf_info["id"],
                                        "upload_time": os.path.getctime(pdf_path)
                                    }
                                )
                                all_documents.append(doc)

            if not all_documents:
                return {"strategies": [], "message": "No content extracted from PDFs"}

            # Sort documents by upload time to prioritize recent content
            all_documents.sort(key=lambda x: x.metadata["upload_time"], reverse=True)

            # Generate embeddings and process documents
            logger.info("Generating embeddings for query")
            query_embedding = self.batch_generate_embeddings([processed_query])[0]

            doc_embeddings = []
            batch_size = 32
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                texts = [doc.page_content for doc in batch]
                batch_embeddings = self.batch_generate_embeddings(texts)
                doc_embeddings.extend(batch_embeddings)

            # Compute similarities and get relevant documents
            similarities = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            k = min(50, len(similarities))
            top_indices = np.argsort(similarities)[-k:][::-1]
            top_docs = [all_documents[i] for i in top_indices]

            # Grade and process relevant documents
            relevant_docs = []
            for i in range(0, len(top_docs), batch_size):
                batch = top_docs[i:i + batch_size]
                graded_batch = self.grade_documents(question, batch, sector)
                relevant_docs.extend(graded_batch)

            logger.info(f"Found {len(relevant_docs)} relevant documents")

            # Generate context chunks
            context_chunks = []
            current_chunk = []
            current_size = 0
            max_chunk_size = 6000

            for doc in relevant_docs:
                doc_text = f"From {doc.metadata['source']} (Page {doc.metadata['page']}):\n{doc.page_content}\n"
                doc_size = len(self.tokenizer.encode(doc_text))

                if current_size + doc_size > max_chunk_size:
                    context_chunks.append("\n".join(current_chunk))
                    current_chunk = [doc_text]
                    current_size = doc_size
                else:
                    current_chunk.append(doc_text)
                    current_size += doc_size

            if current_chunk:
                context_chunks.append("\n".join(current_chunk))

            # Generate strategies from context chunks
            all_strategies = []
            for context in context_chunks:
                result = self.generate_strategies(context, sector, stored_params)
                if 'strategies' in result:
                    all_strategies.extend(result['strategies'])

            return {
                "strategies": all_strategies,
                "metadata": {
                    "sector": sector,
                    "parameters_used": stored_params,
                    "documents_processed": len(all_documents),
                    "relevant_docs": len(relevant_docs),
                    "processing_time": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return {"strategies": [], "error": str(e)}





    # Additional utility methods remain the same as in your original code
    def extract_sector_parameters(self, question: str) -> Dict[str, Any]:
        """Extract sector-specific parameters from the query."""
        params = {}
        for sector, parameters in self.sector_params.items():
            for param_name, param_type in parameters.items():
                # Match patterns like "Investment Amount ($): 99998"
                pattern = rf"{param_name}\s*[\(\$)]*[:=]?\s*([\d.]+)"
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    try:
                        params[param_name] = param_type(match.group(1).replace(",", ""))
                    except ValueError:
                        self.logger.warning(f"Could not convert {param_name} to {param_type}")
        self.logger.info(f"Extracted parameters: {params}")
        return params




    def grade_documents(self, question: str, documents: List[Dict], sector: str) -> List[Dict]:
        logger.info(f"Starting document grading for {len(documents)} documents")
        filtered_docs = []
        stored_params = getattr(self, 'user_parameters', {}).get(sector, {})

        # Convert SystemMessage to string representation for JSON serialization
        system_prompt = {
            "role": "system",
            "content": f"""
        You are a specialized financial document grader evaluating content relevance for {sector} sector strategies.

        Consider these specific parameters:
        {json.dumps(stored_params, indent=2)}

        Score documents based on:
        1. Content relevant to these exact parameter values
        2. Insights that can be applied within these parameter constraints
        3. Metrics or benchmarks that align with these parameters
        """
        }

        for idx, doc in enumerate(documents):
            try:
                logger.info(f"Grading document {idx+1}/{len(documents)}")
                content = doc.page_content
                metadata = doc.metadata

                logger.info(f"Document content preview: {content[:200]}...")

                # Use dictionary format instead of SystemMessage
                result = self.retrieval_grader.invoke({
                    "question": question,
                    "document": content,
                    "system": system_prompt["content"],  # Pass content directly
                    "metadata": {
                        "source": metadata.get("source"),
                        "page": metadata.get("page")
                    }
                })

                if result and isinstance(result, str):
                    binary_score = "yes" if "yes" in result.lower() else "no"
                    if binary_score == "yes":
                        filtered_docs.append(doc)
                        logger.info(f"Document graded as relevant: {len(content)} chars")

            except Exception as e:
                logger.warning(f"Grading error handled: {e}")
                if doc.page_content.strip():
                    filtered_docs.append(doc)

        logger.info(f"Graded {len(documents)} docs, {len(filtered_docs)} relevant")
        return filtered_docs


    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file with page tracking"""
        try:
            text_chunks = []
            table_chunks = []

            with pdfplumber.open(pdf_path) as pdf:
                batch_size = 5
                total_pages = len(pdf.pages)

                for batch_start in range(0, total_pages, batch_size):
                    batch_end = min(batch_start + batch_size, total_pages)
                    batch = pdf.pages[batch_start:batch_end]

                    for page_idx, page in enumerate(batch, start=batch_start):
                        current_page = page_idx + 1

                        # Extract text with page markers
                        if page_text := page.extract_text():
                            text_chunks.append(f"[PAGE_START_{current_page}]\n{page_text}\n[PAGE_END_{current_page}]")

                        # Extract tables with page markers
                        for table in page.extract_tables():
                            if table and any(any(row) for row in table):
                                df = pd.DataFrame(table[1:], columns=table[0] or [f"Col{i}" for i in range(len(table[0]))])
                                table_chunks.append(f"[TABLE_PAGE_{current_page}]\n{df.to_string()}\n[TABLE_END_{current_page}]")

                        page.flush_cache()

                all_content = "\n".join(filter(None, text_chunks + table_chunks))

                if not all_content.strip():
                    raise ValueError("No text content extracted from PDF")

                self.logger.info(f"Extracted {len(all_content)} chars and {len(table_chunks)} tables")
                return all_content

        except Exception as e:
            self.logger.error(f"PDF extraction error: {str(e)}")
            raise


    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        try:
            logger.info(f"Starting PDF processing for: {pdf_path}")
            if pdf_path in self.pdf_cache:
                logger.info(f"Found cached PDF content for: {pdf_path}")
                return self.pdf_cache[pdf_path]

            self.logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")

            pdf_id = str(uuid.uuid4())
            pdf_name = os.path.basename(pdf_path)

            # Enhanced text extraction with structure preservation
            text_chunks = []
            page_numbers = {}

            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"PDF opened successfully. Total pages: {len(pdf.pages)}")
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}")

                    # Extract text with formatting
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        logger.debug(f"Page {page_num} raw text:\n{page_text[:500]}...")
                        # Add section markers
                        marked_text = f"[SECTION_START]Page {page_num}\n{page_text}\n[SECTION_END]"
                        text_chunks.append(marked_text)
                        logger.info(f"Extracted {len(page_text)} chars from page {page_num}")

                    # Extract tables as structured text
                    for table in page.extract_tables():
                        logger.info(f"Found {len(table)} tables on page {page_num}")
                        if table:
                            df = pd.DataFrame(table)
                            table_text = f"[TABLE_START]Page {page_num}\n{df.to_string()}\n[TABLE_END]"
                            text_chunks.append(table_text)

            full_text = "\n".join(text_chunks)

            # Improved chunking with semantic boundaries
            chunks = self.chunk_document(full_text, chunk_size=1000)

            # Track page numbers for each chunk
            for chunk in chunks:
                page_match = re.search(r'Page (\d+)', chunk)
                page_num = int(page_match.group(1)) if page_match else 1
                page_numbers[chunk] = page_num

            logger.info(f"PDF processing complete. Total chunks: {len(text_chunks)}")


            pdf_info = {
                "id": pdf_id,
                "filename": pdf_name,
                "content": full_text,
                "chunks": chunks,
                "page_numbers": page_numbers,
                "processed_at": datetime.utcnow().isoformat()
            }

            self.pdf_cache[pdf_path] = pdf_info
            return pdf_info

        except Exception as e:
            self.logger.error(f"PDF processing error for {pdf_path}: {str(e)}")
            return None

    def add_document(self, text: str, name: str = None) -> bool:
        if not text.strip():
            logger.error("Empty document content.")
            return False

        chunks = self.chunk_document(text)
        embeddings = self.generate_bert_embeddings(chunks)

        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            document_id = str(uuid.uuid4())
            chunk_name = f"{name or 'Document'} (Part {i + 1})" if len(chunks) > 1 else name or f"Document {self.document_count}"
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": chunk_name,
                    "embedding": embedding
                }
            )
            documents.append(doc)
            self.documents.append({
                'id': document_id,
                'name': chunk_name,
                'content': chunk,
                'embedding': embedding
            })
            self.document_count += 1

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=list(zip(chunks, embeddings)),
                embedding=self.embeddings,  # Keep for compatibility
                metadatas=[{"source": d.metadata["source"]} for d in documents]
            )
        else:
            self.vectorstore.add_embeddings(
                text_embeddings=list(zip(chunks, embeddings)),
                metadatas=[{"source": d.metadata["source"]} for d in documents]
            )

        self.save_documents()
        logger.info(f"Successfully added document: {name or document_id} in {len(chunks)} chunks")
        return True

    def save_documents(self) -> None:
            with open(os.path.join(self.persist_dir, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)

    def load_documents(self) -> None:
            documents_path = os.path.join(self.persist_dir, "documents.pkl")
            if os.path.exists(documents_path):
                with open(documents_path, "rb") as f:
                    self.documents = pickle.load(f)
                self.document_count = len(self.documents)
                logger.info(f"Loaded {self.document_count} documents from {documents_path}")

    def chunk_document(self, text: str, chunk_size: int = 8000) -> List[str]:
        # Use Azure OpenAI's tokenization
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # Split on semantic boundaries
        sections = text.split('\n\n')
        chunks, current_chunk, current_size = [], [], 0

        for section in sections:
            # Preserve paragraph structure
            paragraphs = section.split('\n')

            for paragraph in paragraphs:
                # Get token count for current paragraph
                paragraph_tokens = len(encoding.encode(paragraph))

                # Check if adding paragraph exceeds chunk size
                if current_size + paragraph_tokens > chunk_size:
                    # Complete current chunk
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                    current_chunk = [paragraph]
                    current_size = paragraph_tokens
                else:
                    current_chunk.append(paragraph)
                    current_size += paragraph_tokens

                # Handle extremely long paragraphs
                if paragraph_tokens > chunk_size:
                    words = paragraph.split()
                    temp_chunk, temp_size = [], 0

                    for word in words:
                        word_tokens = len(encoding.encode(word))
                        if temp_size + word_tokens > chunk_size:
                            chunks.append(' '.join(temp_chunk))
                            temp_chunk, temp_size = [word], word_tokens
                        else:
                            temp_chunk.append(word)
                            temp_size += word_tokens

                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))

        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks


    def preprocess_query(self, query: str) -> str:
        query = re.sub(r"[^\w\s]", " ", query).lower()
        query = " ".join(query.split())
        return query



    def normalize_url(self, url: str) -> str:
        """Normalize URL for consistent comparison."""
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

    def make_request_with_retries(self, url: str, retries: int = 3) -> requests.Response:
        """Make HTTP requests with retries and exponential backoff."""
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=20, verify=False)
                if response.status_code == 200:
                    return response
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                sleep(2 ** attempt + random.random())
        raise Exception(f"Failed to fetch {url} after {retries} retries")

    def process_url_batch(self, url: str, shared_data: Dict, depth: int = 0, max_depth: int = 2) -> List[Any]:
        """Process single URL and its internal links recursively with enhanced source tracking"""
        try:
            normalized_url = self.normalize_url(url)
            if normalized_url in shared_data['visited_urls'] or depth > max_depth:
                return []

            response = self.make_request_with_retries(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            shared_data['visited_urls'].add(normalized_url)

            # Enhanced metadata extraction
            title = soup.title.string if soup.title else ""
            meta_description = " ".join(meta.get('content', '') for meta in soup.find_all('meta', {'name': 'description'}))
            h1_content = " ".join(h1.text for h1 in soup.find_all('h1'))
            preview = f"{title} {meta_description} {h1_content}"

            # More detailed relevance check
            if not preview.strip():
                return []

            # Extract page-specific identifiers
            current_path = urlparse(url).path
            page_id = soup.find('body').get('id', '') if soup.find('body') else ''

            # Process content with enhanced tracking
            loader = WebBaseLoader(url)
            docs = loader.load()
            processed_docs = self.process_documents(docs)

            # Enhanced metadata with exact source tracking
            for doc in processed_docs:
                doc.metadata.update({
                    "source_url": url,
                    "exact_path": current_path,
                    "page_title": title,
                    "page_id": page_id,
                    "domain": urlparse(url).netloc,
                    "timestamp": datetime.utcnow().isoformat(),
                    "depth": depth,
                    "content_type": "main" if depth == 0 else "internal",
                    "breadcrumb": f"{url} > {title}"
                })

            # Intelligent internal link processing
            if depth < max_depth:
                base_domain = urlparse(url).netloc
                internal_links = []

                # Prioritize navigation and content links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    if urlparse(full_url).netloc == base_domain:
                        # Score links based on location and content
                        score = 0
                        if link.parent.name in ['nav', 'header', 'main']:
                            score += 2
                        if any(term in href.lower() for term in ['about', 'product', 'service', 'detail']):
                            score += 1
                        internal_links.append((full_url, score))

                # Sort by score and take top 5
                sorted_links = sorted(internal_links, key=lambda x: x[1], reverse=True)
                for internal_url, _ in sorted_links[:5]:
                    internal_docs = self.process_url_batch(internal_url, shared_data, depth + 1, max_depth)
                    processed_docs.extend(internal_docs)

            return processed_docs

        except Exception as e:
            self.logger.error(f"Error processing {url}: {e}")
            return []




    def generate_embeddings_for_url(self, urls: List[str]) -> List[Any]:
        shared_data = {
            'visited_urls': set(),
            'processed_docs': []
        }

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {
                executor.submit(self.process_url_batch, url, shared_data): url
                for url in urls
            }

            all_docs = []
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    docs = future.result()
                    if docs:
                        all_docs.extend(docs)
                        self.logger.info(f"Processed {url}: got {len(docs)} documents")
                except Exception as e:
                    self.logger.error(f"Failed processing {url}: {e}")

        if all_docs:
            texts = [doc.page_content for doc in all_docs]
            batch_size = 32
            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.extend(batch_embeddings)

            # Update document metadata with embeddings directly
            for doc, embedding in zip(all_docs, embeddings):
                doc.metadata['embedding'] = embedding

        self.logger.info(f"Completed processing {len(urls)} URLs, got {len(all_docs)} total documents")
        return all_docs

    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings efficiently in batches using DistilBERT"""
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # Get model outputs
                outputs = self.model(**inputs)

                # Use [CLS] token embeddings as document embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.extend(batch_embeddings)

        return embeddings


    def get_more_info(self, strategy: dict, urls: list):
        try:
            # Initialize DistilBERT components if not already initialized
            if not hasattr(self, 'tokenizer') or not hasattr(self, 'model'):
                self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
                self.model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")
                self.model.eval()

            # Generate embeddings using DistilBERT
            def generate_embedding(text):
                with torch.no_grad():
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    return outputs.last_hidden_state[:, 0, :].numpy()[0]

            url_docs = self.generate_embeddings_for_url(urls)

            # Generate embeddings for documents
            embeddings = []
            texts = []
            for doc in url_docs:
                embedding = generate_embedding(doc.page_content)
                doc.metadata['embedding'] = embedding
                embeddings.append(embedding)
                texts.append(doc.page_content)

            # Generate query embedding
            query_embedding = generate_embedding(strategy['title'])

            # Simple vector similarity search
            similarities = []
            for doc_embedding in embeddings:
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append(similarity)

            # Get top k most similar documents
            k = min(100, len(similarities))
            top_indices = np.argsort(similarities)[-k:][::-1]
            relevant_docs = [url_docs[i] for i in top_indices]
            similarity_scores = [similarities[i] for i in top_indices]

            sources = [{
                "url": doc.metadata.get("source_url"),
                "exact_path": doc.metadata.get("exact_path"),
                "page_title": doc.metadata.get("page_title"),
                "content_preview": doc.page_content[:200],
                "depth": doc.metadata.get("depth"),
                "breadcrumb": doc.metadata.get("breadcrumb"),
                "timestamp": doc.metadata.get("timestamp")
            } for doc in relevant_docs[:5]]

            raw_analysis = self.summarize_text(
                text="\n".join(doc.page_content for doc in relevant_docs),
                context=strategy['title'],
                strategy=strategy
            )

            formatted_analysis = self.preprocess_strategy_narrative(raw_analysis)

            return {
                "analysis": {
                    "title": strategy['title'],
                    "detailed_analysis": formatted_analysis,
                    "source_count": len(relevant_docs),
                    "confidence_score": float(np.mean(similarity_scores))
                },
                "sources": sources,
                "metadata": {
                    "total_chunks_processed": len(url_docs),
                    "relevant_chunks_found": len(relevant_docs),
                    "processing_timestamp": datetime.utcnow().isoformat(),
                    "source_domains": list(set(doc.metadata.get("domain") for doc in relevant_docs)),
                    "embedding_model": "distilbert-base-uncased"
                }
            }

        except Exception as e:
            self.logger.error(f"Error in get_more_info: {str(e)}", exc_info=True)
            raise e




    def get_more_info(self, strategy: dict, urls: list):
        try:
            # Initialize components if needed
            if not hasattr(self, 'tokenizer') or not hasattr(self, 'model'):
                self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
                self.model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")
                self.model.eval()

            # Step 1: Get documents from URLs
            url_docs = self.generate_embeddings_for_url(urls)
            logger.info(f"Fetched {len(url_docs)} documents from URLs.")

            # Step 2: Grade documents for relevance
            graded_docs = []
            batch_size = 32

            # Create a strategy-specific grading prompt
            system_prompt = {
                "role": "system",
                "content": f"""
                You are evaluating web content relevance for the strategy: {strategy['title']}

                Grade documents based on:
                1. Direct relevance to strategy implementation
                2. Supporting evidence and examples
                3. Current market data and statistics
                4. Expert opinions and analysis
                5. Implementation guidelines
                """
            }

            # Grade documents in batches
            for i in range(0, len(url_docs), batch_size):
                batch = url_docs[i:i + batch_size]
                logger.info(f"Processing batch {i // batch_size + 1} with {len(batch)} documents.")

                for doc in batch:
                    try:
                        result = self.retrieval_grader.invoke({
                            "question": strategy['title'],
                            "document": doc.page_content,
                            "system": system_prompt["content"],
                            "metadata": doc.metadata
                        })

                        if result and isinstance(result, str) and "yes" in result.lower():
                            graded_docs.append(doc)

                    except Exception as grade_error:
                        logger.warning(f"Error grading document: {grade_error}")

            logger.info(f"Graded {len(url_docs)} documents, {len(graded_docs)} passed relevance filtering.")

            # Step 3: Generate embeddings for graded documents
            embeddings, texts = [], []

            def generate_embedding(text):
                with torch.no_grad():
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    return outputs.last_hidden_state[:, 0, :].numpy()[0]

            for doc in graded_docs:
                try:
                    embedding = generate_embedding(doc.page_content)
                    doc.metadata['embedding'] = embedding
                    embeddings.append(embedding)
                    texts.append(doc.page_content)
                except Exception as embed_error:
                    logger.warning(f"Error generating embedding for document: {embed_error}")

            # Step 4: Generate query embedding and compute document similarity
            query_embedding = generate_embedding(strategy['title'])
            logger.info("Generated query embedding for strategy title.")

            similarities = []
            for doc_embedding in embeddings:
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append(similarity)

            # Step 5: Get top relevant documents
            k = min(100, len(similarities))
            top_indices = np.argsort(similarities)[-k:][::-1]
            relevant_docs = [graded_docs[i] for i in top_indices]
            similarity_scores = [similarities[i] for i in top_indices]

            logger.info(f"Selected top {len(relevant_docs)} relevant documents.")

            # Step 6: Extract source information
            sources = [
                {
                    "url": doc.metadata.get("source_url"),
                    "exact_path": doc.metadata.get("exact_path"),
                    "page_title": doc.metadata.get("page_title"),
                    "content_preview": doc.page_content[:200],
                    "depth": doc.metadata.get("depth"),
                    "breadcrumb": doc.metadata.get("breadcrumb"),
                    "timestamp": doc.metadata.get("timestamp"),
                    "relevance_score": float(similarities[i])
                }
                for i, doc in enumerate(relevant_docs[:5])
            ]

            # Step 7: Generate strategy analysis from relevant documents
            combined_text = "\n".join(doc.page_content for doc in relevant_docs)
            raw_analysis = self.summarize_text(
                text=combined_text,
                context=strategy['title'],
                strategy=strategy
            )

            formatted_analysis = self.preprocess_strategy_narrative(raw_analysis)

            # Step 8: Return results
            return {
                "analysis": {
                    "title": strategy['title'],
                    "detailed_analysis": formatted_analysis,
                    "source_count": len(relevant_docs),
                    "confidence_score": float(np.mean(similarity_scores))
                },
                "sources": sources,
                "metadata": {
                    "total_documents": len(url_docs),
                    "graded_relevant": len(graded_docs),
                    "final_relevant": len(relevant_docs),
                    "processing_timestamp": datetime.utcnow().isoformat(),
                    "source_domains": list(set(doc.metadata.get("domain") for doc in relevant_docs)),
                    "embedding_model": "finbert-tone",
                    "grading_stats": {
                        "processed": len(url_docs),
                        "passed_grading": len(graded_docs),
                        "top_relevant": len(relevant_docs)
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error in get_more_info: {str(e)}", exc_info=True)
            raise e









    def summarize_text(self, text: str, context: str, strategy: Dict[str, str]) -> str:
        strategy_title = strategy.get('title', '')
        strategy_desc = strategy.get('description', '')

        # Single paragraph format prompt
        formatted_prompt = f"""
        Based on the strategy "{strategy_title}", create a comprehensive single-paragraph analysis that:

        1. Opens with a clear statement of the strategy's core purpose and value proposition
        2. Seamlessly integrates how it addresses specific parameters and benefits users
        3. Includes concrete implementation steps and expected outcomes
        4. Identifies precise target audience segments
        5. Concludes with 2-3 most critical action items

        Focus on quality over quantity. Be specific, practical, and concise.

        SOURCE MATERIAL:
        {text}

        STRATEGY CONTEXT:
        {context}
        """

        try:
            response = self.azure_client.chat.completions.create(
                model="RAG",
                messages=[{
                    "role": "system",
                    "content": """You are a financial strategy expert crafting concise, high-impact analyses.
                                Prioritize specificity and actionable insights.
                                Focus on measurable outcomes and practical implementation.
                                Maintain a professional, authoritative tone."""
                },
                {
                    "role": "user",
                    "content": formatted_prompt
                }],
                temperature=0.1,  # Lower temperature for more focused output
                max_tokens=1000,  # Reduced tokens for conciseness
                top_p=0.85
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Analysis generation error: {e}")
            return f"Unable to generate analysis for {context}"
    def preprocess_strategy_narrative(self, text: str) -> str:
        # Single paragraph formatting
        text = text.strip()

        # Remove any existing headers or formatting
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Clean up any remaining formatting artifacts
        text = text.replace('*', '')
        text = text.replace('#', '')
        text = text.replace('-', '')

        # Ensure proper sentence spacing
        text = re.sub(r'\.(?=\S)', '. ', text)

        # Capitalize first letter of sentences
        text = '. '.join(s.capitalize() for s in text.split('. '))

        # Add proper paragraph indentation
        text = text.replace('. ', '.\n\n')

        # Final cleanup
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text





