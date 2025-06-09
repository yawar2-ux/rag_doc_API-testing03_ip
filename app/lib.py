import os
from pprint import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List
from dotenv import load_dotenv

load_dotenv()
#
os.environ["OPENAI_API_KEY"] = os.getenv(
    "OPENAI_API_KEY", "sk-proj-CWh0Z3737a33EnfpNS3FT3BlbkFJW5cFS6GFYtz6uLDidvzL"
)
os.environ["TAVILY_API_KEY"] = os.getenv(
    "TAVILY_API_KEY", "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"
)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)
os.environ["LANGCHAIN_API_KEY"] = os.getenv(
    "LANGCHAIN_API_KEY", "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"
)

# urls = [
#     "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=10005&Mode=0",
#     "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566",
#     "https://en.wikipedia.org/wiki/Foreign_Account_Tax_Compliance_Act",
#     "https://towardsdatascience.com/string-matching-with-fuzzywuzzy-e982c61f8a84"
# ]
# folder = ''


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


class RagAgent:
    def __init__(self):
        self.app = None
        self.vectorstore = None
        self.retriever = None
        self.retrieval_grader = None
        # self.web_search = None
        self.question_rewriter = None
        self.web_search_tool = None
        self.rag_chain = None
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    def retrieve(self, state):
        """
        Retrieve documents
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}

    def generate(self, state):
        """
        Generate answer
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
        }

    def transform_query(self, state):
        """
        Transform the query to produce a better question.
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(self, state):
        """
        Web search based on the re-phrased question.
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): Updates documents key with appended web results
        """
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        return {"documents": documents, "question": question}

    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.
        Args:
            state (dict): The current graph state
        Returns:
            str: Binary decision for next node to call
        """
        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        web_search = state["web_search"]
        filtered_documents = state["documents"]
        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def generate_embeddings_for_documents(self, paths):
      documents = []

      def process_file(file_path):
        try:
            loader = PyPDFLoader(file_path)
            doc_content = loader.load()
            documents.extend(doc_content)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

      def process_directory(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    process_file(file_path)

      for path in paths:
        path = os.path.abspath(path)  # Ensure absolute path
        path = os.path.normpath(path)  # Normalize path to handle mixed slashes

        if os.path.isdir(path):
            process_directory(path)
        elif os.path.isfile(path) and path.endswith(".pdf"):
            process_file(path)
        else:
            print(f"Path does not exist or is not a PDF file: {path}")
      try:
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)
        print("--------len(documents)----", len(documents))
      except Exception as e:
        print(f"Error splitting documents: {e}")

      return documents

    def generate_embeddings_for_url(self, urls):
        print("read the url")
        doc_splits = []
        if len(urls):
            docs = [WebBaseLoader(url).load() for url in urls]
            docs_list = [item for sublist in docs for item in sublist]
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=250, chunk_overlap=0
            )
            doc_splits = text_splitter.split_documents(docs_list)
            print("-----------len(doc_splits)----", len(doc_splits))
        return doc_splits

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def preprcessing(self, urls, folder):
        doc_data = self.generate_embeddings_for_documents(folder)
        urls_data = self.generate_embeddings_for_url(urls)
        doc_list = []
        if len(urls_data) > 0 and len(doc_data) > 0:
            urls_data.extend(doc_data)
            doc_list = urls_data
        elif len(urls_data) > 0:
            doc_list = urls_data
        elif len(doc_data) > 0:
            doc_list = doc_data

        self.vectorstore = Chroma.from_documents(
            documents=doc_list,
            collection_name="rag-chroma",
            embedding=OpenAIEmbeddings(),
            persist_directory="db/",
        )
        self.retriever = self.vectorstore.as_retriever()

        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {question}",
                ),
            ]
        )

        self.retrieval_grader = grade_prompt | self.structured_llm_grader
        question = "agent memory"
        docs = self.retriever.get_relevant_documents(question)
        doc_txt = docs[1].page_content
        self.retrieval_grader.invoke({"question": question, "document": doc_txt})
        prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = prompt | self.llm | StrOutputParser()
        generation = self.rag_chain.invoke({"context": docs, "question": question})
        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
             for web search. Look at the input and try to reason about the underlying sematic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        self.question_rewriter = re_write_prompt | self.llm | StrOutputParser()
        self.question_rewriter.invoke({"question": question})

        ### Search
        self.web_search_tool = TavilySearchResults(k=3)

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query
        workflow.add_node("web_search_node", self.web_search)  # web search

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)

        # Compile
        self.app = workflow.compile()
        return self.app
