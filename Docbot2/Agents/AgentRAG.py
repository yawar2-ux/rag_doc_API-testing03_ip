"""
Agent RAG Module: Handles document processing and question answering using RAG.
"""

from typing import List, Dict, Any
from ..Services.RAG import RAGService # Corrected relative import
from ..Model.LLM import Groq_Model, get_llm_instance # Corrected relative import
from .MemoryAgent import MemoryAgent # Corrected relative import
from ..Services.thumbsup import thumbsup_service # Import the singleton instance
from ..Services.thumbsdown import thumbsdown_service 
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_groq import ChatGroq

class AgentRAG:
    def __init__(self):
        """Initialize the RAG agent with services."""
        self.rag_service = RAGService()
        self.memory_agent = MemoryAgent()
        self.thumbsup_service = thumbsup_service 
        self.thumbsdown_service = thumbsdown_service
        self.last_response_type = {}  # Track what type of response was given to each user
    
    def _is_followup_question(self, message: str) -> bool:
        """Check if the message is a follow-up question."""
        followup_phrases = [
            "are you sure", "really", "how do you know", "why not", "but why", 
            "can you check", "double check", "check again", "try again",
            "you're wrong", "you are wrong", "incorrect", "not right",
            "i don't believe", "that's wrong", "are you certain",
            "is it true", "is that true", "is this true", "really true",
            "confirm that", "verify that", "double check that"
        ]
        return any(phrase in message.lower() for phrase in followup_phrases)
    
    def _should_use_memory_only(self, user_id: str, message: str) -> bool:
        """Determine if we should only use memory context, not vector DB."""
        # If last response was "I don't know" and this is a follow-up, use memory only
        last_response_type = self.last_response_type.get(user_id, "normal")
        is_followup = self._is_followup_question(message)
        
        # Also check if this is a very short generic question that likely isn't domain-specific
        short_generic_phrases = ["hi", "hello", "yes", "no", "ok", "okay", "thanks", "thank you"]
        is_short_generic = message.lower().strip() in short_generic_phrases
        
        return (last_response_type == "no_knowledge" and is_followup) or is_short_generic
    
    def process_query(self, user_id: str, message: str, hybrid_alpha: float = 0.7, use_reranking: bool = True, temperature: float = 0.36, max_tokens: int = 1024, top_p: float = 1.0, thumbsup_score_threshold: float = 0.78) -> Dict[str, Any]:
            """Process a user query and generate a response using RAG."""

            # 1. Check thumbsdown collection
            thumbsdown_collection_name = f"{user_id}_thumbsdown"
            thumbsdown_results = self.thumbsdown_service.query_collection(
                collection_name=thumbsdown_collection_name,
                query_text=message,
                k=1,
                score_threshold=0.0  # Any match means thumbsdown
            )

            if thumbsdown_results:
                # Regenerate response
                print("Regenerating response due to thumbsdown.")
                self.thumbsdown_service.clear_collection(thumbsdown_collection_name)  # Clear the thumbsdown

                # Get chat history for context
                chat_history = self.memory_agent.get_chat_history(user_id)

                # Retrieve relevant documents with enhanced retrieval
                docs = self.rag_service.retrieve_relevant_docs(
                    message,
                    k=5,
                    hybrid_alpha=hybrid_alpha,
                    use_reranking=use_reranking
                )

                # Prepare context from retrieved documents
                context = self.rag_service.summarize_context(docs)

                # Get a configured LLM instance
                llm_instance = get_llm_instance(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )

                # Create the prompt template
                prompt_template = PromptTemplate.from_template("""
                    You are a knowledgeable assistant that answers questions strictly using the provided context and conversation history.
                    
                    First, determine if the user's question is related to any information in the provided context:
                    - If the context contains relevant information to answer the question, use that information to provide an accurate response.
                    - If the context is empty or contains NO information related to the user's question, respond ONLY with:
                      "I don't know. I'm sorry, I don't have enough information to answer that."
                    
                    Do not mention or describe the unrelated content in your response.
                    Do not invent or fabricate any information.

                    Previous conversation:
                    {chat_history}

                    User question:
                    {question}

                    Relevant information:
                    {context}

                    Answer:
                    """)

                # Create runnable chain using the pipe operator
                rag_chain = prompt_template | llm_instance

                # Run the chain
                response = rag_chain.invoke({
                    "question": message,
                    "context": context,
                    "chat_history": chat_history
                })

                # Save to memory and extract sources
                response_text = response.content if hasattr(response, 'content') else str(response)
                self.memory_agent.save_to_memory(user_id, message, response_text)
                sources = self.rag_service.extract_source_references(response_text, context)

                # Include retrieval method info in debug info
                debug_info = {
                    "retrieval_method": "hybrid" if hybrid_alpha < 1.0 else "semantic",
                    "hybrid_alpha": hybrid_alpha,
                    "reranking_used": use_reranking and self.rag_service.cross_encoder is not None,
                    "regenerated": True  # Add a flag to indicate regeneration
                }

                return {
                    "response": response_text,
                    "sources": sources,
                    "debug_info": debug_info
                }

            # 2. Check thumbsup collection
            thumbsup_collection_name = f"{user_id}_thumbsup"
            # Using a score threshold (L2 distance). Lower is better. 0.5 is moderately similar.
            # For very confident matches from thumbsup, a lower threshold like 0.2-0.3 might be better.
            # Let's use the default 0.5 from thumbsup_service.query_collection for now.
            # Updated: Explicitly setting a more lenient threshold.
            # Updated again: Using the passed thumbsup_score_threshold parameter.
            thumbsup_results = self.thumbsup_service.query_collection(
                collection_name=thumbsup_collection_name,
                query_text=message,
                k=1,
                score_threshold=thumbsup_score_threshold # Use the passed parameter
            )

            if thumbsup_results:
                # Found a relevant item in the user's liked content
                liked_content = thumbsup_results[0]['content']
                liked_score = thumbsup_results[0]['score']

                # Save to memory
                self.memory_agent.save_to_memory(user_id, message, liked_content)

                return {
                    "response": liked_content,
                    "sources": [{
                        "filename": "From Your Liked Content",
                        "page_number": "N/A"  # Or include metadata if available and relevant
                    }],
                    "debug_info": {
                        "retrieval_method": "thumbs_up_collection",
                        "collection_name": thumbsup_collection_name,
                        "score": liked_score,
                        "query_used_for_thumbsup": message,
                        "thumbsup_score_threshold_used": thumbsup_score_threshold # Add threshold to debug
                    }
                }

            # 3. Main RAG processing with smart routing
            chat_history = self.memory_agent.get_chat_history(user_id)
            use_memory_only = self._should_use_memory_only(user_id, message)
            
            if use_memory_only:
                # For follow-ups to "I don't know" responses, use memory-only prompt
                llm_instance = get_llm_instance(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                
                prompt_template = PromptTemplate.from_template("""
                    You are a helpful assistant. Based on our previous conversation, respond to the user's follow-up question.
                    
                    If your previous response was "I don't know" about a topic, maintain consistency:
                    - Acknowledge the follow-up politely
                    - Reaffirm that you don't have information on that specific topic
                    - Do not introduce new information from other sources
                    
                    Previous conversation:
                    {chat_history}
                    
                    User's follow-up question:
                    {question}
                    
                    Response:
                    """)
                
                rag_chain = prompt_template | llm_instance
                response = rag_chain.invoke({
                    "question": message,
                    "chat_history": chat_history
                })
                
                response_text = response.content if hasattr(response, 'content') else str(response)
                self.memory_agent.save_to_memory(user_id, message, response_text)
                
                # Track response type
                self.last_response_type[user_id] = "memory_followup"
                
                return {
                    "response": response_text,
                    "sources": [],
                    "debug_info": {
                        "retrieval_method": "memory_only_followup",
                        "reason": "Follow-up to no-knowledge response"
                    }
                }
            
            # Standard RAG processing
            docs = self.rag_service.retrieve_relevant_docs(
                message,
                k=5,
                hybrid_alpha=hybrid_alpha,
                use_reranking=use_reranking
            )

            context = self.rag_service.summarize_context(docs)
            
            # Determine if context is relevant
            context_is_relevant = self._is_context_relevant(context, message)
            
            llm_instance = get_llm_instance(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )

            if context_is_relevant:
                # Use full RAG prompt with context and memory
                prompt_template = PromptTemplate.from_template("""
                    You are a knowledgeable assistant that answers questions using the provided context and conversation history.
                    
                    Use the relevant information below to answer the question accurately.
                    If the context doesn't contain enough information, say so clearly.

                    Previous conversation:
                    {chat_history}

                    User question:
                    {question}

                    Relevant information:
                    {context}

                    Answer:
                    """)
                
                response = (prompt_template | llm_instance).invoke({
                    "question": message,
                    "context": context,
                    "chat_history": chat_history
                })
                
                response_text = response.content if hasattr(response, 'content') else str(response)
                self.memory_agent.save_to_memory(user_id, message, response_text)
                sources = self.rag_service.extract_source_references(response_text, context)
                
                # Track response type
                self.last_response_type[user_id] = "normal"
                
                debug_info = {
                    "retrieval_method": "hybrid" if hybrid_alpha < 1.0 else "semantic",
                    "hybrid_alpha": hybrid_alpha,
                    "reranking_used": use_reranking and self.rag_service.cross_encoder is not None,
                    "context_relevant": True
                }
                
            else:
                # No relevant context - use memory-only prompt
                prompt_template = PromptTemplate.from_template("""
                    You are a helpful assistant. Based on our conversation history, respond to the user's question.
                    
                    The available information doesn't contain details about this topic.
                    Respond with: "I don't know. I'm sorry, I don't have enough information to answer that."

                    Previous conversation:
                    {chat_history}

                    User question:
                    {question}

                    Answer:
                    """)
                
                response = (prompt_template | llm_instance).invoke({
                    "question": message,
                    "chat_history": chat_history
                })
                
                response_text = "I don't know. I'm sorry, I don't have enough information to answer that."
                self.memory_agent.save_to_memory(user_id, message, response_text)
                sources = []
                
                # Track response type
                self.last_response_type[user_id] = "no_knowledge"
                
                debug_info = {
                    "retrieval_method": "memory_only",
                    "hybrid_alpha": hybrid_alpha,
                    "reranking_used": False,
                    "context_relevant": False,
                    "reason": "No relevant context found"
                }

            return {
                "response": response_text,
                "sources": sources,
                "debug_info": debug_info
            }
    
    def _is_context_relevant(self, context: str, query: str) -> bool:
        """Enhanced relevance check between context and query."""
        if not context or len(context.strip()) < 50:
            return False
        
        # Check for obviously off-topic queries first
        general_knowledge_patterns = [
            "what is the color", "what is the colour", "what color is", "what colour is",
            "how tall is", "what year was", "who is the president", "who won",
            "what day is", "what time is", "what is the weather", "what's the weather",
            "what is the capital", "what's the capital", "when was", "where is",
            "how old is", "what is the temperature", "what's the temperature"
        ]
        
        query_lower = query.lower().strip()
        
        # If query matches general knowledge patterns, it's likely not relevant to document content
        if any(pattern in query_lower for pattern in general_knowledge_patterns):
            # Double-check by seeing if key terms from query appear in context
            # Extract meaningful words from query
            query_words = set(query_lower.split())
            stop_words = {'what', 'is', 'the', 'of', 'a', 'an', 'in', 'on', 'at', 'who', 'how', 'when', 'was', 'were', 'are', 'you', 'can', 'could', 'would', 'should', 'did', 'do', 'does', 'has', 'have', 'had', 'and', 'or', 'but', 'for', 'with', 'from', 'to', 'it', 'that', 'this'}
            meaningful_words = query_words - stop_words
            
            if meaningful_words:
                context_lower = context.lower()
                # Only consider relevant if multiple meaningful words appear in context
                matches = sum(1 for word in meaningful_words if word in context_lower)
                return matches >= min(2, len(meaningful_words))  # Need at least 2 matches or all words if <2
            else:
                return False  # No meaningful words to check
        
        # For non-general knowledge queries, use basic keyword overlap
        query_words = set(query_lower.split())
        context_words = set(context.lower().split())
        
        stop_words = {'what', 'is', 'the', 'of', 'a', 'an', 'in', 'on', 'at', 'who', 'how', 'when', 'was', 'were', 'are', 'you', 'can', 'could', 'would', 'should', 'did', 'do', 'does', 'has', 'have', 'had', 'and', 'or', 'but', 'for', 'with', 'from', 'to'}
        
        meaningful_query_words = query_words - stop_words
        if len(meaningful_query_words) == 0:
            return False  # Changed from True - if no meaningful words, assume not relevant
            
        # Check overlap - need higher ratio for relevance
        overlap = meaningful_query_words.intersection(context_words)
        overlap_ratio = len(overlap) / len(meaningful_query_words)
        
        return overlap_ratio >= 0.5  # Increased from 0.3 to 0.5 - need stronger relevance signal
    
    def process_files(self, file_paths: List[str], advanced_extraction: bool = False, perform_ocr: bool = False) -> List[str]:
        """Process multiple files (PDF, text, etc.)."""
        # Ensure uploads directory exists (though UPLOAD_DIR is created by api.py)
        # RAGService will handle creation of extracted_content_dir
        uploads_dir = Path("temp/uploads")
        uploads_dir.mkdir(exist_ok=True, parents=True)
        
        results = []
        
        for file_path_str in file_paths:
            file_name = os.path.basename(file_path_str)
            
            try:
                # Process with RAG service
                rag_results = self.rag_service.process_file(file_path_str, advanced_extraction, perform_ocr=perform_ocr)
                results.extend(rag_results)
                
            except Exception as e:
                results.append(f"Error processing {file_name}: {str(e)}")
        
        return results
        
    # Keep the original method for backward compatibility
    def process_pdfs(self, file_paths: List[str], advanced_extraction: bool = False) -> List[str]:
        """Legacy method that calls process_files."""
        return self.process_files(file_paths, advanced_extraction, perform_ocr=False)
