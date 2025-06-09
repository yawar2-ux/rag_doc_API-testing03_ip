"""
Memory Agent Module: Manages conversation history for users.
"""

from typing import Dict, Optional
from langchain.memory import ConversationSummaryBufferMemory
from ..Model.LLM import Groq_Model # Corrected relative import

class MemoryAgent:
    """
    Manages conversation memory for users with hybrid approach:
    - Keeps recent messages verbatim
    - Summarizes older messages for context
    """
    
    def __init__(self):
        """Initialize user session storage."""
        # Store user sessions with their memory
        self.user_sessions = {}
    
    def get_user_memory(self, user_id: str) -> ConversationSummaryBufferMemory:
        """
        Get or create memory for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            ConversationSummaryBufferMemory instance for the user
        """
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = ConversationSummaryBufferMemory(
                llm=Groq_Model,
                max_token_limit=2000,
                memory_key="chat_history",
                return_messages=True
            )
        return self.user_sessions[user_id]
    
    def save_to_memory(self, user_id: str, question: str, answer: str) -> None:
        """
        Save a question-answer pair to the user's memory.
        
        Args:
            user_id: Unique identifier for the user
            question: The user's question
            answer: The system's response
        """
        memory = self.get_user_memory(user_id)
        memory.save_context({"question": question}, {"output": answer})
    
    def get_chat_history(self, user_id: str) -> str:
        """
        Get the chat history for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Formatted string of chat history
        """
        memory = self.get_user_memory(user_id)
        return memory.buffer

    def clear_memory(self, user_id: Optional[str] = None) -> None:
        """
        Clear memory for a specific user or all users.
        
        Args:
            user_id: Optional unique identifier for the user. 
                     If None, clears memory for all users.
        """
        if user_id:
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
                print(f"Memory cleared for user: {user_id}")
            else:
                print(f"No memory session found for user: {user_id}")
        else:
            self.user_sessions.clear()
            print("All user memory sessions cleared.")
