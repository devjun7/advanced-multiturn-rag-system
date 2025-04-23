from typing import List, Tuple, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from config.settings import HISTORY_MAX_TOKENS_THRESHOLD, HISTORY_RAG_K
from utils.token_counters import count_history_tokens
from utils.helpers import format_history_for_prompt

class AdaptiveChatHistoryHandler:
    """
    Handles chat history, applying RAG-based summarization/retrieval
    if the history exceeds a token threshold.
    """
    def __init__(
        self,
        embedding_function: Embeddings,
        max_tokens: int = HISTORY_MAX_TOKENS_THRESHOLD,
        rag_k: int = HISTORY_RAG_K,
    ):
        """
        Initializes the adaptive history handler.

        Args:
            embedding_function: Embeddings model for RAG-on-history.
            max_tokens: Token threshold to trigger RAG-on-history.
            rag_k: Number of history turns to retrieve when RAG is applied.
        """
        self.embedding_function = embedding_function
        self.max_tokens = max_tokens
        self.rag_k = rag_k
        print(f"AdaptiveHistory initialized. Threshold: {max_tokens} tokens, RAG K: {rag_k}")

    def _create_history_documents(self, chat_history: List[Tuple[str, str]]) -> List[Document]:
        """Converts chat history turns into Langchain Documents."""
        docs = []
        for i, (user_msg, ai_msg) in enumerate(chat_history):
            content = f"Turn {i+1}:\nUser: {user_msg}\nAI: {ai_msg}"
            # Add metadata
            metadata = {"turn_number": i + 1, "source": "chat_history"}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _retrieve_relevant_history(
        self,
        current_query: str,
        history_documents: List[Document]
    ) -> List[Document]:
        """
        Performs similarity search on history documents using an in-memory FAISS index.

        Args:
            current_query: The latest user query to find relevant history for.
            history_documents: List of history turns formatted as Documents.

        Returns:
            A list of the most relevant history documents (turns).
        """
        if not history_documents:
            return []

        print(f"Performing RAG on {len(history_documents)} history turns for query: '{current_query}'")
        try:
            # Create a temporary, in-memory FAISS index for the history
            # This is potentially slow for very long histories but avoids persistent state
            print("Creating temporary FAISS index for history...")
            history_vector_store = FAISS.from_documents(
                history_documents,
                self.embedding_function
            )
            print("Temporary index created.")

            # Determine the number of turns to retrieve (cannot exceed available turns)
            k_retrieve = min(self.rag_k, len(history_documents))
            if k_retrieve <= 0: return [] # Handle edge case

            print(f"Retrieving top {k_retrieve} relevant history turns.")
            relevant_docs = history_vector_store.similarity_search(
                current_query,
                k=k_retrieve
            )
            print(f"Retrieved {len(relevant_docs)} relevant history turns.")
            # Sort by turn number to maintain some chronological order in the retrieved subset
            relevant_docs.sort(key=lambda doc: doc.metadata.get("turn_number", float('inf')))
            return relevant_docs

        except Exception as e:
            print(f"Error during RAG-on-history retrieval: {e}")
            # Fallback: Returning empty list for now to signal failure.
            return []


    def process_history(
        self,
        chat_history: List[Tuple[str, str]],
        current_query: str
    ) -> str:
        """
        Processes the chat history. Returns either the full formatted history
        or a RAG-based summary/selection based on token count.

        Args:
            chat_history: The list of (user_message, ai_message) tuples.
            current_query: The current user query (needed for RAG-on-history).

        Returns:
            A formatted string representing the relevant history for the prompt.
        """
        if not chat_history:
            return "No conversation history yet."

        history_token_count = count_history_tokens(chat_history)
        print(f"Current history token count: {history_token_count} (Threshold: {self.max_tokens})")

        if history_token_count <= self.max_tokens:
            print("History is below threshold. Using full history.")
            return format_history_for_prompt(chat_history)
        else:
            print("History exceeds threshold. Applying RAG to history.")
            history_docs = self._create_history_documents(chat_history)
            relevant_history_docs = self._retrieve_relevant_history(current_query, history_docs)

            if not relevant_history_docs:
                print("Warning: RAG-on-history failed to retrieve relevant turns. Returning indication.")
                # Fallback: could return last K turns, or just a message.
                # Using a message to indicate RAG was attempted but yielded nothing specific.
                last_turn_formatted = format_history_for_prompt(chat_history[-1:])
                return f"History is long. Could not retrieve specific relevant turns.\nMost Recent Turn:\n{last_turn_formatted}"

            # Format the retrieved documents
            formatted_relevant_history = "\n\n".join([doc.page_content for doc in relevant_history_docs])
            return f"History is long. Relevant turns based on the current query:\n{formatted_relevant_history}"