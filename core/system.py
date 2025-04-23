import time
import logging
from typing import List, Tuple, Optional
from langchain_core.documents import Document

from config.settings import RETRIEVER_K
from embeddings.embeddings import Embeddings
from llm.vllm_client import VLLMClient
from retrievers.faiss_retriever import FAISSVectorStoreManager
from history.manager import ChatHistoryManager
from history.adaptive_handler import AdaptiveChatHistoryHandler
from rewriting.rewriter import QueryRewriter
from core.prompt_builder import PromptBuilder

# Setup logger for this module
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Orchestrates the advanced RAG pipeline, integrating components like
    embeddings, LLM, vector store, history management, query rewriting,
    and prompt building.
    """
    def __init__(self, index_path_prefix: str):
        """Initializes all components of the RAG system."""
        logger.info("Initializing Advanced RAG System...")
        start_time = time.time()

        try:
            # Embedding Model
            self.index_path_prefix = index_path_prefix
            logger.info("Loading Embedding Model...")
            self.embedding_model = Embeddings()
            if not self.embedding_model.client:
                 # Specific exception for clearer error source
                 raise ValueError("Failed to initialize Embedding Model client. Cannot proceed.")
            logger.info("Embedding Model loaded successfully.")

            # LLM Client
            logger.info("Initializing LLM Client...")
            self.llm_client = VLLMClient() # Configuration loaded internally from settings.py
            logger.info("LLM Client initialized successfully.")

            # Vector Store and Retriever
            logger.info("Initializing Vector Store Manager...")
            self.vector_store_manager = FAISSVectorStoreManager(
                embedding_function=self.embedding_model,
                index_path_prefix=self.index_path_prefix
                # Paths and K loaded internally from settings.py
            )
            self.retriever = self.vector_store_manager.get_retriever() # Get initial retriever (might be None if no index)
            if self.retriever:
                logger.info("FAISS Vector Store Manager initialized and retriever created (existing index found).")
            else:
                logger.info("FAISS Vector Store Manager initialized (no existing index found or error loading).")

            # Chat History Management
            logger.info("Initializing Chat History Manager...")
            self.history_manager = ChatHistoryManager()
            logger.info("Chat History Manager initialized.")

            # Adaptive History Handler (uses same embedding model)
            logger.info("Initializing Adaptive History Handler...")
            self.adaptive_history_handler = AdaptiveChatHistoryHandler(
                embedding_function=self.embedding_model
                # Configuration loaded internally from settings.py
            )
            logger.info("Adaptive History Handler initialized.")

            # Query Rewriter (uses the main LLM client if configured)
            logger.info("Initializing Query Rewriter...")
            self.query_rewriter = QueryRewriter(llm=self.llm_client)
            logger.info("Query Rewriter initialized.")

            # Prompt Builder
            logger.info("Initializing Prompt Builder...")
            self.prompt_builder = PromptBuilder() # Uses default system prompt from settings
            logger.info("Prompt Builder initialized.")

        except Exception as e:
            logger.critical(f"Failed to initialize a core component of the RAG system: {e}", exc_info=True)
            # Re-raise the exception to prevent the system from being used in an inconsistent state
            raise RuntimeError(f"RAG System initialization failed: {e}") from e

        end_time = time.time()
        logger.info(f"RAG System initialized successfully in {end_time - start_time:.2f} seconds.")

    def add_documents_to_knowledge_base(self, documents: List[Document]):
        """
        Adds documents to the FAISS vector store via the manager.

        Args:
            documents: A list of Langchain Document objects to add.
        """
        if not documents:
            logger.warning("add_documents_to_knowledge_base called with no documents.")
            return

        logger.info(f"Adding {len(documents)} document chunks to the knowledge base...")
        try:
            self.vector_store_manager.add_documents(documents)
            # Refresh the retriever instance as the underlying store has changed
            self.retriever = self.vector_store_manager.get_retriever()
            if self.retriever:
                logger.info("Documents added and retriever updated successfully.")
            else:
                 # This case might occur if adding docs failed internally in the manager
                 logger.error("Documents potentially added, but failed to get updated retriever.")
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}", exc_info=True)
            # Optionally re-raise or handle more gracefully depending on desired behavior
            raise

    def get_response(self, user_query: str) -> str:
        """
        Processes a user query through the full RAG pipeline: history retrieval,
        query rewriting, document retrieval, adaptive history processing,
        prompt building, LLM call, and history update.

        Args:
            user_query: The raw query string from the user.

        Returns:
            The AI's response string.

        Raises:
            Exception: Propagates exceptions from underlying components (e.g., LLM client).
        """
        logger.info(f"--- Processing User Query: '{user_query}...' ---") # Log truncated query
        start_time = time.time()

        try:
            # 1. Get current chat history
            current_history = self.history_manager.get_history()
            logger.debug(f"Retrieved {len(current_history)} turns from history.")

            # 2. Rewrite the query based on history (if applicable)
            # Use original user_query for logging, rewritten_query for retrieval
            rewritten_query = self.query_rewriter.rewrite_query(current_history, user_query)
            if rewritten_query != user_query:
                logger.info(f"Original query rewritten for retrieval: '{rewritten_query}...'")
            else:
                logger.info("Query rewriting deemed unnecessary.")

            # 3. Retrieve relevant documents using the rewritten query
            retrieved_docs: List[Document] = []
            if self.retriever:
                logger.info("Retrieving relevant documents from vector store...")
                try:
                    retrieved_docs = self.retriever.invoke(rewritten_query)
                    logger.info(f"Retrieved {len(retrieved_docs)} documents.")
                    if logger.isEnabledFor(logging.DEBUG):
                         for i, doc in enumerate(retrieved_docs):
                             logger.debug(f"  Doc {i+1}: {doc.page_content}... (Source: {doc.metadata.get('source', 'N/A')})")
                except Exception as e:
                    logger.error(f"Error during document retrieval: {e}", exc_info=True)
                    retrieved_docs = [] # Ensure empty list on error, proceed without docs
            else:
                logger.warning("Retriever not available (no index loaded?). Skipping document retrieval.")

            # 4. Process chat history adaptively (full or RAG-based)
            # Pass the original user_query, as RAG-on-history relates to the user's current intent
            logger.info("Processing chat history adaptively...")
            processed_history_str = self.adaptive_history_handler.process_history(
                current_history,
                user_query
            )
            logger.debug(f"Processed History for Prompt (length: {len(processed_history_str)}): {processed_history_str[:500]}...")

            # 5. Build the final prompt for the LLM
            logger.info("Building final prompt for LLM...")
            final_prompt = self.prompt_builder.build_prompt(
                processed_history=processed_history_str,
                retrieved_docs=retrieved_docs,
                rewritten_query=rewritten_query # Use the potentially rewritten query
            )
            logger.debug(f"Final Prompt for LLM (length: {len(final_prompt)}): {final_prompt[:500]}...")

            # 6. Get response from the LLM
            logger.info("Sending request to LLM...")
            llm_response = self.llm_client.invoke(final_prompt)
            logger.info("Received response from LLM.")
            logger.debug(f"LLM Raw Response (length: {len(llm_response)}): {llm_response[:500]}...")

            # 7. Update chat history (only after successful LLM call)
            self.history_manager.add_turn(user_query, llm_response)
            logger.debug("Chat history updated with the new turn.")

            end_time = time.time()
            logger.info(f"--- Query processed successfully in {end_time - start_time:.2f} seconds. ---")

            return llm_response

        except Exception as e:
            # Catch any unexpected errors during the pipeline execution
            logger.error(f"Error during RAG pipeline execution for query '{user_query[:100]}...': {e}", exc_info=True)
            # Re-raise the exception to be handled by the calling UI layer
            raise

    def clear_chat_history(self):
        """Clears the current conversation history managed by the ChatHistoryManager."""
        logger.info("Clearing chat history.")
        self.history_manager.clear_history()
        logger.info("Chat history cleared successfully.")