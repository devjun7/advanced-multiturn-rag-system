import streamlit as st
import os
import sys
import time
import logging
from typing import List, Optional
import io
import uuid

# try to import libraries for PDF and DOCX processing
try:
    import pypdf
except ImportError:
    st.error("The 'pypdf' library is required for PDF processing. Please install it using 'pip install pypdf'.")
    pypdf = None

try:
    import docx
except ImportError:
    st.error("The 'python-docx' library is required for Word document processing. Please install it using 'pip install python-docx'.")
    docx = None

# root path of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from langchain_core.documents import Document
    from core.system import RAGSystem
    from embeddings.embeddings import Embeddings
    from config.settings import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, DEVICE, DEVICE_NUMBER
    from utils.file_processor import process_text_to_docs, process_file_to_docs
except ImportError as e:
    st.error(f"Module import error: {e}. Please check your path settings.")
    st.stop() # app stop when error occurs

if DEVICE_NUMBER:
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NUMBER

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
    ]
)
logger = logging.getLogger(__name__)

# --- Streamlit page settings ---
st.set_page_config(page_title="Advanced Multi-Turn RAG System", layout="wide")
st.title("📄 Advanced Multi-Turn RAG System")
st.caption(f"Embedding: {EMBEDDING_MODEL_NAME} | Vector DB: FAISS | Device: {DEVICE} {f'(GPU: {DEVICE_NUMBER})' if DEVICE_NUMBER else ''}")

# RAG system instance initialization (once per session)
# Initialize session state variables if they don't exist
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"New session started. Session ID: {st.session_state.session_id}")
if 'rag_system' not in st.session_state:
    st.session_state.rag_system: Optional[RAGSystem] = None
if 'rag_system_initialized' not in st.session_state:
    st.session_state.rag_system_initialized: bool = False
if 'context_loaded' not in st.session_state:
    st.session_state.context_loaded: bool = False
if 'messages' not in st.session_state:
    st.session_state.messages: List[dict] = []
if 'text_input_area' not in st.session_state:
    st.session_state.text_input_area = ""


base_vector_dir = os.path.dirname(FAISS_INDEX_PATH)
if not os.path.exists(base_vector_dir):
    try:
        os.makedirs(base_vector_dir)
        logger.info(f"Created base vector store directory: {base_vector_dir}")
    except OSError as e:
        st.error(f"Failed to create base vector store directory: {base_vector_dir}. Error: {e}")
        st.stop()

SESSION_FAISS_INDEX_PATH = os.path.join(
    base_vector_dir,
    f"session_{st.session_state.session_id}", # Session-specific sub-directory name
    os.path.basename(FAISS_INDEX_PATH) # Index file prefix (e.g., "faiss_index")
)
logger.info(f"Using session-specific index path prefix: {SESSION_FAISS_INDEX_PATH}")

# --- RAG System loading ---
def load_rag_system(index_path_prefix: str) -> Optional[RAGSystem]:
    """
    Loads the RAGSystem instance, caching it in Streamlit's resource cache.
    Returns the RAGSystem instance or None if initialization fails.
    """
    if not st.session_state.rag_system_initialized:
        try:
            # Keep spinner text in English
            with st.spinner("Initializing RAG System... (Models loading, may take time)"):
                logger.info("Attempting to initialize RAGSystem...")
                st.session_state.rag_system = RAGSystem(index_path_prefix=index_path_prefix) # Initialization happens here
                st.session_state.rag_system_initialized = True
                # Check if an existing index was loaded
                if st.session_state.rag_system.vector_store_manager.vector_store:
                    logger.info("Existing FAISS index loaded successfully.")
                    st.session_state.context_loaded = True # Mark context as loaded if index exists
                    st.success("RAG system initialization complete. Existing knowledge base loaded.")
                else:
                    logger.info("RAGSystem initialized, no existing knowledge base found.")
                    st.success("RAG system initialization complete. Ready to load a new knowledge base.")
                    st.session_state.context_loaded = False
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
            st.error(f"Error occurred during RAG system initialization: {e}")
            st.session_state.rag_system = None
            st.session_state.rag_system_initialized = False
            st.stop() # Stop the app if core system fails

    return st.session_state.rag_system

# --- Helper functions ---
def clear_existing_index(index_path_prefix: str):
    """Delete existing FAISS index files for the given path prefix."""
    index_file = index_path_prefix + ".faiss"
    pkl_file = index_path_prefix + ".pkl"
    deleted = False
    logger.info(f"Attempting to clear index files for path: {index_path_prefix}")
    try:
        if os.path.exists(index_file):
            os.remove(index_file)
            logger.info(f"Deleted existing index file: {index_file}")
            deleted = True
        if os.path.exists(pkl_file):
            os.remove(pkl_file)
            logger.info(f"Deleted existing pkl file: {pkl_file}")
            deleted = True
    except Exception as e:
        logger.error(f"Error occurred while deleting index files ({index_path_prefix}): {e}")
        st.warning(f"Error occurred while deleting index files ({index_path_prefix}): {e}")
    return deleted # Return status

# --- RAG System loading ---
if st.session_state.rag_system is None and not st.session_state.rag_system_initialized:
    st.session_state.rag_system = load_rag_system(index_path_prefix=SESSION_FAISS_INDEX_PATH)

rag_system = st.session_state.rag_system

# --- UI configuration ---
if rag_system: # show UI only if system is loaded successfully
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 Context Input")

        context_option = st.radio(
            "Select Context Input Method:",
            ("File Upload", "Direct Text Input"),
            key="context_input_method"
        )

        docs_to_load: List[Document] = []
        uploaded_file_obj = None # Keep track of the uploaded file object

        if context_option == "Direct Text Input": # Changed Korean option to English
            context_text = st.text_area("Enter context here:", height=300, key="text_input_area")
            if context_text:
                # Process pasted text into documents using the utility function
                docs_to_load = process_text_to_docs(context_text, source="pasted_text")
        else: # Changed Korean option to English: "File Upload"
            uploaded_file_obj = st.file_uploader(
                # Keep English label
                "Upload a file (PDF, DOCX, TXT supported):",
                type=["pdf", "docx", "txt"],
                 # accept_multiple_files=False # Current setup processes one file at a time
                 key="file_uploader"
            )
            if uploaded_file_obj:
                # Process uploaded file into documents using the utility function
                with st.spinner(f"Processing file: {uploaded_file_obj.name}..."):
                    docs_to_load = process_file_to_docs(uploaded_file_obj)
                    if not docs_to_load:
                         # Error/warning is logged by process_file_to_docs
                         st.warning(f"Text extraction or file processing failed: '{uploaded_file_obj.name}'. Check logs for details.")

        # Option to clear the existing index before loading
        clear_index_before_load = st.checkbox("Clear current session's knowledge base before adding new data", value=False, key="clear_index_cb")

        if st.button("➕ Add Input Context to Knowledge Base", key="add_context_button"):
            if docs_to_load:
                with st.spinner("Processing context and updating vector DB..."):
                    try:
                        # 1. Optionally clear the existing index
                        if clear_index_before_load:
                            if clear_existing_index(SESSION_FAISS_INDEX_PATH):
                                # Reset the internal state of the manager for this session
                                rag_system.vector_store_manager.vector_store = None
                                rag_system.retriever = None
                                st.session_state.context_loaded = False
                                logger.info(f"FAISSVectorStoreManager internal store reset for session {st.session_state.session_id}.")
                                st.info("Deleted the vector index files for the current session.")
                            else:
                                logger.warning("Clear index requested, but no index files found or deletion failed.")
                                st.info("No existing index files found to delete.")
                        # 2. Add the new documents (RAGSystem already knows its session path)
                        logger.info(f"Adding {len(docs_to_load)} document chunks to the knowledge base for session {st.session_state.session_id}...")
                        rag_system.add_documents_to_knowledge_base(docs_to_load)

                        st.session_state.context_loaded = True
                        st.success(f"{len(docs_to_load)} document chunks added/updated in the current session's knowledge base!")
                    except Exception as e:
                        logger.error(f"Error occurs while adding documents to the knowledge base {e}", exc_info=True)
                        st.error(f"Failed to add context: {e}")
                        st.session_state.context_loaded = rag_system.vector_store_manager.vector_store is not None
            else:
                # Only show warning if no file was processed OR text was empty
                if not (context_option == "File Upload" and uploaded_file_obj): # Use English option name
                    st.warning("No context to add. Please paste text or upload a file.")

        if st.button("➖ Delete Knowledge Base Created in Current Session", key="delete_context_button"):
            with st.spinner("Deleting knowledge base..."):
                try:
                    # clear the existing index
                    if clear_existing_index(SESSION_FAISS_INDEX_PATH):
                        # Reset the internal state of the manager for this session
                        rag_system.vector_store_manager.vector_store = None
                        rag_system.retriever = None
                        st.session_state.context_loaded = False
                        logger.info(f"FAISSVectorStoreManager internal store reset for session {st.session_state.session_id}.")
                        st.info("Deleted the vector index files for the current session.")
                    else:
                        logger.warning("Clear index requested, but no index files found or deletion failed.")
                        st.info("No existing index files found to delete.")
                except Exception as e:
                    logger.error(f"Error occurs while deleting documents to the knowledge base {e}", exc_info=True)
                    st.error(f"Failed to delete context: {e}")

    with col2:
        st.subheader("💬 Chatbot")

        # Button to clear chat history
        if st.button("🧹 Reset Chat Session"):
            st.session_state.messages = []
            if rag_system and hasattr(rag_system, 'clear_chat_history'):
                rag_system.clear_chat_history() # Clear RAG system's internal history too
            logger.info("Chat session cleared by user.")
            st.rerun()

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Ensure retriever is fetched if vector store exists
        if rag_system.vector_store_manager.vector_store and not rag_system.retriever:
            rag_system.retriever = rag_system.vector_store_manager.get_retriever()

        # Chat input field
        if prompt := st.chat_input("Ask a question based on the knowledge base..."):
            # Add user message to chat history and display it
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Check if context has been loaded before allowing query
            if not st.session_state.context_loaded:
                st.warning("Please load context (text or file) into the knowledge base first.")
                # Remove the user message we just added, as we can't process it
                st.session_state.messages.pop()
            else:
                # Get AI response from RAG system
                with st.spinner("Thinking..."):
                    try:
                        logger.info(f"Processing user query: '{prompt}'")
                        response = rag_system.get_response(prompt)
                        logger.info(f"Generated response for query '{prompt}'")

                        # Display AI response and add to history
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        logger.error(f"Error getting response for query '{prompt}': {e}", exc_info=True)
                        st.error(f"Error occurred while generating response: {e}")

else:
    st.warning("Failed to load the RAG system. Please check settings or contact an administrator.")

# app execution information
st.sidebar.header("System Information")
st.sidebar.info(f"Session ID: {st.session_state.session_id}")
st.sidebar.info(f"Device Type: {DEVICE}")
if DEVICE_NUMBER:
    st.sidebar.info(f"Device Number: {DEVICE_NUMBER}")
st.sidebar.info(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
