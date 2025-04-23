import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file if it exists

# --- Model Configuration ---
# Embedding Model
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct" # 임베딩 모델
MAX_SEQ_LENGTH = 512

# LLM Configuration (Replace with your actual vLLM endpoint)
VLLM_ENDPOINT_URL = os.getenv("VLLM_ENDPOINT_URL", "http://localhost:8087/v1/completions") # Or your actual vLLM API endpoint
VLLM_MODEL_IDENTIFIER = os.getenv("VLLM_MODEL_IDENTIFIER", "qwq-32b") # Name used internally or potentially in API requests if needed
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY") # Default for local vLLM, change if auth is needed
VLLM_REQUEST_TIMEOUT = 1000 # Seconds
VLLM_DEFAULT_TEMPERATURE = 0.01
VLLM_DEFAULT_MAX_TOKENS = 8192
VLLM_DEFAULT_TOP_P = 0.01
VLLM_DEFAULT_TOP_K = -1

# --- Retriever Configuration ---
# 벡터 스토어 경로 (Streamlit 앱에서 컨텍스트 로드 시 재생성될 수 있음)
FAISS_INDEX_PATH = "./vector_store/faiss_index"
FAISS_METADATA_PATH = "./vector_store/faiss_metadata.pkl"
RETRIEVER_K = 3 # Number of documents to search
CHUNK_SIZE = 200  # Target size of each chunk (in characters)
CHUNK_OVERLAP = 20 # Number of characters to overlap between chunks

# --- Chat History Configuration ---
HISTORY_MAX_TOKENS_THRESHOLD = 4096 # Token limit before RAG-on-history is triggered
HISTORY_RAG_K = 3 # Number of relevant history turns to retrieve when over threshold
HISTORY_TOKENIZER_MODEL = "gpt-4o" # Model used for token counting (tiktoken)

# --- Query Rewriting Configuration ---
REWRITER_USE_LLM = True # Set to False to use a simpler template-based rewrite
REWRITER_LLM_TEMPERATURE = 0.1
REWRITER_LLM_MAX_TOKENS = 2048
REWRITER_PROMPT = """Rewrite the user query into a clear, standalone question or instruction that incorporates relevant context from the history.
This rewritten query will be used to search for relevant documents. Focus on keywords and the core intent.
Do NOT answer the query, just rewrite it. Output ONLY the rewritten query.
You must rewrite the query using the same language the user used for their query."""

# --- Prompt for LLM ---
DEFAULT_SYSTEM_PROMPT = """You are a capable AI assistant designed to answer user questions.
Your main goal is to provide the most accurate and relevant answer to the user's question based on the provided Context Documents and Conversation History.
If the context documents are relevant, prioritize information from them.
If the context is not relevant or missing, rely on the conversation history and your general knowledge.
Be precise and informative. If you don't know the answer or cannot find it in the context or history.
Respond using the same language the user used for their question.
"""

# --- Device Settings ---
DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_NUMBER = None # "0" if torch.cuda.is_available() else None