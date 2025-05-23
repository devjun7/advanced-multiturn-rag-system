# 📚 Advanced-Multiturn-RAG-System

**An advanced Retrieval-Augmented Generation (RAG) system designed for robust multi-turn conversations, efficient inference, and flexible knowledge integration.**

This project implements a sophisticated RAG system leveraging a modern technology stack to handle complex question-answering and information retrieval tasks. Key features include multi-turn conversation management, dynamic query rewriting, adaptive conversation history handling, and support for various document formats. It utilizes a Streamlit-based web UI for user interaction and performs efficient inference via a Large Language Model (LLM) hosted on a vLLM server.

## 1\. Overview

The Advanced-Multiturn-RAG-System goes beyond simple retrieval by understanding context within ongoing conversations to generate accurate and relevant responses.

* **Handles Complex Queries:** Effectively answers questions that require understanding previous turns in the conversation.
* **Versatile Document Support:** Builds knowledge bases from text input as well as `.txt`, `.pdf`, and `.docx` file uploads.
* **High-Performance Inference:** Integrates with vLLM for fast and efficient LLM serving, ensuring low latency responses.
* **Developer-Friendly Design:** Features a modular architecture for ease of maintenance and extension.

## 2\. Features

* **🚀 High-Performance Embeddings:** Utilizes state-of-the-art embedding models (e.g., `intfloat/multilingual-e5-large-instruct`) to generate high-quality vector representations for documents and queries.
* **⚡️ vLLM-Powered LLM Inference:** Interfaces with an external LLM (e.g., `Qwen/QWQ-32B` or any huggingface/local models supporting vLLM) served via vLLM for rapid and scalable text generation. (`llm/vllm_client.py`)
* **💾 FAISS Vector Store:** Employs FAISS for efficient storage and fast similarity searches over large-scale document embeddings. (`retrievers/faiss_retriever.py`)
* **🧠 Multi-turn Conversation Management:** Maintains conversation history and leverages it for context-aware query processing and response generation. (`history/manager.py`)
* **💡 Adaptive History Handling:** Dynamically adjusts history processing based on length. Short histories are used entirely; for longer histories, a RAG approach is applied to the history itself, retrieving relevant turns based on the current query to manage context limits effectively. (`history/adaptive_handler.py`)
* **✍️ LLM-Based Query Rewriting:** Rewrites the user's current query into a standalone, search-optimized format based on the conversation history, enhancing retrieval accuracy. (`rewriting/rewriter.py`)
* **📄 Diverse Document Format Support:** Accepts context via direct text input or file uploads (`.txt`, `.pdf`, `.docx`). (`main.py` via `utils/file_processor.py`)
* **💻 Intuitive Web UI (Streamlit):** Provides a user-friendly web interface for loading context and interacting with the system. (`main.py`)
* **🧩 Modular Design:** Core functionalities (embedding, LLM client, retrieval, history management) are encapsulated in distinct modules, promoting maintainability and extensibility.

## 3\. System Architecture

This system features the following modular structure and data flow:

1.  **UI (Streamlit)**: Receives context (text/file) and query input from the user. Calls `utils/file_processor.py` to process documents when context is loaded via file upload. Calls the `get_response` method of `RAGSystem` upon query request and displays the result. Manages and displays the chat history.

2.  **RAGSystem (Core Orchestrator)**: Coordinates the entire RAG pipeline.

    - Retrieves the current conversation history from `ChatHistoryManager`.
    - Calls `QueryRewriter` to rewrite the user query.
    - Uses `FAISSVectorStoreManager` (and its internal Retriever) to search for relevant documents using the rewritten query.
    - Calls `AdaptiveChatHistoryHandler` to process the conversation history (Full or RAG-based).
    - Uses `PromptBuilder` to generate the final LLM prompt by combining the system prompt, processed conversation history, retrieved documents, and the rewritten query.
    - Sends an inference request to the vLLM server via `vLLMClient` and receives the response.
    - Adds the new user query and LLM response to `ChatHistoryManager`.
    - Returns the final response to the UI.

3.  **Components**:

    - `Embeddings`: Converts text into vectors.
    - `VectorStore`: Stores and retrieves vectors.
    - `LLM Client`: Communicates with the external LLM server.
    - `History`: Stores and adaptively processes conversation history.
    - `Rewriter`: Rewrites queries.
    - `PromptBuilder`: Constructs LLM inputs.
    - `FileProcessor`: Processes files inputted as context.
    - `Config`: Manages system settings.

4.  **Core Components**

    - **llm/vllm_client.py (LLM Client)**

      Implements Langchain's LLM interface.

      Communicates with the vLLM server's OpenAI-compatible API endpoint via `VLLM_ENDPOINT_URL` defined in `config/settings.py`.

      Uses the `httpx` library to handle synchronous/asynchronous HTTP requests.

      Supports API key authentication and timeout settings.

    - **embeddings/embeddings.py (Embedding Model)**

      Implements Langchain's Embeddings interface.

      Uses the `sentence-transformers` library to load and manage Hugging Face models (e.g., `intfloat/multilingual-e5-large-instruct`).

      Automatically detects and uses CPU or CUDA devices based on the `DEVICE` setting in `config/settings.py`.

      Provides an option for embedding normalization.

    - **retrievers/faiss_retriever.py (Vector Store)**

      Manages FAISS indexes through the `FAISSVectorStoreManager` class.

      Handles loading and saving of index files (`.faiss`, `.pkl`) to support persistence.

      Provides functionality for adding documents (`add_documents`) and similarity search (`similarity_search`).

      Creates a Langchain `VectorStoreRetriever` object for integration with the RAG pipeline.

    - **history/manager.py (Conversation History Manager)**

      Stores conversation history as a list of (user message, AI message) tuples using a simple in-memory approach.

      Provides functions for adding history, retrieving the full history, retrieving the last N turns, and clearing the history.

    - **history/adaptive_handler.py (Adaptive History Handler)**

      Calculates the token count of the conversation history and compares it with `HISTORY_MAX_TOKENS_THRESHOLD` from `config/settings.py`.

      If below the threshold, formats and returns the entire history.

      If exceeding the threshold, converts conversation turns into `Document` objects and creates a temporary in-memory FAISS index. Uses the current user query to retrieve the `HISTORY_RAG_K` most relevant turns from this temporary index, formats them, and returns them.

    - **rewriting/rewriter.py (Query Rewriter)**

      Performs LLM-based rewriting or simple rewriting (currently returns the original query) based on the `REWRITER_USE_LLM` setting.

      When using the LLM, generates a specific prompt including the conversation history and current query, then calls `VLLMClient`.

    - **core/system.py (Core System)**

      Acts as the orchestrator, initializing all components and controlling the overall flow of the RAG pipeline.

      Executes each step of query processing sequentially within the `get_response` method.

      Provides functions for adding documents to the knowledge base (`add_documents_to_knowledge_base`) and clearing chat history (`clear_chat_history`).

    - **core/prompt_builder.py (Prompt Builder)**

      Combines the system instructions, processed conversation history, retrieved context documents, and the (rewritten) user query to generate the final LLM input prompt.

    - **config/settings.py (Configuration Management)**

      Sets major system configurations and hyperparameters such as model names, API endpoints, file paths, and thresholds.
      Sensitive information (e.g., API keys) can also be managed via an `.env` file.

    - **main.py (User Interface)**

      Implements the web-based UI using Streamlit.

      Provides context loading functionality via text input and file upload (TXT, PDF, DOCX).

    - **utils/file_processor.py (File Processor)**

      Contains logic for extracting text from different file types (requires `pypdf`, `python-docx`).

      Determines how to split the extracted text into chunks and the chunk processing method using a splitter.

## 5\. Setup & Installation

- **Prerequisites**

  - Python 3.10+
  - Linux (For vLLM and FAISS-GPU)
  - Git
  - NVIDIA driver, CUDA Toolkit (PyTorch compatible version), NVIDIA GPU (Supporting vLLM)
  - Running vLLM server: A vLLM instance serving the model corresponding to `VLLM_MODEL_IDENTIFIER` in `config/settings.py` is required.

- **Clone Repository**

  ```bash
  git clone https://github.com/devjun7/advanced-multiturn-rag-system.git
  cd advanced-multiturn-rag-system
  ```

- **Install Dependencies**

  `pip install -r requirements.txt`

  **requirements.txt**

  ```
  langchain==0.3.23
  langchain-core==0.3.51
  langchain-community==0.3.21
  faiss-gpu==1.7.2
  torch==2.6.0
  sentence-transformers==4.0.2
  transformers==4.51.2
  httpx==0.28.1
  tiktoken==0.9.0
  python-dotenv==1.0.1
  pypdf==5.4.0
  python-docx==1.1.2
  numpy==1.26.4
  vllm==0.8.3
  streamlit==1.44.1
  ```

- **Configuration**

  - Open the `config/settings.py` file, review the settings, and modify if necessary.
  - **Most important setting:** `VLLM_ENDPOINT_URL` must be changed to the OpenAI-compatible API endpoint address of your running vLLM server (e.g., `"http://localhost:9090/v1/completions"`).
  - `VLLM_MODEL_IDENTIFIER` should match the model identifier used by the vLLM server.
  - If needed, create a `.env` file to manage sensitive information like `VLLM_ENDPOINT_URL` or `VLLM_API_KEY` (requires the `python-dotenv` library).
  - Verify that `EMBEDDING_MODEL_NAME` is correct.
  - Ensure the `DEVICE` setting ("cuda" or "cpu") matches your environment. (This determines the device for retriever model)

- **Run vLLM Server**

  - This is an external dependency of the project. You need to run the vLLM server in a separate environment. Refer to the vLLM documentation for setup and execution.
  - Below is an example command:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --port 9090 \
    --model Qwen/QwQ-32B \
    --dtype auto \
    --gpu-memory-utilization 0.95 \
    --max-model-len 20000 \
    --served-model-name qwq-32b \
    --enforce-eager \
    --max-num-seqs 16 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-reasoning \
    --reasoning-parser deepseek_r1
    ```

## 6\. Running the Application

Execute the following command from the project root directory:

`streamlit run ./main.py --server.port [PORT_NUMBER]`

The Streamlit application will automatically open in your web browser.

## 7\. Usage

_Once the application loads, the RAG system initializes. (Model loading might take some time.)_

- **Load Context**

  - Under `Select Context Input Method`, choose either `Direct Text Input` or `File Upload`.
  - **Text Input**: Paste the context content into the text area.
  - **File Upload**: Upload files in supported formats (PDF, DOCX, TXT).
  - **Add Input Context to Knowledge Base**: The system processes the input context and creates/updates the FAISS vector database with your context.

- **Ask a Question**

  - Once context loading is complete, enter your question in the chat input box at the bottom and press Enter.
  - The system executes the question processing pipeline (query rewriting, document retrieval, LLM call, etc.) and displays the generated response in the chat window.

- **Conversation**

  - You can continue the conversation with follow-up questions based on the previous interactions.

- **Clear Chat History**

  - Click the `Clear Chat History` button to erase the current conversation.

## 8\. Project Structure

```
    rag_system/
    ├── config/
    │   └── settings.py         # System configuration file
    ├── core/
    │   ├── system.py           # RAG system orchestrator
    │   └── prompt_builder.py   # LLM prompt generation logic
    ├── embeddings/
    │   └── embeddings.py       # Embedding model wrapper class
    ├── history/
    │   ├── manager.py          # Basic conversation history storage/management
    │   └── adaptive_handler.py # Adaptive conversation history processing logic
    ├── llm/
    │   └── vllm_client.py      # vLLM server communication client
    ├── rewriting/
    │   └── rewriter.py         # Query rewriting logic
    ├── retrievers/
    │   └── faiss_retriever.py  # FAISS vector store management and retrieval
    ├── utils/
    │   ├── file_processor.py   # File processor
    │   ├── helpers.py          # Various utilities
    │   └── token_counters.py   # Tiktoken-based token counter
    ├── vector_store/           # Storage location for FAISS index files (auto-generated)
    │   └── ...
    ├── main.py                 # Streamlit UI and application entry point
    ├── requirements.txt        # Python dependency list
    └── README.md               # This document
```

## 9\. Design Philosophy and Key Decisions
* **vLLM for Scalability**: Chosen for serving LLMs due to its high throughput and low latency capabilities, offering advantages in scalability and resource management over local model loading.
* **FAISS for Efficiency**: Selected for its fast and memory-efficient similarity search performance on large datasets, suitable for both CPU and GPU environments, with support for persistence.
* **Adaptive History for Long Contexts**: Implemented an adaptive RAG approach for conversation history to mitigate LLM context length limitations and maintain relevance in extended dialogues.
* **Modular Architecture**: Designed with clear separation of concerns to enhance code readability, testability, maintainability, and ease of future feature integration. Leverages Langchain interfaces for improved component compatibility.
* **Streamlit for Rapid UI Development**: Chosen for its ability to quickly build interactive data applications, suitable for user interaction and result visualization.
* **Flexible File Input**: Integrated support for common document formats using libraries like pypdf and python-docx.

---
