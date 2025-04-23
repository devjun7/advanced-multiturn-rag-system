import os
import pickle
from typing import List, Any, Tuple
from langchain_core.documents import Document

def ensure_dir_exists(file_path: str):
    """Ensures the directory for a given file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_object(obj: Any, file_path: str):
    """Saves a Python object to a file using pickle."""
    ensure_dir_exists(file_path)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Object successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving object to {file_path}: {e}")

def load_object(file_path: str) -> Any:
    """Loads a Python object from a pickle file."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}. Returning None.")
        return None
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Object successfully loaded from {file_path}")
        return obj
    except Exception as e:
        print(f"Error loading object from {file_path}: {e}. Returning None.")
        return None

def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single string for the prompt."""
    if not docs:
        return "No relevant documents found."
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

def format_history_for_prompt(chat_history: List[Tuple[str, str]]) -> str:
    """Formats chat history list into a string suitable for prompt inclusion."""
    if not chat_history:
        return "No conversation history yet."
    formatted = []
    for user_msg, ai_msg in chat_history:
        formatted.append(f"User: {user_msg}")
        formatted.append(f"AI: {ai_msg}")
    return "\n".join(formatted)