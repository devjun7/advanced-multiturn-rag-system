import tiktoken
from typing import List, Dict, Any
from config.settings import HISTORY_TOKENIZER_MODEL

# --- Token Counting ---
# Initialize tokenizer globally to avoid reloading
try:
    tokenizer = tiktoken.encoding_for_model(HISTORY_TOKENIZER_MODEL)
except KeyError:
    print(f"Warning: Model {HISTORY_TOKENIZER_MODEL} not found for tiktoken. Using 'cl100k_base'.")
    tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Counts the number of tokens in a given text string."""
    if not text:
        return 0
    return len(tokenizer.encode(text))

def count_message_tokens(messages: List[Dict[str, str]]) -> int:
    """Counts tokens for a list of messages in OpenAI format."""
    num_tokens = 0
    for message in messages:
        # Rough approximation based on OpenAI cookbook:
        num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += count_tokens(value)
            if key == "name":  # If there's a name, the role is omitted
                num_tokens -= 1  # Role is always required and always 1 token
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens

def count_history_tokens(chat_history: List[tuple[str, str]]) -> int:
    """Counts tokens in the custom chat history format (list of tuples)."""
    num_tokens = 0
    for user_msg, ai_msg in chat_history:
        num_tokens += count_tokens(user_msg)
        num_tokens += count_tokens(ai_msg)
        num_tokens += 8 # Approximate overhead per turn (role tags, newlines etc.)
    return num_tokens