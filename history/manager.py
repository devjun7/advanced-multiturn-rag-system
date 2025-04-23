from typing import List, Tuple

class ChatHistoryManager:
    """
    Simple in-memory storage for chat history.
    Stores history as a list of (user_message, ai_message) tuples.
    """
    def __init__(self):
        self._history: List[Tuple[str, str]] = []
        print("ChatHistoryManager initialized (in-memory).")

    def add_turn(self, user_message: str, ai_message: str):
        """Adds a new turn to the chat history."""
        if not isinstance(user_message, str) or not isinstance(ai_message, str):
            print("Warning: Both user and AI messages must be strings.")
            return
        self._history.append((user_message, ai_message))
        print(f"Added turn {len(self._history)} to chat history.")

    def get_history(self) -> List[Tuple[str, str]]:
        """Returns the entire chat history."""
        return self._history.copy() # Return a copy to prevent external modification

    def clear_history(self):
        """Clears the chat history."""
        self._history = []
        print("Chat history cleared.")

    def get_last_n_turns(self, n: int) -> List[Tuple[str, str]]:
        """Returns the last N turns of the chat history."""
        if n <= 0:
            return []
        return self._history[-n:]