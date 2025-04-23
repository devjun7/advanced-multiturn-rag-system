from typing import List, Tuple, Optional
from langchain_core.language_models.llms import LLM
from config.settings import REWRITER_PROMPT, REWRITER_USE_LLM, REWRITER_LLM_TEMPERATURE, REWRITER_LLM_MAX_TOKENS
from utils.helpers import format_history_for_prompt

class QueryRewriter:
    """
    Rewrites the user's query based on chat history to make it standalone
    and more suitable for retrieval. Can use an LLM or a simple template.
    """
    def __init__(self, llm: Optional[LLM] = None):
        """
        Initializes the QueryRewriter.

        Args:
            llm: An optional Langchain LLM instance to use for rewriting.
                 Required if REWRITER_USE_LLM is True.
        """
        self.use_llm = REWRITER_USE_LLM
        self.llm = llm
        if self.use_llm and self.llm is None:
            raise ValueError("LLM instance must be provided if REWRITER_USE_LLM is True.")
        print(f"QueryRewriter initialized. Using LLM: {self.use_llm}")

    def _build_rewrite_prompt_llm(self, chat_history: List[Tuple[str, str]], current_query: str) -> str:
        """Builds the prompt for the LLM to rewrite the query."""
        formatted_history = format_history_for_prompt(chat_history)
        prompt = f"""Given the following conversation history:
<history>
{formatted_history}
</history>

And the latest user query:
<query>
{current_query}
</query>

{REWRITER_PROMPT}

Rewritten Query:"""
        return prompt

    def _rewrite_with_llm(self, chat_history: List[Tuple[str, str]], current_query: str) -> str:
        """Uses the provided LLM to rewrite the query."""
        if not self.llm: # Should not happen due to constructor check, but belts and suspenders
             print("Error: LLM not available for rewriting.")
             return current_query # Fallback to original query

        prompt = self._build_rewrite_prompt_llm(chat_history, current_query)
        print("Sending query rewrite request to LLM...")
        try:
            # Use specific temperature/max_tokens for rewriting task
            rewritten_query = self.llm.invoke(
                prompt,
                temperature=REWRITER_LLM_TEMPERATURE,
                max_tokens=REWRITER_LLM_MAX_TOKENS,
                stop=["\n"] # Stop at newline if possible to get just the query
            )
            # Basic cleaning: remove potential quotes or leading/trailing whitespace
            rewritten_query = rewritten_query.strip().strip('"').strip("'")
            print(f"LLM Rewritten Query: '{rewritten_query}'")
            if not rewritten_query or len(rewritten_query) < 5 :
                 print("Warning: LLM rewrite resulted in empty or very short query. Using original.")
                 return current_query
            return rewritten_query
        except Exception as e:
            print(f"Error during LLM query rewriting: {e}")
            return current_query # Fallback to original query on error

    def _rewrite_simple(self, chat_history: List[Tuple[str, str]], current_query: str) -> str:
        """A simple heuristic rewrite (e.g., just use the current query)."""
        print("Using simple rewriting (returning original query).")
        return current_query

    def rewrite_query(self, chat_history: List[Tuple[str, str]], current_query: str) -> str:
        """
        Rewrites the user query based on configuration and history.

        Args:
            chat_history: The list of (user_message, ai_message) tuples.
            current_query: The latest raw user query.

        Returns:
            The rewritten query string.
        """
        if not current_query:
            return "" # Handle empty query case

        print(f"Original query for rewriting: '{current_query}'")
        if self.use_llm:
            return self._rewrite_with_llm(chat_history, current_query)
        else:
            return self._rewrite_simple(chat_history, current_query)
