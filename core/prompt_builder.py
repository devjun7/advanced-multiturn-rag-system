from typing import List, Tuple
from langchain_core.documents import Document
from config.settings import DEFAULT_SYSTEM_PROMPT
from utils.helpers import format_docs

class PromptBuilder:
    """
    Constructs the final prompt to be sent to the LLM, combining
    system instructions, processed history, retrieved context, and the query.
    """
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_prompt(
        self,
        processed_history: str,
        retrieved_docs: List[Document],
        rewritten_query: str
    ) -> str:
        """
        Builds the final prompt string.

        Args:
            processed_history: Formatted string of relevant chat history.
            retrieved_docs: List of retrieved context documents.
            rewritten_query: The (potentially rewritten) user query.

        Returns:
            The complete prompt string for the LLM.
        """
        formatted_context = format_docs(retrieved_docs)

        prompt = f"""{self.system_prompt}

Relevant Conversation History:
<history>
{processed_history}
</history>

Context Documents:
<context>
{formatted_context}
</context>

User Query:
{rewritten_query}

Assistant Response:"""
        return prompt