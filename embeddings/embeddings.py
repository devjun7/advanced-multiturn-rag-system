import os
import logging
from typing import List

import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config.settings import (
    EMBEDDING_MODEL_NAME,
    DEVICE,
    MAX_SEQ_LENGTH,
    DEVICE_NUMBER,
    # Optional: Add parameters like BATCH_SIZE if needed for SentenceTransformer
)

# Setup logger for this module
logger = logging.getLogger(__name__)

# Set CUDA device visibility early, if specified
if DEVICE_NUMBER:
    logger.info(f"Setting CUDA_VISIBLE_DEVICES='{DEVICE_NUMBER}' for embedding process.")
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NUMBER

class Embeddings(Embeddings):
    """
    Langchain compatible Embeddings class using SentenceTransformer models.

    Wraps a SentenceTransformer model to provide `embed_documents` and `embed_query`
    methods compatible with Langchain.
    """
    client: SentenceTransformer
    model_name: str
    device: str
    normalize_embeddings: bool

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = DEVICE,
        normalize_embeddings: bool = True,
        max_seq_length: int = MAX_SEQ_LENGTH,
        **kwargs # Allow pass-through for other SentenceTransformer args
    ):
        """
        Initializes the Embeddings wrapper.

        Args:
            model_name: Name/path of the SentenceTransformer model (e.g., 'all-MiniLM-L6-v2').
            device: Device to run the model on ('cuda', 'cpu', etc.). Auto-detects fallback.
            normalize_embeddings: Whether to normalize embeddings to unit length (recommended for semantic search).
            max_seq_length: Maximum sequence length the model can handle. Texts longer than this will be truncated.
            **kwargs: Additional arguments passed directly to the SentenceTransformer constructor.

        Raises:
            ValueError: If the specified device is 'cuda' but CUDA is not available.
            RuntimeError: If the SentenceTransformer model fails to load.
        """
        super().__init__(**kwargs) # Pass extra kwargs to parent if needed, though Embeddings has none
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.normalize_embeddings = normalize_embeddings
        logger.info(f"Initializing Embeddings with model: '{self.model_name}' on device: '{self.device}'")

        try:
            # Load the SentenceTransformer model
            self.client = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
                **kwargs # Pass through any extra arguments
            )
            # Set max sequence length (important for handling long documents)
            self.client.max_seq_length = max_seq_length
            logger.info(f"SentenceTransformer model '{self.model_name}' loaded successfully. Max sequence length: {max_seq_length}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{self.model_name}' from HuggingFace or path: {e}", exc_info=True)
            logger.error("Ensure the model name is correct, you have internet connection (if downloading), and necessary dependencies are installed.")
            # Fail fast if the core component (the model) cannot be loaded.
            raise RuntimeError(f"Failed to load SentenceTransformer model '{self.model_name}': {e}") from e

    def _resolve_device(self, requested_device: str) -> str:
        """Validates CUDA availability if requested, falling back to CPU if necessary."""
        if requested_device.lower().startswith("cuda") and not torch.cuda.is_available():
            logger.warning(f"CUDA device ('{requested_device}') requested but not available. Falling back to CPU.")
            return "cpu"
        logger.info(f"Using device: '{requested_device}' for embeddings.")
        return requested_device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents using the SentenceTransformer model.

        Args:
            texts: A list of strings (documents) to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
            Returns an empty list if the input `texts` is empty.

        Raises:
            TypeError: If `texts` is not a list of strings.
        """
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("Input must be a list of strings.")
        if not texts:
            logger.debug("embed_documents called with an empty list.")
            return []

        logger.info(f"Embedding {len(texts)} documents using '{self.model_name}'...")
        # SentenceTransformer encode handles batching internally.
        # Consider adding `batch_size` parameter to `encode` if performance tuning is needed.
        embeddings = self.client.encode(
            texts,
            convert_to_tensor=False, # Langchain expects lists of floats, not tensors
            normalize_embeddings=self.normalize_embeddings,
            device=self.device,
            # show_progress_bar=True # Optional: Show a progress bar for long lists
        )
        logger.info(f"Finished embedding {len(texts)} documents.")
        # Ensure the output is List[List[float]] (encode returns ndarray)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query text using the SentenceTransformer model.

        Args:
            text: The query string to embed.

        Returns:
            The embedding as a list of floats.
            Returns an empty list if the input `text` is empty or not a string.

        Raises:
            TypeError: If `text` is not a string.
        """
        if not isinstance(text, str):
             raise TypeError("Input must be a string.")
        if not text:
            logger.debug("embed_query called with an empty string.")
            # Returning empty list might be ambiguous; consider raising ValueError
            # raise ValueError("Cannot embed an empty string.")
            return []

        logger.info(f"Embedding query using '{self.model_name}'...")
        # Encode expects a list, even for a single query
        embedding = self.client.encode(
            [text],
            convert_to_tensor=False,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device
        )[0] # Get the first (and only) embedding from the result list
        logger.info("Finished embedding query.")
        return embedding.tolist()