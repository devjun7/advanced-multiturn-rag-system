import os
import logging
import time
import sys
from typing import List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from config.settings import RETRIEVER_K
from utils.helpers import ensure_dir_exists

# Setup logger for this module
logger = logging.getLogger(__name__)

# Security Note regarding FAISS deserialization
FAISS_SECURITY_WARNING = ("Loading FAISS index using `allow_dangerous_deserialization=True`. "
                          "Ensure the index file comes from a trusted source to prevent potential security risks.")

class FAISSVectorStoreManager:
    """
    Manages a FAISS vector store, including loading, saving, adding documents,
    and creating a retriever. Handles incremental updates to the index.
    """
    def __init__(
        self,
        embedding_function: Embeddings,
        # metadata_path is not directly used by FAISS load/save, consider removing if unused
        index_path_prefix: str,
        retriever_k: int = RETRIEVER_K
    ):
        """
        Initializes the FAISS manager.

        Args:
            embedding_function: The Langchain Embeddings object to use.
            index_path: Path prefix for the FAISS index (.faiss and .pkl files).
            retriever_k: Default number of documents to retrieve.
        """
        if not embedding_function:
            raise ValueError("Embedding function must be provided.")

        self.embedding_function = embedding_function
        self.index_path = index_path_prefix
        self.retriever_k = retriever_k
        self.vector_store: Optional[FAISS] = self._load_vector_store()

    def _load_vector_store(self) -> Optional[FAISS]:
        """Loads the FAISS index and docstore (.pkl) from disk if they exist."""
        ensure_dir_exists(os.path.dirname(self.index_path))
        index_file = self.index_path + ".faiss"
        pkl_file = self.index_path + ".pkl"
        index_name = os.path.basename(self.index_path)
        folder_path = os.path.dirname(self.index_path)

        if os.path.exists(index_file) and os.path.exists(pkl_file):
            logger.info(f"Attempting to load FAISS index from folder: '{folder_path}', index name: '{index_name}'")
            logger.warning(FAISS_SECURITY_WARNING)
            try:
                store = FAISS.load_local(
                    folder_path=folder_path,
                    embeddings=self.embedding_function,
                    index_name=index_name,
                    allow_dangerous_deserialization=True # Required for loading pickled data
                )
                logger.info("FAISS index loaded successfully.")
                return store
            except FileNotFoundError:
                 # This can happen if index_name is wrong or files moved unexpectedly
                 logger.error(f"FAISS load_local failed: Index files (.faiss, .pkl) not found at expected location.", exc_info=True)
                 return None
            except EOFError:
                 logger.error(f"FAISS load_local failed: EOFError encountered. Index files might be corrupted or incomplete.", exc_info=True)
                 # Attempt to remove corrupted files to allow creating a new one later
                 self._try_remove_index_files()
                 return None
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}. Index might be corrupted or incompatible.", exc_info=True)
                # Attempt cleanup
                self._try_remove_index_files()
                return None
        else:
            logger.info("No existing FAISS index found (.faiss and .pkl files). A new one will be created upon adding documents.")
            return None

    def _try_remove_index_files(self):
        """Attempts to remove potentially corrupted index files."""
        index_file = self.index_path + ".faiss"
        pkl_file = self.index_path + ".pkl"
        logger.warning(f"Attempting to remove potentially corrupted index files: {index_file}, {pkl_file}")
        try:
            if os.path.exists(index_file): os.remove(index_file)
            if os.path.exists(pkl_file): os.remove(pkl_file)
            logger.info("Successfully removed potentially corrupted index files.")
        except OSError as e:
             logger.error(f"Failed to remove corrupted index files: {e}", exc_info=True)

    def _save_vector_store(self):
        """Saves the FAISS index and docstore (.pkl) to disk."""
        if self.vector_store:
            folder_path = os.path.dirname(self.index_path)
            index_name = os.path.basename(self.index_path)
            ensure_dir_exists(folder_path)
            logger.info(f"Saving FAISS index to folder: '{folder_path}', index name: '{index_name}'")
            try:
                self.vector_store.save_local(
                     folder_path=folder_path,
                     index_name=index_name
                )
                logger.info("FAISS index saved successfully.")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}", exc_info=True)
        else:
            logger.warning("No vector store initialized. Nothing to save.")

    def add_documents(self, documents: List[Document], batch_size: int = 64):
        """
        Adds documents to the vector store. Creates a new store if one doesn't exist,
        otherwise adds incrementally to the existing store.

        Args:
            documents: A list of Langchain Document objects.
            batch_size: Number of documents to embed and add in each batch.

        Raises:
            ValueError: If input `documents` list is empty.
            RuntimeError: If the vector store fails to be created or updated.
        """
        if not documents:
            logger.warning("No documents provided to add.")
            return

        logger.info(f"Adding {len(documents)} documents to FAISS store (batch size: {batch_size})...")
        start_time = time.time() # Added time import

        if self.vector_store is None:
            logger.info("No existing index found. Creating new FAISS index from provided documents.")
            try:
                # Create index from the first batch, then add the rest
                # FAISS.from_documents handles batching internally to some extent, but explicit batching for large lists is safer
                self.vector_store = FAISS.from_documents(documents, self.embedding_function)
                logger.info(f"New FAISS index created and {len(documents)} documents added.")
                self._save_vector_store() # Save after initial creation
            except Exception as e:
                logger.error(f"Error creating initial FAISS index: {e}", exc_info=True)
                self.vector_store = None # Ensure store is None on failure
                raise RuntimeError("Failed to create initial FAISS index") from e
        else:
            logger.info("Adding documents incrementally to existing FAISS index.")
            total_docs = len(documents)
            num_batches = (total_docs + batch_size - 1) // batch_size
            try:
                # Manual batching using add_texts for more control (original implementation)
                for i in range(0, total_docs, batch_size):
                    batch = documents[i:min(i + batch_size, total_docs)]
                    if not batch: continue
                    batch_num = (i // batch_size) + 1
                    logger.debug(f"Adding batch {batch_num}/{num_batches} ({len(batch)} documents)..." )
                    try:
                        texts = [doc.page_content for doc in batch]
                        metadata = [doc.metadata for doc in batch]
                        ids = self.vector_store.add_texts(texts=texts, metadatas=metadata)
                        logger.debug(f"Added batch {batch_num}/{num_batches}. First 5 new IDs: {ids[:5]}...")
                    except Exception as batch_e:
                         logger.error(f"Error adding batch {batch_num}/{num_batches} to FAISS index: {batch_e}", exc_info=True)
                logger.info(f"Successfully added {total_docs} documents in {num_batches} batches.")
                self._save_vector_store() # Save after adding all batches
            except Exception as e:
                 logger.error(f"Error during incremental addition of documents: {e}", exc_info=True)

        end_time = time.time()
        logger.info(f"Document addition completed in {end_time - start_time:.2f} seconds.")


    def get_retriever(self, k: Optional[int] = None) -> Optional[VectorStoreRetriever]:
        """
        Gets a Langchain retriever object for the vector store.

        Args:
            k: The number of documents to retrieve (overrides default if provided).
               If None, uses the default `self.retriever_k`.

        Returns:
            A Langchain VectorStoreRetriever instance, or None if the vector store
            is not initialized or available.
        """
        if self.vector_store:
            search_k = k if k is not None else self.retriever_k
            logger.debug(f"Creating retriever with k={search_k}")
            try:
                return self.vector_store.as_retriever(search_kwargs={"k": search_k})
            except Exception as e:
                logger.error(f"Error creating retriever from vector store: {e}", exc_info=True)
                return None
        else:
            logger.warning("Vector store not initialized. Cannot create retriever.")
            return None

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Performs a similarity search directly on the vector store.

        Args:
            query: The query string.
            k: The number of documents to retrieve (overrides default `self.retriever_k` if provided).

        Returns:
            A list of relevant Document objects, or an empty list if the store
            is not initialized or the search fails.
        """
        if self.vector_store:
            search_k = k if k is not None else self.retriever_k
            logger.debug(f"Performing similarity search with k={search_k} for query: '{query[:100]}...'")
            try:
                results = self.vector_store.similarity_search(query, k=search_k)
                logger.info(f"Similarity search retrieved {len(results)} documents.")
                return results
            except Exception as e:
                logger.error(f"Error during similarity search: {e}", exc_info=True)
                return []
        else:
            logger.warning("Vector store not initialized. Cannot perform similarity search.")
            return []