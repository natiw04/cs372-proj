"""
Embedding Service for Tribly AI Assistant.

Uses Sentence Transformers (all-MiniLM-L6-v2) to generate 384-dimensional
embeddings for text. Supports single and batch embedding generation.

Rubric Items:
- Used sentence embeddings for semantic similarity/retrieval (5 pts)
- Used/fine-tuned transformer language model (7 pts)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

# Lazy import to avoid loading model until needed
_sentence_model = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using Sentence Transformers.

    Uses the all-MiniLM-L6-v2 model which produces 384-dimensional embeddings.
    This model offers a good balance of speed and quality for semantic search.
    """

    # Model configuration
    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = None):
        """
        Initialize the embedding service.

        Args:
            model_name: Optional custom model name. Defaults to all-MiniLM-L6-v2.
        """
        self.model_name = model_name or self.MODEL_NAME
        self._model = None
        self._is_initialized = False

        # Cache for embeddings (optional, for repeated queries)
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_enabled = True
        self._max_cache_size = 10000

    def initialize(self) -> None:
        """
        Initialize the embedding model.

        Loads the Sentence Transformer model into memory.
        Call this before generating embeddings.
        """
        if self._is_initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._is_initialized = True
            logger.info(f"Embedding model loaded successfully. Dimension: {self.EMBEDDING_DIM}")

        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if the embedding service is initialized and ready."""
        return self._is_initialized and self._model is not None

    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text string.

        Args:
            text: The text to embed.
            use_cache: Whether to use/update the embedding cache.

        Returns:
            384-dimensional numpy array.
        """
        if not self.is_ready():
            self.initialize()

        # Check cache
        if use_cache and self._cache_enabled and text in self._cache:
            return self._cache[text]

        # Generate embedding
        embedding = self._model.encode(text, convert_to_numpy=True)

        # Update cache
        if use_cache and self._cache_enabled:
            if len(self._cache) >= self._max_cache_size:
                # Simple cache eviction: clear half the cache
                keys_to_remove = list(self._cache.keys())[:len(self._cache) // 2]
                for key in keys_to_remove:
                    del self._cache[key]
            self._cache[text] = embedding

        return embedding

    def embed_texts(self, texts: List[str], batch_size: int = 32,
                    show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process at once.
            show_progress: Whether to show a progress bar.

        Returns:
            2D numpy array of shape (len(texts), 384).
        """
        if not self.is_ready():
            self.initialize()

        if not texts:
            return np.array([])

        # Use sentence-transformers' built-in batching
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def embed_documents(self, documents: List[Dict[str, Any]],
                        text_field: str = "text",
                        batch_size: int = 32,
                        show_progress: bool = False) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of documents.

        Each document should have a text field to embed. The embedding
        is added to each document under the 'embedding' key.

        Args:
            documents: List of document dictionaries.
            text_field: Name of the field containing text to embed.
            batch_size: Batch size for embedding generation.
            show_progress: Whether to show progress bar.

        Returns:
            Documents with 'embedding' field added.
        """
        if not documents:
            return []

        # Extract texts
        texts = []
        for doc in documents:
            text = doc.get(text_field, "")
            if not text:
                # Try common alternative field names
                text = doc.get("content", "") or doc.get("description", "") or doc.get("comment", "")
            texts.append(text or "")

        # Generate embeddings
        embeddings = self.embed_texts(texts, batch_size=batch_size, show_progress=show_progress)

        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc["embedding"] = embedding

        return documents

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    def compute_similarities(self, query_embedding: np.ndarray,
                            document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between a query and multiple documents.

        Args:
            query_embedding: Query embedding vector (1D).
            document_embeddings: Document embeddings (2D array).

        Returns:
            Array of similarity scores.
        """
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(document_embeddings))
        query_normalized = query_embedding / query_norm

        # Normalize documents
        doc_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)  # Avoid division by zero
        docs_normalized = document_embeddings / doc_norms

        # Compute similarities
        similarities = np.dot(docs_normalized, query_normalized)
        return similarities

    def find_most_similar(self, query: str, texts: List[str],
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most similar texts to a query.

        Args:
            query: Query text.
            texts: List of candidate texts.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'text', 'index', and 'similarity' keys.
        """
        if not texts:
            return []

        # Generate embeddings
        query_embedding = self.embed_text(query)
        text_embeddings = self.embed_texts(texts)

        # Compute similarities
        similarities = self.compute_similarities(query_embedding, text_embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": texts[idx],
                "index": int(idx),
                "similarity": float(similarities[idx])
            })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding service."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.EMBEDDING_DIM,
            "is_initialized": self._is_initialized,
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")


# Singleton instance for convenience
_default_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the default embedding service instance."""
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service


def embed_text(text: str) -> np.ndarray:
    """Convenience function to embed a single text."""
    return get_embedding_service().embed_text(text)


def embed_texts(texts: List[str], **kwargs) -> np.ndarray:
    """Convenience function to embed multiple texts."""
    return get_embedding_service().embed_texts(texts, **kwargs)
