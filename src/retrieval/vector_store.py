"""
Vector Store using ChromaDB for Tribly AI Assistant.

Provides persistent vector storage and similarity search capabilities.
Supports multiple collections for different data types (reviews, events, etc.).

Rubric Items:
- Used significant software framework for applied ML not covered in class (ChromaDB) (5 pts)
- Built RAG system with document retrieval (10 pts - partial)
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Default persist directory
DEFAULT_PERSIST_DIR = Path(__file__).parent.parent.parent / "data" / "chroma"


class VectorStore:
    """
    ChromaDB-based vector store for semantic search.

    Manages multiple collections for different document types
    (reviews, events, posts, resources, etc.).
    """

    # Collection names for different data types
    COLLECTIONS = [
        "reviews",
        "events",
        "hangouts",
        "posts",
        "resources",
        "classes",
        "teachers",
        "groups"
    ]

    def __init__(self, persist_directory: str = None):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the ChromaDB database.
                             Defaults to data/chroma/ in project root.
        """
        self.persist_directory = Path(persist_directory) if persist_directory else DEFAULT_PERSIST_DIR
        self._client = None
        self._collections: Dict[str, Any] = {}
        self._is_initialized = False

    def initialize(self) -> None:
        """
        Initialize ChromaDB client and create collections.

        Creates the persist directory if it doesn't exist and
        initializes all document collections.
        """
        if self._is_initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            # Create persist directory
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client with persistence
            logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create or get collections
            for collection_name in self.COLLECTIONS:
                self._collections[collection_name] = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                logger.debug(f"Collection '{collection_name}' ready")

            self._is_initialized = True
            logger.info(f"Vector store initialized with {len(self.COLLECTIONS)} collections")

        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if the vector store is initialized."""
        return self._is_initialized

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized before operations."""
        if not self.is_ready():
            self.initialize()

    def add_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: List[str] = None
    ) -> int:
        """
        Add documents with embeddings to a collection.

        Args:
            collection_name: Name of the collection (e.g., 'reviews', 'events').
            documents: List of document dictionaries with metadata.
            embeddings: List of embedding vectors.
            ids: Optional list of document IDs. Auto-generated if not provided.

        Returns:
            Number of documents added.
        """
        self._ensure_initialized()

        if collection_name not in self._collections:
            raise ValueError(f"Unknown collection: {collection_name}. Valid: {self.COLLECTIONS}")

        if len(documents) != len(embeddings):
            raise ValueError(f"documents ({len(documents)}) and embeddings ({len(embeddings)}) must have same length")

        if not documents:
            return 0

        collection = self._collections[collection_name]

        # Generate IDs if not provided
        if ids is None:
            ids = [doc.get("id", f"{collection_name}_{i}") for i, doc in enumerate(documents)]

        # Prepare documents for ChromaDB
        # ChromaDB stores: ids, embeddings, metadatas, documents (text)
        metadatas = []
        texts = []

        for doc in documents:
            # Extract text content
            text = self._extract_text(doc)
            texts.append(text)

            # Prepare metadata (ChromaDB only supports str, int, float, bool)
            metadata = self._prepare_metadata(doc)
            metadatas.append(metadata)

        # Add to collection
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            logger.info(f"Added {len(documents)} documents to '{collection_name}'")
            return len(documents)

        except Exception as e:
            logger.error(f"Error adding documents to {collection_name}: {e}")
            raise

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 10,
        where: Dict[str, Any] = None,
        where_document: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in a collection.

        Args:
            collection_name: Name of the collection to search.
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            where: Optional metadata filter (e.g., {"category": "academic"}).
            where_document: Optional document text filter.

        Returns:
            List of results with 'id', 'document', 'metadata', 'distance', 'similarity'.
        """
        self._ensure_initialized()

        if collection_name not in self._collections:
            raise ValueError(f"Unknown collection: {collection_name}")

        collection = self._collections[collection_name]

        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }

        if where:
            query_params["where"] = where
        if where_document:
            query_params["where_document"] = where_document

        # Execute query
        try:
            results = collection.query(**query_params)
        except Exception as e:
            logger.error(f"Error searching {collection_name}: {e}")
            return []

        # Format results
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results.get("documents") else [None] * len(ids)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            distances = results["distances"][0] if results.get("distances") else [0] * len(ids)

            for i, doc_id in enumerate(ids):
                # Convert distance to similarity (ChromaDB returns L2 distance for cosine)
                # For cosine space, distance = 1 - similarity
                distance = distances[i]
                similarity = 1 - distance

                formatted_results.append({
                    "id": doc_id,
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "distance": distance,
                    "similarity": similarity,
                    "collection": collection_name
                })

        return formatted_results

    def search_all_collections(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        collections: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple collections and merge results.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results per collection.
            collections: List of collections to search. Defaults to all.

        Returns:
            Merged results sorted by similarity.
        """
        self._ensure_initialized()

        collections_to_search = collections or self.COLLECTIONS
        all_results = []

        for collection_name in collections_to_search:
            if collection_name in self._collections:
                results = self.search(collection_name, query_embedding, top_k=top_k)
                all_results.extend(results)

        # Sort by similarity and return top results
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete all documents from a collection.

        Args:
            collection_name: Name of the collection to clear.

        Returns:
            True if successful.
        """
        self._ensure_initialized()

        if collection_name not in self._collections:
            return False

        try:
            self._client.delete_collection(collection_name)
            self._collections[collection_name] = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    def get_collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """
        Get statistics about collections.

        Args:
            collection_name: Specific collection, or None for all.

        Returns:
            Dictionary with collection statistics.
        """
        self._ensure_initialized()

        if collection_name:
            if collection_name not in self._collections:
                return {"error": f"Unknown collection: {collection_name}"}

            collection = self._collections[collection_name]
            return {
                "name": collection_name,
                "count": collection.count()
            }

        # Return stats for all collections
        stats = {
            "persist_directory": str(self.persist_directory),
            "collections": {}
        }
        total_count = 0

        for name, collection in self._collections.items():
            count = collection.count()
            stats["collections"][name] = {"count": count}
            total_count += count

        stats["total_documents"] = total_count
        return stats

    def reset(self) -> None:
        """Reset the entire vector store (delete all data)."""
        self._ensure_initialized()

        try:
            self._client.reset()
            self._collections.clear()

            # Recreate collections
            for collection_name in self.COLLECTIONS:
                self._collections[collection_name] = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )

            logger.info("Vector store reset complete")
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
            raise

    def _extract_text(self, document: Dict[str, Any]) -> str:
        """Extract searchable text from a document."""
        # Try common text fields in order of preference
        text_fields = [
            "text", "content", "text_content", "description",
            "comment", "title", "bio", "research_description"
        ]

        text_parts = []
        for field in text_fields:
            if field in document and document[field]:
                text_parts.append(str(document[field]))

        return " ".join(text_parts) if text_parts else ""

    def _prepare_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB storage.

        ChromaDB only supports str, int, float, bool as metadata values.
        """
        metadata = {}

        # Fields to include in metadata
        metadata_fields = [
            "id", "category", "type", "rating", "avg_rating",
            "upvotes", "downvotes", "helpful_count", "review_count",
            "is_anonymous", "free_food", "everyone",
            "created", "updated", "date_time",
            "author", "creator", "group", "class_name", "professor_name"
        ]

        for field in metadata_fields:
            if field in document:
                value = document[field]

                # Handle different types
                if value is None:
                    continue
                elif isinstance(value, (str, int, float, bool)):
                    metadata[field] = value
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings
                    if value and isinstance(value[0], str):
                        metadata[field] = ",".join(value)
                else:
                    # Convert other types to string
                    metadata[field] = str(value)

        return metadata


# Singleton instance
_default_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get the default vector store instance."""
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store
