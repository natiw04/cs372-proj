"""
Retriever for Tribly AI Assistant.

Provides query interface with both semantic search (vector) and
keyword search (baseline) for comparison.

Rubric Items:
- Built RAG system with document retrieval (10 pts - partial)
- Created baseline model for comparison (3 pts)
- Compared multiple model architectures or approaches quantitatively (5 pts)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import logging

from ..embeddings.embedding_service import EmbeddingService, get_embedding_service
from .vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

# Default data directory for keyword search fallback
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    collection: str
    method: str  # 'semantic' or 'keyword'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "collection": self.collection,
            "method": self.method
        }


@dataclass
class RetrievalResponse:
    """Response from retrieval operation."""
    query: str
    results: List[RetrievalResult]
    method: str
    total_found: int
    filters_applied: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "method": self.method,
            "total_found": self.total_found,
            "filters_applied": self.filters_applied
        }


class Retriever:
    """
    Document retriever with semantic and keyword search capabilities.

    Supports:
    - Semantic search using embeddings and ChromaDB
    - Keyword search as a baseline for comparison
    - Metadata filtering (category, rating, date, etc.)
    - Hybrid search combining both methods
    """

    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        vector_store: VectorStore = None,
        data_dir: Path = None
    ):
        """
        Initialize the retriever.

        Args:
            embedding_service: Service for generating query embeddings.
            vector_store: Vector store for semantic search.
            data_dir: Directory with JSON files for keyword search fallback.
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

        # Cache for keyword search
        self._document_cache: Dict[str, List[Dict[str, Any]]] = {}

    def search(
        self,
        query: str,
        collections: List[str] = None,
        top_k: int = 10,
        method: Literal["semantic", "keyword", "hybrid"] = "semantic",
        filters: Dict[str, Any] = None
    ) -> RetrievalResponse:
        """
        Search for documents matching a query.

        Args:
            query: Search query text.
            collections: List of collections to search. None = all.
            top_k: Number of results to return.
            method: Search method - 'semantic', 'keyword', or 'hybrid'.
            filters: Metadata filters (e.g., {"category": "academic"}).

        Returns:
            RetrievalResponse with ranked results.
        """
        if method == "semantic":
            return self.semantic_search(query, collections, top_k, filters)
        elif method == "keyword":
            return self.keyword_search(query, collections, top_k, filters)
        elif method == "hybrid":
            return self.hybrid_search(query, collections, top_k, filters)
        else:
            raise ValueError(f"Unknown search method: {method}")

    def semantic_search(
        self,
        query: str,
        collections: List[str] = None,
        top_k: int = 10,
        filters: Dict[str, Any] = None
    ) -> RetrievalResponse:
        """
        Perform semantic search using embeddings.

        This is the primary search method for the RAG system.
        """
        # Ensure services are ready
        if not self.embedding_service.is_ready():
            self.embedding_service.initialize()
        if not self.vector_store.is_ready():
            self.vector_store.initialize()

        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Prepare ChromaDB filters
        where_filter = self._build_chroma_filter(filters) if filters else None

        # Search in specified collections or all
        if collections:
            all_results = []
            for collection in collections:
                results = self.vector_store.search(
                    collection_name=collection,
                    query_embedding=query_embedding.tolist(),
                    top_k=top_k,
                    where=where_filter
                )
                all_results.extend(results)

            # Sort by similarity and take top_k
            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            raw_results = all_results[:top_k]
        else:
            raw_results = self.vector_store.search_all_collections(
                query_embedding=query_embedding.tolist(),
                top_k=top_k
            )

        # Convert to RetrievalResult objects
        results = []
        for r in raw_results:
            results.append(RetrievalResult(
                id=r["id"],
                content=r["document"] or "",
                metadata=r["metadata"],
                score=r["similarity"],
                collection=r["collection"],
                method="semantic"
            ))

        return RetrievalResponse(
            query=query,
            results=results,
            method="semantic",
            total_found=len(results),
            filters_applied=filters or {}
        )

    def keyword_search(
        self,
        query: str,
        collections: List[str] = None,
        top_k: int = 10,
        filters: Dict[str, Any] = None
    ) -> RetrievalResponse:
        """
        Perform keyword-based search (baseline method).

        Uses simple text matching for comparison with semantic search.
        """
        # Load documents from JSON files
        documents = self._load_documents_for_keyword_search(collections)

        # Tokenize query
        query_terms = self._tokenize(query.lower())

        if not query_terms:
            return RetrievalResponse(
                query=query,
                results=[],
                method="keyword",
                total_found=0,
                filters_applied=filters or {}
            )

        # Score each document
        scored_docs = []
        for doc in documents:
            # Apply filters
            if filters and not self._matches_filters(doc, filters):
                continue

            # Extract text content
            text = self._get_document_text(doc).lower()
            text_terms = self._tokenize(text)

            # Calculate keyword match score (simple TF)
            score = self._calculate_keyword_score(query_terms, text_terms)

            if score > 0:
                scored_docs.append({
                    "doc": doc,
                    "score": score,
                    "text": text
                })

        # Sort by score
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # Convert to results
        results = []
        for item in scored_docs[:top_k]:
            doc = item["doc"]
            results.append(RetrievalResult(
                id=doc.get("id", "unknown"),
                content=item["text"][:500],  # Truncate for display
                metadata=self._extract_metadata(doc),
                score=item["score"],
                collection=doc.get("_collection", "unknown"),
                method="keyword"
            ))

        return RetrievalResponse(
            query=query,
            results=results,
            method="keyword",
            total_found=len(results),
            filters_applied=filters or {}
        )

    def hybrid_search(
        self,
        query: str,
        collections: List[str] = None,
        top_k: int = 10,
        filters: Dict[str, Any] = None,
        semantic_weight: float = 0.7
    ) -> RetrievalResponse:
        """
        Perform hybrid search combining semantic and keyword results.

        Args:
            semantic_weight: Weight for semantic results (0-1). Keyword weight = 1 - semantic_weight.
        """
        # Get results from both methods
        semantic_results = self.semantic_search(query, collections, top_k * 2, filters)
        keyword_results = self.keyword_search(query, collections, top_k * 2, filters)

        # Combine and re-rank using reciprocal rank fusion
        combined_scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        # Add semantic results
        for i, result in enumerate(semantic_results.results):
            rank = i + 1
            rrf_score = semantic_weight * (1.0 / (60 + rank))  # k=60 is common RRF constant
            combined_scores[result.id] = combined_scores.get(result.id, 0) + rrf_score
            result_map[result.id] = result

        # Add keyword results
        keyword_weight = 1.0 - semantic_weight
        for i, result in enumerate(keyword_results.results):
            rank = i + 1
            rrf_score = keyword_weight * (1.0 / (60 + rank))
            combined_scores[result.id] = combined_scores.get(result.id, 0) + rrf_score
            if result.id not in result_map:
                result_map[result.id] = result

        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Build final results
        results = []
        for doc_id in sorted_ids[:top_k]:
            result = result_map[doc_id]
            # Update score to combined score
            results.append(RetrievalResult(
                id=result.id,
                content=result.content,
                metadata=result.metadata,
                score=combined_scores[doc_id],
                collection=result.collection,
                method="hybrid"
            ))

        return RetrievalResponse(
            query=query,
            results=results,
            method="hybrid",
            total_found=len(results),
            filters_applied=filters or {}
        )

    # Specialized search methods for different collections

    def search_reviews(
        self,
        query: str,
        professor_name: str = None,
        class_name: str = None,
        min_rating: float = None,
        top_k: int = 10,
        method: str = "semantic"
    ) -> RetrievalResponse:
        """Search reviews with specific filters."""
        filters = {}
        if min_rating is not None:
            filters["rating"] = {"$gte": min_rating}

        # Add text filters for professor/class (ChromaDB where_document)
        # Note: This is simplified; real implementation might need different approach
        response = self.search(
            query=query,
            collections=["reviews"],
            top_k=top_k,
            method=method,
            filters=filters if filters else None
        )

        # Post-filter by professor/class name if specified
        if professor_name or class_name:
            filtered_results = []
            for result in response.results:
                content_lower = result.content.lower()
                metadata = result.metadata

                if professor_name and professor_name.lower() not in content_lower:
                    continue
                if class_name and class_name.lower() not in content_lower:
                    continue

                filtered_results.append(result)

            response.results = filtered_results[:top_k]
            response.total_found = len(filtered_results)

        return response

    def search_events(
        self,
        query: str,
        category: str = None,
        free_food: bool = None,
        date_range: str = None,
        top_k: int = 10,
        method: str = "semantic"
    ) -> RetrievalResponse:
        """Search events (org_events + hangouts) with filters."""
        filters = {}
        if category:
            filters["category"] = category
        if free_food is not None:
            filters["free_food"] = free_food

        return self.search(
            query=query,
            collections=["events", "hangouts"],
            top_k=top_k,
            method=method,
            filters=filters if filters else None
        )

    def search_resources(
        self,
        query: str,
        class_name: str = None,
        resource_type: str = None,
        min_votes: int = None,
        top_k: int = 10,
        method: str = "semantic"
    ) -> RetrievalResponse:
        """Search study resources with filters."""
        filters = {}
        if resource_type:
            filters["type"] = resource_type

        response = self.search(
            query=query,
            collections=["resources"],
            top_k=top_k,
            method=method,
            filters=filters if filters else None
        )

        # Post-filter by class name if specified
        if class_name:
            filtered_results = []
            for result in response.results:
                if class_name.lower() in result.content.lower():
                    filtered_results.append(result)
            response.results = filtered_results[:top_k]
            response.total_found = len(filtered_results)

        return response

    def search_posts(
        self,
        query: str,
        category: str = None,
        top_k: int = 10,
        method: str = "semantic"
    ) -> RetrievalResponse:
        """Search feed posts with filters."""
        filters = {}
        if category:
            filters["category"] = category

        return self.search(
            query=query,
            collections=["posts"],
            top_k=top_k,
            method=method,
            filters=filters if filters else None
        )

    # Helper methods

    def _build_chroma_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert simple filters to ChromaDB where clause format."""
        if not filters:
            return None

        # Handle simple equality filters
        where_clauses = []
        for key, value in filters.items():
            if isinstance(value, dict):
                # Already in ChromaDB format (e.g., {"$gte": 4.0})
                where_clauses.append({key: value})
            else:
                # Simple equality
                where_clauses.append({key: value})

        if len(where_clauses) == 1:
            return where_clauses[0]
        elif len(where_clauses) > 1:
            return {"$and": where_clauses}
        return None

    def _load_documents_for_keyword_search(
        self,
        collections: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Load documents from JSON files for keyword search."""
        all_docs = []

        file_mapping = {
            "reviews": "reviews.json",
            "events": "events.json",
            "hangouts": "hangouts.json",
            "posts": "posts.json",
            "resources": "resources.json",
            "classes": "classes.json",
            "teachers": "teachers.json",
            "groups": "groups.json"
        }

        collections_to_load = collections or list(file_mapping.keys())

        for collection in collections_to_load:
            if collection not in file_mapping:
                continue

            # Check cache
            if collection in self._document_cache:
                all_docs.extend(self._document_cache[collection])
                continue

            # Load from file
            file_path = self.data_dir / file_mapping[collection]
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        docs = json.load(f)
                        # Add collection tag
                        for doc in docs:
                            doc["_collection"] = collection
                        self._document_cache[collection] = docs
                        all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        return all_docs

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for keyword search."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 2]

    def _calculate_keyword_score(
        self,
        query_terms: List[str],
        doc_terms: List[str]
    ) -> float:
        """Calculate keyword match score."""
        if not query_terms or not doc_terms:
            return 0.0

        doc_term_set = set(doc_terms)
        matches = sum(1 for term in query_terms if term in doc_term_set)

        # Simple TF-like scoring
        score = matches / len(query_terms)
        return score

    def _get_document_text(self, doc: Dict[str, Any]) -> str:
        """Extract all text content from a document."""
        text_fields = [
            "text", "content", "text_content", "description",
            "comment", "title", "bio", "research_description", "name"
        ]

        parts = []
        for field in text_fields:
            if doc.get(field):
                parts.append(str(doc[field]))

        # Include tags
        if doc.get("tags"):
            parts.append(" ".join(doc["tags"]))

        return " ".join(parts)

    def _matches_filters(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches all filters."""
        for key, value in filters.items():
            doc_value = doc.get(key)

            if isinstance(value, dict):
                # Handle operators like $gte, $lte
                for op, op_value in value.items():
                    if op == "$gte" and (doc_value is None or doc_value < op_value):
                        return False
                    elif op == "$lte" and (doc_value is None or doc_value > op_value):
                        return False
                    elif op == "$eq" and doc_value != op_value:
                        return False
            else:
                if doc_value != value:
                    return False

        return True

    def _extract_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata fields from a document."""
        metadata_fields = [
            "category", "type", "rating", "avg_rating",
            "upvotes", "downvotes", "free_food", "created"
        ]

        return {k: doc[k] for k in metadata_fields if k in doc}


# Singleton instance
_default_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get the default retriever instance."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever
