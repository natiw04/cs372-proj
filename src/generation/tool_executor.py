"""
Tool Executor for Tribly AI Assistant.

Bridges Claude's tool calls to the retrieval system, executing
searches and formatting results for the agent.

Rubric Items:
- Implemented agentic system with tool calls (7 pts)
- Built multi-stage ML pipeline (7 pts - partial)
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable, Union

from .tools import ToolName
from ..retrieval.retriever import RetrievalResponse, RetrievalResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executes tool calls made by Claude.

    Maps tool names to retrieval functions and formats results
    appropriately for the agent to process.
    """

    def __init__(self, retriever=None):
        """
        Initialize the tool executor.

        Args:
            retriever: The retriever instance to use for searches.
                      If not provided, will be lazily loaded.
        """
        self._retriever = retriever
        self._tool_handlers: Dict[str, Callable] = {}
        self._setup_handlers()

    def _get_retriever(self):
        """Lazy load retriever if not provided."""
        if self._retriever is None:
            from ..retrieval.retriever import get_retriever
            self._retriever = get_retriever()
            if not self._retriever.is_ready():
                self._retriever.initialize()
        return self._retriever

    def _setup_handlers(self):
        """Set up tool name to handler mappings."""
        self._tool_handlers = {
            ToolName.SEARCH_REVIEWS.value: self._handle_search_reviews,
            ToolName.SEARCH_EVENTS.value: self._handle_search_events,
            ToolName.SEARCH_RESOURCES.value: self._handle_search_resources,
            ToolName.SEARCH_POSTS.value: self._handle_search_posts,
            ToolName.SEARCH_CLASSES.value: self._handle_search_classes,
            ToolName.SEARCH_TEACHERS.value: self._handle_search_teachers,
            ToolName.SEARCH_ALL.value: self._handle_search_all,
            ToolName.GET_DOCUMENT.value: self._handle_get_document,
        }

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Input parameters for the tool.

        Returns:
            JSON string with tool results.
        """
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool input: {json.dumps(tool_input, indent=2)}")

        if tool_name not in self._tool_handlers:
            return json.dumps({
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self._tool_handlers.keys())
            })

        try:
            handler = self._tool_handlers[tool_name]
            result = handler(tool_input)
            return self._format_result(result)
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return json.dumps({
                "error": str(e),
                "tool": tool_name
            })

    def _format_result(self, result: Any) -> str:
        """Format result as JSON string."""
        if isinstance(result, str):
            return result
        return json.dumps(result, indent=2, default=str)

    def _extract_results(
        self,
        response: Union[RetrievalResponse, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Extract results from RetrievalResponse or raw list.

        Converts RetrievalResult objects to dicts for formatting.
        """
        if isinstance(response, RetrievalResponse):
            results = []
            for r in response.results:
                results.append({
                    "id": r.id,
                    "document": r.content,
                    "similarity": r.score,
                    "final_score": r.score,
                    "collection": r.collection,
                    "metadata": r.metadata,
                    "method": r.method
                })
            return results
        elif isinstance(response, list):
            return response
        else:
            return []

    # ========================================================================
    # TOOL HANDLERS
    # ========================================================================

    def _handle_search_reviews(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_reviews tool."""
        retriever = self._get_retriever()

        query = input.get("query", "")
        limit = input.get("limit", 5)

        # Build filters
        filters = {}
        if input.get("professor_name"):
            # Will search in document text
            query = f"{query} {input['professor_name']}"
        if input.get("class_name"):
            query = f"{query} {input['class_name']}"

        # Execute search
        response = retriever.search_reviews(query, top_k=limit)
        results = self._extract_results(response)

        # Filter by min_rating if specified
        min_rating = input.get("min_rating")
        if min_rating:
            results = [
                r for r in results
                if r.get("metadata", {}).get("rating", 0) >= min_rating
            ]

        return {
            "query": query,
            "num_results": len(results),
            "results": self._format_search_results(results, "review")
        }

    def _handle_search_events(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_events tool."""
        retriever = self._get_retriever()

        query = input.get("query", "")
        limit = input.get("limit", 5)

        # Execute search
        response = retriever.search_events(query, top_k=limit)
        results = self._extract_results(response)

        # Filter by free_food if specified
        if input.get("free_food") is not None:
            results = [
                r for r in results
                if r.get("metadata", {}).get("free_food") == input["free_food"]
            ]

        return {
            "query": query,
            "num_results": len(results),
            "results": self._format_search_results(results, "event")
        }

    def _handle_search_resources(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_resources tool."""
        retriever = self._get_retriever()

        query = input.get("query", "")
        limit = input.get("limit", 5)

        # Add resource type to query if specified
        if input.get("resource_type"):
            query = f"{query} {input['resource_type']}"
        if input.get("class_name"):
            query = f"{query} {input['class_name']}"

        # Execute search
        response = retriever.search_resources(query, top_k=limit)
        results = self._extract_results(response)

        return {
            "query": query,
            "num_results": len(results),
            "results": self._format_search_results(results, "resource")
        }

    def _handle_search_posts(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_posts tool."""
        retriever = self._get_retriever()

        query = input.get("query", "")
        limit = input.get("limit", 5)

        # Execute search
        response = retriever.search_posts(query, top_k=limit)
        results = self._extract_results(response)

        # Filter by category if specified
        category = input.get("category")
        if category:
            results = [
                r for r in results
                if r.get("metadata", {}).get("category") == category
            ]

        return {
            "query": query,
            "num_results": len(results),
            "results": self._format_search_results(results, "post")
        }

    def _handle_search_classes(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_classes tool."""
        retriever = self._get_retriever()

        query = input.get("query", "")
        limit = input.get("limit", 5)

        # Add department to query if specified
        if input.get("department"):
            query = f"{input['department']} {query}"

        # Execute search on classes collection
        response = retriever.search(
            query=query,
            collections=["classes"],
            top_k=limit
        )
        results = self._extract_results(response)

        return {
            "query": query,
            "num_results": len(results),
            "results": self._format_search_results(results, "class")
        }

    def _handle_search_teachers(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_teachers tool."""
        retriever = self._get_retriever()

        query = input.get("query", "")
        limit = input.get("limit", 5)

        # Execute search on teachers collection
        response = retriever.search(
            query=query,
            collections=["teachers"],
            top_k=limit
        )
        results = self._extract_results(response)

        return {
            "query": query,
            "num_results": len(results),
            "results": self._format_search_results(results, "teacher")
        }

    def _handle_search_all(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_all tool."""
        retriever = self._get_retriever()

        query = input.get("query", "")
        limit = input.get("limit", 10)

        # Execute search across all collections
        response = retriever.search(query=query, top_k=limit)
        results = self._extract_results(response)

        return {
            "query": query,
            "num_results": len(results),
            "results": self._format_search_results(results, "mixed")
        }

    def _handle_get_document(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_document tool."""
        document_id = input.get("document_id")
        collection = input.get("collection")

        if not document_id or not collection:
            return {"error": "document_id and collection are required"}

        retriever = self._get_retriever()

        # Search for exact document by ID
        # This is a workaround since we don't have direct ID lookup
        results = retriever.vector_store.search(
            collection_name=collection,
            query_embedding=retriever.embedding_service.embed_text(""),
            top_k=100  # Get many results to find by ID
        )

        # Find the document by ID
        for result in results:
            if result.get("id") == document_id:
                return {
                    "document": {
                        "id": result.get("id"),
                        "collection": collection,
                        "content": result.get("document"),
                        "metadata": result.get("metadata", {})
                    }
                }

        return {"error": f"Document not found: {document_id} in {collection}"}

    # ========================================================================
    # RESULT FORMATTING
    # ========================================================================

    def _format_search_results(
        self,
        results: List[Dict[str, Any]],
        result_type: str
    ) -> List[Dict[str, Any]]:
        """Format search results for the agent."""
        formatted = []

        for result in results:
            formatted_result = {
                "id": result.get("id"),
                "type": result_type,
                "relevance_score": round(result.get("final_score", result.get("similarity", 0)), 3),
                "content": result.get("document", ""),
                "collection": result.get("collection", "")
            }

            # Add type-specific fields from metadata
            metadata = result.get("metadata", {})

            if result_type == "review":
                formatted_result.update({
                    "rating": metadata.get("rating"),
                    "helpful_count": metadata.get("helpful_count", 0),
                    "is_anonymous": metadata.get("is_anonymous", False)
                })
            elif result_type == "event":
                formatted_result.update({
                    "title": self._extract_title(result),
                    "location": metadata.get("location"),
                    "date_time": metadata.get("date_time"),
                    "free_food": metadata.get("free_food", False)
                })
            elif result_type == "resource":
                formatted_result.update({
                    "title": self._extract_title(result),
                    "resource_type": metadata.get("type"),
                    "upvotes": metadata.get("upvotes", 0)
                })
            elif result_type == "post":
                formatted_result.update({
                    "title": self._extract_title(result),
                    "category": metadata.get("category"),
                    "upvotes": metadata.get("upvotes", 0),
                    "comment_count": metadata.get("comment_count", 0)
                })
            elif result_type == "class":
                formatted_result.update({
                    "title": self._extract_title(result),
                    "avg_rating": metadata.get("avg_rating"),
                    "review_count": metadata.get("review_count", 0)
                })
            elif result_type == "teacher":
                formatted_result.update({
                    "name": metadata.get("name") or self._extract_title(result),
                    "avg_rating": metadata.get("avg_rating"),
                    "review_count": metadata.get("review_count", 0)
                })

            formatted.append(formatted_result)

        return formatted

    def _extract_title(self, result: Dict[str, Any]) -> str:
        """Extract title from result document or metadata."""
        metadata = result.get("metadata", {})

        # Try metadata fields
        for field in ["title", "name"]:
            if field in metadata and metadata[field]:
                return metadata[field]

        # Try to extract from document text
        doc = result.get("document", "")
        if doc:
            # Get first line or first part before delimiter
            parts = doc.split("|")
            if parts:
                return parts[0].strip()[:100]

        return "Untitled"


# Singleton instance
_default_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    """Get the default tool executor instance."""
    global _default_executor
    if _default_executor is None:
        _default_executor = ToolExecutor()
    return _default_executor


def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Convenience function to execute a tool."""
    return get_tool_executor().execute(tool_name, tool_input)
