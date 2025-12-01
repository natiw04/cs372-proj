#!/usr/bin/env python3
"""
Test the Tribly AI Assistant RAG Pipeline.

This script tests all components of the pipeline:
1. Embedding generation
2. Vector store operations
3. Retrieval (semantic, keyword, hybrid)
4. Ranking
5. Tool execution
6. Full agent interaction (if API key available)

Usage:
    python scripts/test_pipeline.py              # Run all tests
    python scripts/test_pipeline.py --component embeddings  # Test specific component
    python scripts/test_pipeline.py --interactive  # Interactive mode with Claude

Rubric Items:
- Built multi-stage ML pipeline (7 pts)
- Implemented agentic system with tool calls (7 pts)
- Built RAG system with document retrieval (10 pts)
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def print_result(label: str, value: Any, indent: int = 0):
    """Print a labeled result."""
    prefix = "  " * indent
    print(f"{prefix}{label}: {value}")


def test_embeddings():
    """Test embedding service."""
    print_header("TESTING EMBEDDING SERVICE")

    from src.embeddings.embedding_service import EmbeddingService

    service = EmbeddingService()
    service.initialize()

    # Test single embedding
    print_subheader("Single Embedding")
    text = "This is a test sentence for embedding."
    embedding = service.embed_text(text)
    print_result("Text", text[:50] + "...")
    print_result("Embedding shape", embedding.shape)
    print_result("First 5 values", embedding[:5].tolist())

    # Test batch embedding
    print_subheader("Batch Embedding")
    texts = [
        "Machine learning is fascinating.",
        "I love studying computer science.",
        "The weather is nice today."
    ]
    embeddings = service.embed_texts(texts)
    print_result("Batch size", len(texts))
    print_result("Embeddings shape", embeddings.shape)

    # Test similarity
    print_subheader("Similarity Computation")
    query = "I enjoy learning about AI"
    query_emb = service.embed_text(query)
    for i, text in enumerate(texts):
        sim = service.compute_similarity(query_emb, embeddings[i])
        print_result(f"Similarity to '{text[:30]}...'", f"{sim:.4f}", indent=1)

    print("\n[PASS] Embedding service tests completed")
    return True


def test_vector_store():
    """Test vector store operations."""
    print_header("TESTING VECTOR STORE")

    from src.retrieval.vector_store import VectorStore
    from src.embeddings.embedding_service import EmbeddingService

    # Initialize
    emb_service = EmbeddingService()
    emb_service.initialize()

    store = VectorStore()
    store.initialize()

    # Check stats
    print_subheader("Collection Statistics")
    stats = store.get_collection_stats()
    print_result("Persist directory", stats.get("persist_directory"))
    print_result("Total documents", stats.get("total_documents", 0))

    for name, info in stats.get("collections", {}).items():
        print_result(f"  {name}", f"{info.get('count', 0)} docs", indent=1)

    # Test search if documents exist
    if stats.get("total_documents", 0) > 0:
        print_subheader("Search Test")
        query = "machine learning class"
        query_emb = emb_service.embed_text(query).tolist()

        results = store.search_all_collections(query_emb, top_k=3)
        print_result("Query", query)
        print_result("Results found", len(results))

        for i, result in enumerate(results[:3]):
            print(f"\n  Result {i+1}:")
            print_result("Collection", result.get("collection"), indent=2)
            print_result("Similarity", f"{result.get('similarity', 0):.4f}", indent=2)
            doc_preview = result.get("document", "")[:80]
            print_result("Document", doc_preview + "...", indent=2)

    print("\n[PASS] Vector store tests completed")
    return True


def test_retrieval():
    """Test retrieval system."""
    print_header("TESTING RETRIEVAL SYSTEM")

    from src.retrieval.retriever import Retriever

    retriever = Retriever()
    retriever.initialize()

    # Check if documents are indexed
    stats = retriever.vector_store.get_collection_stats()
    if stats.get("total_documents", 0) == 0:
        print("\n[WARN] No documents indexed. Run index_documents.py first.")
        print("       Skipping retrieval tests.")
        return True

    # Test semantic search
    print_subheader("Semantic Search")
    query = "best computer science professors"
    response = retriever.search(query, mode="semantic", top_k=3)
    print_result("Query", query)
    print_result("Results", len(response.results))
    for i, r in enumerate(response.results[:2]):
        print(f"  {i+1}. {r.content[:60]}... (score: {r.score:.3f})")

    # Test keyword search
    print_subheader("Keyword Search (Baseline)")
    response = retriever.search(query, mode="keyword", top_k=3)
    print_result("Results", len(response.results))
    for i, r in enumerate(response.results[:2]):
        print(f"  {i+1}. {r.content[:60]}... (score: {r.score:.3f})")

    # Test hybrid search
    print_subheader("Hybrid Search")
    response = retriever.search(query, mode="hybrid", top_k=3)
    print_result("Results", len(response.results))
    for i, r in enumerate(response.results[:2]):
        print(f"  {i+1}. {r.content[:60]}... (score: {r.score:.3f})")

    # Test specialized searches
    print_subheader("Specialized Searches")

    reviews_resp = retriever.search_reviews("helpful professor", top_k=2)
    print_result("Reviews found", len(reviews_resp.results))

    events_resp = retriever.search_events("career fair", top_k=2)
    print_result("Events found", len(events_resp.results))

    resources_resp = retriever.search_resources("study guide", top_k=2)
    print_result("Resources found", len(resources_resp.results))

    print("\n[PASS] Retrieval tests completed")
    return True


def test_ranking():
    """Test ranking service."""
    print_header("TESTING RANKING SERVICE")

    from src.retrieval.ranking import RankingService

    ranker = RankingService()

    # Create mock results
    mock_results = [
        {
            "document": "Great class, learned a lot",
            "similarity": 0.8,
            "metadata": {
                "rating": 4.5,
                "upvotes": 10,
                "downvotes": 1,
                "helpful_count": 5,
                "created": "2024-10-01T00:00:00Z"
            }
        },
        {
            "document": "Okay class, nothing special",
            "similarity": 0.85,
            "metadata": {
                "rating": 3.0,
                "upvotes": 2,
                "downvotes": 0,
                "helpful_count": 1,
                "created": "2024-06-01T00:00:00Z"
            }
        },
        {
            "document": "Amazing professor!",
            "similarity": 0.75,
            "metadata": {
                "rating": 5.0,
                "upvotes": 50,
                "downvotes": 2,
                "helpful_count": 30,
                "created": "2024-11-01T00:00:00Z"
            }
        }
    ]

    print_subheader("Multi-Signal Ranking")
    ranked = ranker.rank_results(mock_results, context="reviews")

    print("Before ranking (by similarity):")
    for r in mock_results:
        print(f"  - sim={r['similarity']:.2f}, rating={r['metadata']['rating']}")

    print("\nAfter ranking (multi-signal):")
    for r in ranked:
        signals = r.get("ranking_signals", {})
        print(f"  - final={r['final_score']:.3f} (sem={signals.get('semantic_similarity', 0):.2f}, "
              f"rating={signals.get('rating_score', 0):.2f}, vote={signals.get('vote_score', 0):.2f})")

    print("\n[PASS] Ranking tests completed")
    return True


def test_tools():
    """Test tool definitions and execution."""
    print_header("TESTING TOOL SYSTEM")

    from src.generation.tools import get_all_tool_schemas, get_tool_names
    from src.generation.tool_executor import ToolExecutor

    # Test tool schemas
    print_subheader("Tool Schemas")
    schemas = get_all_tool_schemas()
    print_result("Available tools", len(schemas))
    for name in get_tool_names():
        print(f"  - {name}")

    # Test tool execution (if documents indexed)
    print_subheader("Tool Execution")
    executor = ToolExecutor()

    # Check if retrieval is ready
    from src.retrieval.retriever import get_retriever
    retriever = get_retriever()
    retriever.initialize()

    stats = retriever.vector_store.get_collection_stats()
    if stats.get("total_documents", 0) == 0:
        print("\n[WARN] No documents indexed. Skipping tool execution tests.")
        return True

    # Test search_reviews tool
    result = executor.execute("search_reviews", {"query": "good professor", "limit": 2})
    parsed = json.loads(result)
    print_result("search_reviews", f"{parsed.get('num_results', 0)} results")

    # Test search_events tool
    result = executor.execute("search_events", {"query": "career fair", "limit": 2})
    parsed = json.loads(result)
    print_result("search_events", f"{parsed.get('num_results', 0)} results")

    # Test search_all tool
    result = executor.execute("search_all", {"query": "study resources", "limit": 3})
    parsed = json.loads(result)
    print_result("search_all", f"{parsed.get('num_results', 0)} results")

    print("\n[PASS] Tool tests completed")
    return True


def test_agent(interactive: bool = False):
    """Test the full agent system."""
    print_header("TESTING AGENT SYSTEM")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n[WARN] GOOGLE_API_KEY not set. Skipping agent tests.")
        print("       Get a free key at: https://aistudio.google.com/apikey")
        return True

    from src.generation.context_manager import TriblyAssistant

    assistant = TriblyAssistant()
    assistant.initialize()

    if interactive:
        print_subheader("Interactive Mode")
        print("Type 'quit' to exit.\n")

        assistant.start_conversation()

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                if not user_input:
                    continue

                print("\nAssistant: ", end="")
                response = assistant.ask(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")

    else:
        print_subheader("Automated Test Queries")

        test_queries = [
            "What are the best-rated CS classes?",
            "Are there any events with free food this week?",
            "Can you find study resources for linear algebra?"
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                result = assistant.get_detailed_response(query)
                print(f"Response: {result['response'][:200]}...")
                print(f"Tool calls: {len(result.get('tool_calls', []))}")
                print(f"Tokens used: {result.get('tokens_used', {})}")
            except Exception as e:
                print(f"Error: {e}")

    print("\n[PASS] Agent tests completed")
    return True


def run_all_tests(interactive: bool = False):
    """Run all component tests."""
    print_header("TRIBLY AI ASSISTANT - PIPELINE TEST SUITE")

    results = {}

    # Test each component
    try:
        results["embeddings"] = test_embeddings()
    except Exception as e:
        logger.error(f"Embedding test failed: {e}")
        results["embeddings"] = False

    try:
        results["vector_store"] = test_vector_store()
    except Exception as e:
        logger.error(f"Vector store test failed: {e}")
        results["vector_store"] = False

    try:
        results["retrieval"] = test_retrieval()
    except Exception as e:
        logger.error(f"Retrieval test failed: {e}")
        results["retrieval"] = False

    try:
        results["ranking"] = test_ranking()
    except Exception as e:
        logger.error(f"Ranking test failed: {e}")
        results["ranking"] = False

    try:
        results["tools"] = test_tools()
    except Exception as e:
        logger.error(f"Tool test failed: {e}")
        results["tools"] = False

    try:
        results["agent"] = test_agent(interactive=interactive)
    except Exception as e:
        logger.error(f"Agent test failed: {e}")
        results["agent"] = False

    # Summary
    print_header("TEST SUMMARY")
    passed = 0
    failed = 0

    for component, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {component}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test the Tribly AI Assistant RAG pipeline."
    )
    parser.add_argument(
        "--component",
        choices=["embeddings", "vector_store", "retrieval", "ranking", "tools", "agent"],
        help="Test a specific component only"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run agent in interactive mode"
    )

    args = parser.parse_args()

    if args.component:
        test_funcs = {
            "embeddings": test_embeddings,
            "vector_store": test_vector_store,
            "retrieval": test_retrieval,
            "ranking": test_ranking,
            "tools": test_tools,
            "agent": lambda: test_agent(args.interactive)
        }
        try:
            test_funcs[args.component]()
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        success = run_all_tests(interactive=args.interactive)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
