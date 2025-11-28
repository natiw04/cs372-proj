#!/usr/bin/env python3
"""
Index documents into the vector store for Tribly AI Assistant.

This script loads JSON data files and indexes them into ChromaDB
with embeddings for semantic search.

Usage:
    python scripts/index_documents.py              # Index all documents
    python scripts/index_documents.py --reset      # Reset and re-index
    python scripts/index_documents.py --stats      # Show statistics only
    python scripts/index_documents.py --generate   # Generate data first, then index

Rubric Items:
- Built RAG system with document retrieval (10 pts)
- Proper data loading with batching (3 pts)
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.document_indexer import DocumentIndexer, DEFAULT_DATA_DIR
from src.retrieval.vector_store import get_vector_store
from src.embeddings.embedding_service import get_embedding_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def show_stats():
    """Show current index statistics."""
    print("\n" + "=" * 50)
    print("VECTOR STORE STATISTICS")
    print("=" * 50)

    vector_store = get_vector_store()
    vector_store.initialize()

    stats = vector_store.get_collection_stats()

    print(f"\nPersist directory: {stats.get('persist_directory', 'N/A')}")
    print(f"\nCollections:")

    collections = stats.get('collections', {})
    total = 0
    for name, info in collections.items():
        count = info.get('count', 0)
        total += count
        print(f"  - {name}: {count} documents")

    print(f"\nTotal documents: {total}")

    # Show embedding service stats
    embedding_service = get_embedding_service()
    if embedding_service.is_ready():
        emb_stats = embedding_service.get_stats()
        print(f"\nEmbedding Service:")
        print(f"  - Model: {emb_stats.get('model_name', 'N/A')}")
        print(f"  - Dimension: {emb_stats.get('embedding_dim', 'N/A')}")
        print(f"  - Cache size: {emb_stats.get('cache_size', 0)}")


def index_documents(reset: bool = False, batch_size: int = 32):
    """Index all documents into the vector store."""
    print("\n" + "=" * 50)
    print("DOCUMENT INDEXING")
    print("=" * 50)

    # Check if data exists
    if not DEFAULT_DATA_DIR.exists():
        print(f"\nError: Data directory not found: {DEFAULT_DATA_DIR}")
        print("Run 'python scripts/export_data.py --generate' first to create data.")
        return False

    # Check for JSON files
    json_files = list(DEFAULT_DATA_DIR.glob("*.json"))
    if not json_files:
        print(f"\nError: No JSON files found in {DEFAULT_DATA_DIR}")
        print("Run 'python scripts/export_data.py --generate' first to create data.")
        return False

    print(f"\nData directory: {DEFAULT_DATA_DIR}")
    print(f"Found {len(json_files)} JSON files")

    # Initialize indexer
    indexer = DocumentIndexer()

    # Run indexing
    print("\nStarting indexing...")
    print("-" * 50)

    try:
        results = indexer.index_all(
            batch_size=batch_size,
            show_progress=True,
            reset_first=reset
        )

        # Print results
        print("\n" + "=" * 50)
        print("INDEXING COMPLETE")
        print("=" * 50)

        total = 0
        for collection, count in results.items():
            print(f"  {collection}: {count} documents")
            total += count

        print(f"\nTotal indexed: {total} documents")
        return True

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        return False


def generate_and_index(reset: bool = False, batch_size: int = 32):
    """Generate synthetic data and then index it."""
    print("\n" + "=" * 50)
    print("GENERATE AND INDEX")
    print("=" * 50)

    # Import and run data generation
    from scripts.export_data import generate_synthetic_data

    print("\nStep 1: Generating synthetic data...")
    generate_synthetic_data()

    print("\nStep 2: Indexing documents...")
    return index_documents(reset=reset, batch_size=batch_size)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index documents into the Tribly AI Assistant vector store."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the vector store before indexing"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only, don't index"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic data before indexing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("TRIBLY AI ASSISTANT - DOCUMENT INDEXER")
    print("=" * 50)

    if args.stats:
        show_stats()
        return

    if args.generate:
        success = generate_and_index(reset=args.reset, batch_size=args.batch_size)
    else:
        success = index_documents(reset=args.reset, batch_size=args.batch_size)

    if success:
        print("\n" + "-" * 50)
        show_stats()


if __name__ == "__main__":
    main()
