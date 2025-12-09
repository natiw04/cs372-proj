#!/usr/bin/env python3
"""
Run evaluation metrics for Tribly AI Assistant.

This script evaluates retrieval quality across different search methods
and generates comparison results.

Usage:
    python scripts/run_evaluation.py                    # Run full evaluation
    python scripts/run_evaluation.py --output results/  # Save to specific directory
    python scripts/run_evaluation.py --methods semantic,keyword  # Specific methods
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    RetrievalEvaluator,
    save_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation metrics"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/metrics",
        help="Output directory for results"
    )
    parser.add_argument(
        "--methods", "-m",
        type=str,
        default="semantic,keyword,hybrid",
        help="Comma-separated list of methods to evaluate"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Parse methods
    methods = [m.strip() for m in args.methods.split(",")]

    print("\n" + "=" * 60)
    print(" TRIBLY AI ASSISTANT - RETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"\nMethods to evaluate: {', '.join(methods)}")
    print(f"Output directory: {args.output}")

    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = RetrievalEvaluator()

    # Run comparison
    print(f"\nRunning evaluation on {len(evaluator.test_queries)} test queries...")
    print("This may take a minute...\n")

    comparisons = evaluator.compare_methods(methods)

    # Print results table
    evaluator.print_comparison_table(comparisons)

    # Get full results
    results = evaluator.run_full_evaluation()

    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "methods_evaluated": methods,
        "num_test_queries": len(evaluator.test_queries)
    }

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "evaluation_results.json"
    save_results(results, results_file)

    print(f"\nResults saved to: {results_file}")

    # Print summary for README
    print("\n" + "=" * 60)
    print(" SUMMARY FOR README.md")
    print("=" * 60)
    print("\nCopy this table to your README:\n")
    print("| Metric | Semantic | Keyword | Hybrid |")
    print("|--------|----------|---------|--------|")

    sem = results["summary"].get("semantic", {})
    kw = results["summary"].get("keyword", {})
    hyb = results["summary"].get("hybrid", {})

    if sem and kw and hyb:
        print(f"| Precision@5 | {sem.get('precision@5', 0):.2f} | {kw.get('precision@5', 0):.2f} | {hyb.get('precision@5', 0):.2f} |")
        print(f"| Recall@5 | {sem.get('recall@5', 0):.2f} | {kw.get('recall@5', 0):.2f} | {hyb.get('recall@5', 0):.2f} |")
        print(f"| MRR | {sem.get('mrr', 0):.2f} | {kw.get('mrr', 0):.2f} | {hyb.get('mrr', 0):.2f} |")
        print(f"| Avg Latency (ms) | {sem.get('avg_latency_ms', 0):.0f} | {kw.get('avg_latency_ms', 0):.0f} | {hyb.get('avg_latency_ms', 0):.0f} |")

    print("\n" + "=" * 60)
    print(" EVALUATION COMPLETE")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
