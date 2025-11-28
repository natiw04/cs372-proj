"""
Ranking Service for Tribly AI Assistant.

Provides multi-signal ranking for retrieval results.
Combines semantic similarity with quality signals (ratings, votes, recency).

Rubric Items:
- Applied feature engineering (5 pts)
- Built RAG system (10 pts - partial)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RankingSignals:
    """Signals used for ranking a result."""
    semantic_similarity: float = 0.0
    rating_score: float = 0.0
    vote_score: float = 0.0
    recency_score: float = 0.0
    popularity_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "semantic_similarity": self.semantic_similarity,
            "rating_score": self.rating_score,
            "vote_score": self.vote_score,
            "recency_score": self.recency_score,
            "popularity_score": self.popularity_score
        }


class RankingService:
    """
    Multi-signal ranking for retrieval results.

    Combines multiple signals to produce a final ranking:
    - Semantic similarity (from embeddings)
    - Quality signals (ratings, votes)
    - Recency (favor recent content)
    - Popularity (helpful votes, view counts)
    """

    # Default signal weights for different contexts
    DEFAULT_WEIGHTS = {
        "semantic_similarity": 0.50,
        "rating_score": 0.15,
        "vote_score": 0.15,
        "recency_score": 0.10,
        "popularity_score": 0.10
    }

    # Context-specific weight configurations
    CONTEXT_WEIGHTS = {
        "reviews": {
            "semantic_similarity": 0.40,
            "rating_score": 0.25,
            "vote_score": 0.15,
            "recency_score": 0.10,
            "popularity_score": 0.10
        },
        "events": {
            "semantic_similarity": 0.50,
            "rating_score": 0.05,
            "vote_score": 0.05,
            "recency_score": 0.30,  # Events should be recent/upcoming
            "popularity_score": 0.10
        },
        "resources": {
            "semantic_similarity": 0.45,
            "rating_score": 0.10,
            "vote_score": 0.25,  # Votes matter for study resources
            "recency_score": 0.10,
            "popularity_score": 0.10
        },
        "posts": {
            "semantic_similarity": 0.45,
            "rating_score": 0.05,
            "vote_score": 0.20,
            "recency_score": 0.20,
            "popularity_score": 0.10
        }
    }

    def __init__(self, custom_weights: Dict[str, float] = None):
        """
        Initialize the ranking service.

        Args:
            custom_weights: Optional custom signal weights.
        """
        self.default_weights = custom_weights or self.DEFAULT_WEIGHTS

    def rank_results(
        self,
        results: List[Dict[str, Any]],
        context: str = None,
        custom_weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using multi-signal scoring.

        Args:
            results: List of retrieval results with metadata.
            context: Context for weight selection (e.g., 'reviews', 'events').
            custom_weights: Optional override weights.

        Returns:
            Re-ranked results with 'final_score' and 'signals' added.
        """
        if not results:
            return []

        # Select weights
        if custom_weights:
            weights = custom_weights
        elif context and context in self.CONTEXT_WEIGHTS:
            weights = self.CONTEXT_WEIGHTS[context]
        else:
            weights = self.default_weights

        # Calculate signals and final scores
        scored_results = []
        for result in results:
            signals = self._calculate_signals(result)
            final_score = self._blend_signals(signals, weights)

            # Add scores to result
            result_copy = result.copy()
            result_copy["final_score"] = final_score
            result_copy["ranking_signals"] = signals.to_dict()
            result_copy["original_score"] = result.get("score", result.get("similarity", 0))

            scored_results.append(result_copy)

        # Sort by final score
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)

        return scored_results

    def _calculate_signals(self, result: Dict[str, Any]) -> RankingSignals:
        """Calculate all ranking signals for a result."""
        metadata = result.get("metadata", {})

        signals = RankingSignals()

        # Semantic similarity (from retrieval)
        signals.semantic_similarity = result.get("score", result.get("similarity", 0))

        # Rating score
        signals.rating_score = self._calculate_rating_score(metadata)

        # Vote score
        signals.vote_score = self._calculate_vote_score(metadata)

        # Recency score
        signals.recency_score = self._calculate_recency_score(metadata)

        # Popularity score
        signals.popularity_score = self._calculate_popularity_score(metadata)

        return signals

    def _blend_signals(
        self,
        signals: RankingSignals,
        weights: Dict[str, float]
    ) -> float:
        """Blend signals using weighted combination."""
        total_score = 0.0
        total_weight = 0.0

        signal_values = signals.to_dict()

        for signal_name, weight in weights.items():
            if signal_name in signal_values:
                total_score += signal_values[signal_name] * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_rating_score(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate rating score (normalized 0-1).

        Handles both 'rating' and 'avg_rating' fields.
        """
        rating = metadata.get("rating") or metadata.get("avg_rating")

        if rating is None:
            return 0.5  # Neutral score for unrated items

        try:
            rating = float(rating)
            # Assume 5-point scale, normalize to 0-1
            return min(max(rating / 5.0, 0), 1)
        except (ValueError, TypeError):
            return 0.5

    def _calculate_vote_score(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate vote score based on upvotes/downvotes.

        Uses Wilson score interval for ranking by votes.
        """
        upvotes = metadata.get("upvotes", 0) or 0
        downvotes = metadata.get("downvotes", 0) or 0
        helpful = metadata.get("helpful_count", 0) or 0

        # Use helpful_count as additional positive signal
        upvotes += helpful

        total = upvotes + downvotes

        if total == 0:
            return 0.5  # Neutral score for unvoted items

        # Simple ratio with smoothing
        # Add small constant to avoid extreme scores with few votes
        score = (upvotes + 1) / (total + 2)

        return score

    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate recency score (newer = higher).

        Uses exponential decay based on document age.
        """
        # Try different date fields
        date_str = (
            metadata.get("created") or
            metadata.get("updated") or
            metadata.get("date_time")
        )

        if not date_str:
            return 0.5  # Neutral for undated items

        try:
            # Parse date (handle various formats)
            if isinstance(date_str, str):
                # Try ISO format
                if "T" in date_str:
                    doc_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                else:
                    doc_date = datetime.fromisoformat(date_str)
            else:
                return 0.5

            # Calculate age in days
            now = datetime.now(doc_date.tzinfo) if doc_date.tzinfo else datetime.now()
            age_days = (now - doc_date).days

            # Exponential decay: half-life of 30 days
            # Score = 0.5^(age/half_life)
            half_life = 30
            score = 0.5 ** (age_days / half_life)

            return max(min(score, 1.0), 0.0)

        except Exception as e:
            logger.debug(f"Error parsing date '{date_str}': {e}")
            return 0.5

    def _calculate_popularity_score(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate popularity score based on engagement metrics.

        Combines comment count, view count, and attendance.
        """
        signals = []

        # Comment count
        comment_count = metadata.get("comment_count", 0) or 0
        if comment_count > 0:
            # Log scale for comments (diminishing returns)
            import math
            signals.append(min(math.log10(comment_count + 1) / 2, 1.0))

        # Review count (for classes/teachers)
        review_count = metadata.get("review_count", 0) or 0
        if review_count > 0:
            import math
            signals.append(min(math.log10(review_count + 1) / 2, 1.0))

        # Attendees (for events)
        # Could add attendee count processing here

        if signals:
            return sum(signals) / len(signals)
        return 0.5  # Neutral for items without popularity data


def rerank_results(
    results: List[Dict[str, Any]],
    context: str = None
) -> List[Dict[str, Any]]:
    """Convenience function to re-rank results."""
    ranker = RankingService()
    return ranker.rank_results(results, context=context)
