from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


@dataclass(frozen=True)
class NormalizedRetrievalResult:
    """
    A semantic search result with a normalized score.
    """

    index: int
    text: str
    score: float
    original_score: float


class ScoreNormalizer:
    """
    Normalizes retrieval scores into the [0, 1] range.

    Uses min-max normalization across the supplied candidate set.
    """

    def normalize(
        self,
        results: List[SemanticSearchResult],
    ) -> List[NormalizedRetrievalResult]:

        if not isinstance(results, list):
            raise TypeError(
                "results must be a list"
            )

        if not results:
            return []

        for result in results:
            if not isinstance(
                result,
                SemanticSearchResult,
            ):
                raise TypeError(
                    "results must contain "
                    "SemanticSearchResult objects"
                )

        scores = [
            result.score
            for result in results
        ]

        minimum = min(scores)
        maximum = max(scores)

        # All scores are identical.
        # There is no relative difference between them,
        # so assign every result the maximum normalized score.
        if maximum == minimum:
            normalized_scores = [
                1.0
                for _ in scores
            ]
        else:
            normalized_scores = [
                (score - minimum)
                / (maximum - minimum)
                for score in scores
            ]

        return [
            NormalizedRetrievalResult(
                index=result.index,
                text=result.text,
                score=normalized_score,
                original_score=result.score,
            )
            for result, normalized_score
            in zip(
                results,
                normalized_scores,
            )
        ]