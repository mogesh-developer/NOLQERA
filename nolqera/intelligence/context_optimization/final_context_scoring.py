from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nolqera.intelligence.semantic_search.models import SemanticSearchResult


@dataclass(frozen=True)
class FinalContextScore:
    result: SemanticSearchResult
    score: float


class FinalContextScorer:
    """
    Calculates a final context score from retrieval quality signals.

    final_score =
        relevance_weight * relevance
        + diversity_weight * diversity
        - redundancy_weight * redundancy
    """

    def __init__(
        self,
        relevance_weight: float = 0.6,
        diversity_weight: float = 0.3,
        redundancy_weight: float = 0.1,
    ) -> None:

        weights = (
            relevance_weight,
            diversity_weight,
            redundancy_weight,
        )

        if any(not isinstance(weight, (int, float)) for weight in weights):
            raise TypeError("weights must be numeric")

        if any(weight < 0 for weight in weights):
            raise ValueError("weights cannot be negative")

        if sum(weights) <= 0:
            raise ValueError("at least one weight must be positive")

        self.relevance_weight = relevance_weight
        self.diversity_weight = diversity_weight
        self.redundancy_weight = redundancy_weight

    def score(
        self,
        result: SemanticSearchResult,
        *,
        diversity: float = 1.0,
        redundancy: float = 0.0,
    ) -> FinalContextScore:

        if not isinstance(result, SemanticSearchResult):
            raise TypeError(
                "result must be a SemanticSearchResult"
            )

        self._validate_signal(diversity, "diversity")
        self._validate_signal(redundancy, "redundancy")

        final_score = (
            self.relevance_weight * result.score
            + self.diversity_weight * diversity
            - self.redundancy_weight * redundancy
        )

        final_score = max(0.0, min(1.0, final_score))

        return FinalContextScore(
            result=result,
            score=final_score,
        )

    def rank(
        self,
        results: List[SemanticSearchResult],
        *,
        diversity_scores: List[float] | None = None,
        redundancy_scores: List[float] | None = None,
    ) -> List[FinalContextScore]:

        if not isinstance(results, list):
            raise TypeError("results must be a list")

        if not all(
            isinstance(result, SemanticSearchResult)
            for result in results
        ):
            raise TypeError(
                "all results must be SemanticSearchResult"
            )

        if diversity_scores is None:
            diversity_scores = [1.0] * len(results)

        if redundancy_scores is None:
            redundancy_scores = [0.0] * len(results)

        if len(diversity_scores) != len(results):
            raise ValueError(
                "diversity_scores length must match results"
            )

        if len(redundancy_scores) != len(results):
            raise ValueError(
                "redundancy_scores length must match results"
            )

        scored = [
            self.score(
                result,
                diversity=diversity,
                redundancy=redundancy,
            )
            for result, diversity, redundancy
            in zip(
                results,
                diversity_scores,
                redundancy_scores,
            )
        ]

        return sorted(
            scored,
            key=lambda item: item.score,
            reverse=True,
        )

    @staticmethod
    def _validate_signal(
        value: float,
        name: str,
    ) -> None:

        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1"
            )