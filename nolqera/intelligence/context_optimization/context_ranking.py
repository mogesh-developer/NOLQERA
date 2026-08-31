from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


@dataclass(frozen=True)
class RankedContext:
    result: SemanticSearchResult
    relevance_score: float
    importance_score: float
    ranking_score: float


class ContextRanker:
    """
    Ranks retrieved context using relevance and importance scores.
    """

    def __init__(
        self,
        relevance_weight: float = 0.7,
        importance_weight: float = 0.3,
    ):
        self._validate_weight(
            relevance_weight,
            "relevance_weight",
        )
        self._validate_weight(
            importance_weight,
            "importance_weight",
        )

        total = (
            relevance_weight
            + importance_weight
        )

        if total <= 0:
            raise ValueError(
                "at least one weight must be greater than zero"
            )

        self.relevance_weight = (
            relevance_weight / total
        )
        self.importance_weight = (
            importance_weight / total
        )

    @staticmethod
    def _validate_weight(
        weight: float,
        name: str,
    ) -> None:
        if not isinstance(weight, (int, float)):
            raise TypeError(
                f"{name} must be numeric"
            )

        if weight < 0:
            raise ValueError(
                f"{name} cannot be negative"
            )

    @staticmethod
    def _validate_score(
        score: float,
        name: str,
    ) -> None:
        if not isinstance(score, (int, float)):
            raise TypeError(
                f"{name} must be numeric"
            )

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1"
            )

    def rank(
        self,
        results: List[SemanticSearchResult],
        importance_scores: List[float],
    ) -> List[RankedContext]:
        """
        Rank results using relevance and importance scores.

        The relevance score comes from SemanticSearchResult.score.
        The corresponding importance score is supplied separately.
        """

        if not isinstance(results, list):
            raise TypeError("results must be a list")

        if not isinstance(importance_scores, list):
            raise TypeError(
                "importance_scores must be a list"
            )

        if len(results) != len(importance_scores):
            raise ValueError(
                "results and importance_scores "
                "must have the same length"
            )

        for result in results:
            if not isinstance(
                result,
                SemanticSearchResult,
            ):
                raise TypeError(
                    "all results must be "
                    "SemanticSearchResult instances"
                )

        ranked = []

        for result, importance in zip(
            results,
            importance_scores,
        ):
            self._validate_score(
                result.score,
                "relevance score",
            )

            self._validate_score(
                importance,
                "importance score",
            )

            ranking_score = (
                self.relevance_weight
                * result.score
                + self.importance_weight
                * importance
            )

            ranked.append(
                RankedContext(
                    result=result,
                    relevance_score=result.score,
                    importance_score=importance,
                    ranking_score=ranking_score,
                )
            )

        ranked.sort(
            key=lambda item: item.ranking_score,
            reverse=True,
        )

        return ranked