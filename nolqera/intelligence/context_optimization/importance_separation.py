from __future__ import annotations

from typing import List, Tuple

from nolqera.intelligence.semantic_search.models import SemanticSearchResult


class ImportanceSeparator:
    """
    Separates retrieval results into important and unnecessary
    information using a configurable score threshold.

    Results are expected to already be ranked by the retrieval
    pipeline.
    """

    def __init__(self, importance_threshold: float = 0.5):
        if not isinstance(importance_threshold, (int, float)):
            raise TypeError(
                "importance_threshold must be numeric"
            )

        if not 0.0 <= importance_threshold <= 1.0:
            raise ValueError(
                "importance_threshold must be between 0 and 1"
            )

        self.importance_threshold = float(
            importance_threshold
        )

    def separate(
        self,
        results: List[SemanticSearchResult],
    ) -> Tuple[
        List[SemanticSearchResult],
        List[SemanticSearchResult],
    ]:
        """
        Return (important, unnecessary) results.

        Results at or above the threshold are important.
        Results below the threshold are unnecessary.
        """
        if not isinstance(results, list):
            raise TypeError("results must be a list")

        for result in results:
            if not isinstance(result, SemanticSearchResult):
                raise TypeError(
                    "all results must be SemanticSearchResult instances"
                )

        important = []
        unnecessary = []

        for result in results:
            if result.score >= self.importance_threshold:
                important.append(result)
            else:
                unnecessary.append(result)

        return important, unnecessary