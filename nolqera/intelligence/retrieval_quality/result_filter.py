from typing import List

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


class ResultFilter:
    """
    Filters semantic search results using a minimum score threshold.

    Results with a score greater than or equal to min_score
    are retained. Results below min_score are removed.
    """

    def filter(
        self,
        results: List[SemanticSearchResult],
        min_score: float,
    ) -> List[SemanticSearchResult]:

        if not isinstance(results, list):
            raise TypeError("results must be a list")

        if not isinstance(min_score, (int, float)):
            raise TypeError("min_score must be numeric")

        if not 0.0 <= min_score <= 1.0:
            raise ValueError(
                "min_score must be between 0.0 and 1.0"
            )

        for result in results:
            if not isinstance(
                result,
                SemanticSearchResult,
            ):
                raise TypeError(
                    "all results must be SemanticSearchResult"
                )

        return [
            result
            for result in results
            if result.score >= min_score
        ]