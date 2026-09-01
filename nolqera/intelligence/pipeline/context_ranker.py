from nolqera.intelligence.context_optimization.context_ranking import (
    ContextRanker,
    RankedContext,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


class ContextRankingAnalyzer:
    """
    Phase 4 pipeline adapter for context ranking.

    Reuses the existing ContextRanker from the
    Context Optimization layer.
    """

    def __init__(self, context_ranker: ContextRanker):
        if not isinstance(context_ranker, ContextRanker):
            raise TypeError(
                "context_ranker must be a ContextRanker"
            )

        self._context_ranker = context_ranker

    def rank(
        self,
        results: list[SemanticSearchResult],
        importance_scores: list[float],
    ) -> list[RankedContext]:
        return self._context_ranker.rank(
            results,
            importance_scores,
        )