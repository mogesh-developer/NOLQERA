
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)


@dataclass(frozen=True)
class PrioritizedContext:
    """
    A ranked context item together with its deterministic
    prioritization position.
    """

    context: RankedContext
    priority: int


class ContextPrioritizer:
    """
    Converts the existing ContextRanker output into the
    deterministic priority order used by compression.

    No new relevance, importance, or ranking algorithm is
    introduced here.

    The existing `ranking_score` remains the source of truth.
    """

    def __init__(
        self,
        descending: bool = True,
    ) -> None:

        if not isinstance(descending, bool):
            raise TypeError(
                "descending must be boolean"
            )

        self.descending = descending

    def prioritize(
        self,
        contexts: Sequence[RankedContext],
    ) -> List[PrioritizedContext]:
        """
        Return contexts ordered by the existing ranking score.

        Tie-breaking is deterministic:

        1. ranking_score
        2. importance_score
        3. relevance_score
        4. original result index
        """

        if not isinstance(contexts, (list, tuple)):
            raise TypeError(
                "contexts must be a list or tuple"
            )

        for context in contexts:
            if not isinstance(context, RankedContext):
                raise TypeError(
                    "contexts must contain RankedContext objects"
                )

        ordered = sorted(
            contexts,
            key=lambda item: (
                item.ranking_score,
                item.importance_score,
                item.relevance_score,
                -item.result.index,
            ),
            reverse=self.descending,
        )

        return [
            PrioritizedContext(
                context=context,
                priority=index,
            )
            for index, context in enumerate(ordered)
        ]

    def select_top(
        self,
        contexts: Sequence[RankedContext],
        limit: int,
    ) -> List[RankedContext]:
        """
        Select the highest-priority contexts.

        This method does not perform compression.
        It only consumes the existing ranking output.
        """

        if not isinstance(limit, int):
            raise TypeError(
                "limit must be an integer"
            )

        if limit < 0:
            raise ValueError(
                "limit must be non-negative"
            )

        prioritized = self.prioritize(contexts)

        return [
            item.context
            for item in prioritized[:limit]
        ]
