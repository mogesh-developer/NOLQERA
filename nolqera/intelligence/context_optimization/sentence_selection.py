
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)


@dataclass(frozen=True)
class SentenceSelection:
    """
    Result of sentence selection.

    selected:
        Final selected RankedContext objects in original context order.

    text:
        Exact concatenation of selected sentence text.
    """

    selected: List[RankedContext]
    text: str


class SentenceSelector:
    """
    Selects high-value sentences from already ranked context.

    Existing ranking intelligence is reused.

    Selection rules:
        1. Highest ranking_score gets priority.
        2. A sentence is selected only once.
        3. max_sentences limits the final selection.
        4. Selected sentences are restored to original order.
    """

    def __init__(self, max_sentences: int = 3) -> None:
        if not isinstance(max_sentences, int):
            raise TypeError(
                "max_sentences must be an integer"
            )

        if max_sentences <= 0:
            raise ValueError(
                "max_sentences must be positive"
            )

        self.max_sentences = max_sentences

    def select(
        self,
        ranked_context: List[RankedContext],
    ) -> SentenceSelection:
        """
        Select the highest-value sentences.

        The existing ranking_score determines priority.
        """

        if not isinstance(ranked_context, list):
            raise TypeError(
                "ranked_context must be a list"
            )

        for item in ranked_context:
            if not isinstance(item, RankedContext):
                raise TypeError(
                    "ranked_context must contain "
                    "RankedContext objects"
                )

        if not ranked_context:
            return SentenceSelection(
                selected=[],
                text="",
            )

        # Existing ContextRanker output is the source
        # of sentence priority.
        candidates = sorted(
            ranked_context,
            key=lambda item: (
                -item.ranking_score,
                item.result.index,
            ),
        )

        selected: List[RankedContext] = []
        seen_indexes = set()

        for candidate in candidates:
            index = candidate.result.index

            if index in seen_indexes:
                continue

            selected.append(candidate)
            seen_indexes.add(index)

            if len(selected) >= self.max_sentences:
                break

        # Preserve original context order.
        selected.sort(
            key=lambda item: item.result.index
        )

        text = " ".join(
            item.result.text
            for item in selected
        )

        return SentenceSelection(
            selected=selected,
            text=text,
        )

