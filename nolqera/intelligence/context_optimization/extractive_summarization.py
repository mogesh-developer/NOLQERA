from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)


@dataclass(frozen=True)
class ExtractiveSummary:
    """
    Represents the result of extractive summarization.

    selected:
        Selected RankedContext objects in original context order.

    text:
        Final extracted summary text.
    """

    selected: List[RankedContext]
    text: str


class ExtractiveSummarizer:
    """
    Selects the highest-ranked existing context sentences.

    No new text is generated.

    Selection is performed using the existing ContextRanker output.
    After selecting the top-ranked sentences, the selected sentences
    are restored to their original context order for coherence.
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

    def summarize(
        self,
        ranked_context: List[RankedContext],
    ) -> ExtractiveSummary:
        """
        Extract the highest-ranked sentences.

        Ranking score determines selection.

        Original result.index determines final ordering.
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
            return ExtractiveSummary(
                selected=[],
                text="",
            )

        # Step 1:
        # Select the highest-ranked context items.
        selected = sorted(
            ranked_context,
            key=lambda item: (
                -item.ranking_score,
                item.result.index,
            ),
        )[: self.max_sentences]

        # Step 2:
        # Restore original document/context order.
        selected = sorted(
            selected,
            key=lambda item: item.result.index,
        )

        summary_text = " ".join(
            item.result.text
            for item in selected
        )

        return ExtractiveSummary(
            selected=selected,
            text=summary_text,
        )