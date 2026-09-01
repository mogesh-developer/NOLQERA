from __future__ import annotations

from typing import List

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)


class ContextCompressor:
    """
    Compresses ranked context by selecting the highest-ranked
    sentences while preserving their ranking order.

    The compressor performs deterministic extractive compression.
    It does not rewrite or paraphrase the original text.
    """

    def compress(
        self,
        ranked_context: List[RankedContext],
        max_sentences: int,
    ) -> str:
        if not isinstance(ranked_context, list):
            raise TypeError(
                "ranked_context must be a list"
            )

        if not isinstance(max_sentences, int):
            raise TypeError(
                "max_sentences must be an integer"
            )

        if max_sentences <= 0:
            raise ValueError(
                "max_sentences must be greater than zero"
            )

        for item in ranked_context:
            if not isinstance(item, RankedContext):
                raise TypeError(
                    "ranked_context must contain "
                    "RankedContext instances"
                )

        if not ranked_context:
            return ""

        selected = ranked_context[:max_sentences]

        return " ".join(
            item.result.text
            for item in selected
        )