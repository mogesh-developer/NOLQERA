
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)


@dataclass(frozen=True)
class SemanticCompressionResult:
    """
    Result of semantic compression.

    selected:
        Sentences retained after semantic redundancy removal.

    removed:
        Sentences removed because their semantic information was
        already represented by a retained sentence.

    text:
        Exact text of the retained sentences in original order.
    """

    selected: List[RankedContext]
    removed: List[RankedContext]
    text: str


class SemanticCompressor:
    """
    Performs semantic compression over already selected/ranked context.

    The actual semantic redundancy decision is supplied through
    `redundancy_checker`.

    This is intentional: NOLQERA already has semantic redundancy
    intelligence, so this class orchestrates that existing logic
    instead of implementing another embedding/similarity engine.

    redundancy_checker(a, b) must return True when `a` and `b`
    represent redundant semantic information.
    """

    def __init__(
        self,
        redundancy_checker: Optional[
            Callable[[str, str], bool]
        ] = None,
    ) -> None:

        if redundancy_checker is not None and not callable(
            redundancy_checker
        ):
            raise TypeError(
                "redundancy_checker must be callable"
            )

        self.redundancy_checker = redundancy_checker

    def compress(
        self,
        sentences: Sequence[RankedContext],
    ) -> SemanticCompressionResult:

        if not isinstance(sentences, (list, tuple)):
            raise TypeError(
                "sentences must be a list or tuple"
            )

        for item in sentences:
            if not isinstance(item, RankedContext):
                raise TypeError(
                    "sentences must contain "
                    "RankedContext objects"
                )

        if not sentences:
            return SemanticCompressionResult(
                selected=[],
                removed=[],
                text="",
            )

        if self.redundancy_checker is None:
            raise ValueError(
                "redundancy_checker is required for "
                "semantic compression"
            )

        # Process higher-value sentences first.
        #
        # The existing ranking score decides which sentence
        # becomes the representative when two sentences carry
        # redundant information.
        candidates = sorted(
            sentences,
            key=lambda item: (
                -item.ranking_score,
                item.result.index,
            ),
        )

        selected: List[RankedContext] = []
        removed: List[RankedContext] = []

        for candidate in candidates:

            is_redundant = False

            for retained in selected:

                if self.redundancy_checker(
                    candidate.result.text,
                    retained.result.text,
                ):
                    is_redundant = True
                    break

            if is_redundant:
                removed.append(candidate)
            else:
                selected.append(candidate)

        # The compression decision is ranking-driven, but the
        # final context must remain readable and preserve source
        # ordering.
        selected.sort(
            key=lambda item: item.result.index
        )

        removed.sort(
            key=lambda item: item.result.index
        )

        text = " ".join(
            item.result.text
            for item in selected
        )

        return SemanticCompressionResult(
            selected=selected,
            removed=removed,
            text=text,
        )