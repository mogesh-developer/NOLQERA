
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)


@dataclass(frozen=True)
class RedundancyAwareCompressionResult:
    """
    Final result of redundancy-aware compression.

    selected:
        Sentences retained after all configured redundancy checks.

    removed:
        Sentences removed because their information was already
        represented by a retained sentence.

    text:
        Exact final context text.
    """

    selected: List[RankedContext]
    removed: List[RankedContext]
    text: str


class RedundancyAwareCompressor:
    """
    Composes existing NOLQERA redundancy intelligence.

    The compressor does not implement a new duplicate or semantic
    similarity algorithm.

    Instead, existing detectors are supplied as callables.

    Each detector must accept:

        candidate_text
        retained_text

    and return True when the candidate should be considered
    redundant with the retained sentence.

    Multiple detectors can therefore be composed:

        exact duplicate
        near duplicate
        semantic redundancy
        redundant information
    """

    def __init__(
        self,
        exact_duplicate_checker: Optional[
            Callable[[str, str], bool]
        ] = None,
        near_duplicate_checker: Optional[
            Callable[[str, str], bool]
        ] = None,
        semantic_redundancy_checker: Optional[
            Callable[[str, str], bool]
        ] = None,
        redundant_information_checker: Optional[
            Callable[[str, str], bool]
        ] = None,
    ) -> None:

        checkers = [
            exact_duplicate_checker,
            near_duplicate_checker,
            semantic_redundancy_checker,
            redundant_information_checker,
        ]

        for checker in checkers:
            if checker is not None and not callable(checker):
                raise TypeError(
                    "all redundancy checkers must be callable"
                )

        self.exact_duplicate_checker = (
            exact_duplicate_checker
        )

        self.near_duplicate_checker = (
            near_duplicate_checker
        )

        self.semantic_redundancy_checker = (
            semantic_redundancy_checker
        )

        self.redundant_information_checker = (
            redundant_information_checker
        )

    def _is_redundant(
        self,
        candidate: str,
        retained: str,
    ) -> bool:

        checkers = [
            self.exact_duplicate_checker,
            self.near_duplicate_checker,
            self.semantic_redundancy_checker,
            self.redundant_information_checker,
        ]

        for checker in checkers:
            if checker is not None and checker(candidate, retained):
                return True

        return False

    def compress(
        self,
        sentences: Sequence[RankedContext],
    ) -> RedundancyAwareCompressionResult:

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

        checkers = [
            self.exact_duplicate_checker,
            self.near_duplicate_checker,
            self.semantic_redundancy_checker,
            self.redundant_information_checker,
        ]

        if not any(checkers):
            raise ValueError(
                "at least one redundancy checker "
                "is required"
            )

        if not sentences:
            return RedundancyAwareCompressionResult(
                selected=[],
                removed=[],
                text="",
            )

        selected: List[RankedContext] = []
        removed: List[RankedContext] = []

        # Highest-ranked sentence becomes the preferred
        # representative when redundancy is detected.
        candidates = sorted(
            sentences,
            key=lambda item: (
                -item.ranking_score,
                item.result.index,
            ),
        )

        for candidate in candidates:

            redundant = False

            for retained in selected:

                if self._is_redundant(
                    candidate.result.text,
                    retained.result.text,
                ):
                    redundant = True
                    break

            if redundant:
                removed.append(candidate)
            else:
                selected.append(candidate)

        # Preserve original context order in final output.
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

        return RedundancyAwareCompressionResult(
            selected=selected,
            removed=removed,
            text=text,
        )
