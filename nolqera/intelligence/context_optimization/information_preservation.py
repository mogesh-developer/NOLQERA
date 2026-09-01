
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)


@dataclass(frozen=True)
class InformationPreservationResult:
    """
    Result of information-preservation validation.
    """

    preserved: List[RankedContext]
    missing: List[RankedContext]
    is_preserved: bool


class InformationPreserver:
    """
    Validates whether important information from the original
    context remains present after compression.

    This class does not generate information and does not perform
    semantic compression itself.

    It validates preservation using the existing ranked context
    representation.
    """

    def __init__(
        self,
        importance_threshold: float = 0.70,
    ) -> None:

        if not isinstance(
            importance_threshold,
            (int, float),
        ):
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

    def identify_important(
        self,
        original: Sequence[RankedContext],
    ) -> List[RankedContext]:
        """
        Return sentences whose existing importance score is
        greater than or equal to the configured threshold.
        """

        if not isinstance(original, (list, tuple)):
            raise TypeError(
                "original must be a list or tuple"
            )

        for item in original:
            if not isinstance(item, RankedContext):
                raise TypeError(
                    "original must contain RankedContext objects"
                )

        important = [
            item
            for item in original
            if item.importance_score
            >= self.importance_threshold
        ]

        important.sort(
            key=lambda item: item.result.index
        )

        return important

    def validate(
        self,
        original: Sequence[RankedContext],
        compressed: Sequence[RankedContext],
    ) -> InformationPreservationResult:
        """
        Validate that every important original sentence is still
        represented in the compressed context.

        Matching is based on the original sentence index.

        The compression pipeline already decides whether content
        is redundant. This validator checks that important source
        information has not disappeared from the final result.
        """

        if not isinstance(original, (list, tuple)):
            raise TypeError(
                "original must be a list or tuple"
            )

        if not isinstance(compressed, (list, tuple)):
            raise TypeError(
                "compressed must be a list or tuple"
            )

        for item in original:
            if not isinstance(item, RankedContext):
                raise TypeError(
                    "original must contain RankedContext objects"
                )

        for item in compressed:
            if not isinstance(item, RankedContext):
                raise TypeError(
                    "compressed must contain RankedContext objects"
                )

        important = self.identify_important(
            original
        )

        compressed_indexes: Set[int] = {
            item.result.index
            for item in compressed
        }

        preserved = [
            item
            for item in important
            if item.result.index in compressed_indexes
        ]

        missing = [
            item
            for item in important
            if item.result.index not in compressed_indexes
        ]

        return InformationPreservationResult(
            preserved=preserved,
            missing=missing,
            is_preserved=len(missing) == 0,
        )
