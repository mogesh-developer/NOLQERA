
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Set

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)


@dataclass(frozen=True)
class FactPreservationResult:
    """
    Result of number/fact preservation validation.
    """

    required_facts: List[str]
    preserved_facts: List[str]
    missing_facts: List[str]
    is_preserved: bool


class FactPreserver:
    """
    Validates preservation of factual numeric information.

    The component extracts deterministic factual tokens such as:

        3.11
        95%
        48.46
        2026
        10

    It does not rewrite or generate facts.

    Existing context-ranking/compression components remain
    responsible for deciding which sentences are selected.
    """

    # Matches:
    #   3.11
    #   48.46
    #   2026
    #   10
    #   95%
    #
    # Avoids matching ordinary standalone punctuation.
    NUMBER_PATTERN = re.compile(
        r"(?<![\w.])"
        r"\d+(?:\.\d+)?"
        r"%?"
        r"(?!\w|(?:\.\d))"
    )

    def __init__(
        self,
        preserve_percentages: bool = True,
    ) -> None:

        if not isinstance(
            preserve_percentages,
            bool,
        ):
            raise TypeError(
                "preserve_percentages must be boolean"
            )

        self.preserve_percentages = (
            preserve_percentages
        )

    def _extract_numbers(
        self,
        text: str,
    ) -> Set[str]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        values = set(
            self.NUMBER_PATTERN.findall(text)
        )

        if not self.preserve_percentages:
            values = {
                value.rstrip("%")
                for value in values
            }

        return values

    def identify_required_facts(
        self,
        original: Sequence[RankedContext],
    ) -> List[str]:
        """
        Extract all unique numeric facts from the original
        context and return them deterministically.
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

        facts: Set[str] = set()

        for sentence in original:
            facts.update(
                self._extract_numbers(
                    sentence.result.text
                )
            )

        return sorted(
            facts,
            key=lambda value: (
                float(value.rstrip("%"))
                if value.rstrip("%").replace(
                    ".", "", 1
                ).isdigit()
                else value
            ),
        )

    def validate(
        self,
        original: Sequence[RankedContext],
        compressed: Sequence[RankedContext],
    ) -> FactPreservationResult:
        """
        Validate that all numeric facts from the original context
        remain present in the compressed context.
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

        required = self.identify_required_facts(
            original
        )

        compressed_facts: Set[str] = set()

        for sentence in compressed:
            compressed_facts.update(
                self._extract_numbers(
                    sentence.result.text
                )
            )

        preserved = [
            fact
            for fact in required
            if fact in compressed_facts
        ]

        missing = [
            fact
            for fact in required
            if fact not in compressed_facts
        ]

        return FactPreservationResult(
            required_facts=required,
            preserved_facts=preserved,
            missing_facts=missing,
            is_preserved=len(missing) == 0,
        )

