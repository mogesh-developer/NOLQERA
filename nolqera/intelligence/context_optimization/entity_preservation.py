
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Set

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)


@dataclass(frozen=True)
class EntityPreservationResult:
    """
    Result of entity-preservation validation.
    """

    required_entities: List[str]
    preserved_entities: List[str]
    missing_entities: List[str]
    is_preserved: bool


class EntityPreserver:
    """
    Validates that important entities identified in the original
    context remain present after compression.

    Entity extraction is intentionally delegated to the existing
    NOLQERA EntityAnalyzer through `entity_extractor`.

    The extractor must accept sentence text and return an iterable
    of entity strings.
    """

    def __init__(
        self,
        entity_extractor: Callable[
            [str],
            Iterable[str],
        ],
    ) -> None:

        if not callable(entity_extractor):
            raise TypeError(
                "entity_extractor must be callable"
            )

        self.entity_extractor = entity_extractor

    def _extract_entities(
        self,
        sentences: Sequence[RankedContext],
    ) -> Set[str]:

        entities: Set[str] = set()

        for sentence in sentences:
            extracted = self.entity_extractor(
                sentence.result.text
            )

            if extracted is None:
                continue

            for entity in extracted:
                if not isinstance(entity, str):
                    raise TypeError(
                        "entity extractor must return "
                        "strings"
                    )

                normalized = entity.strip()

                if normalized:
                    entities.add(normalized)

        return entities

    def identify_required_entities(
        self,
        original: Sequence[RankedContext],
    ) -> List[str]:
        """
        Extract all unique entities from original context.

        Results are returned deterministically.
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

        entities = self._extract_entities(
            original
        )

        return sorted(
            entities,
            key=lambda value: value.casefold(),
        )

    def validate(
        self,
        original: Sequence[RankedContext],
        compressed: Sequence[RankedContext],
    ) -> EntityPreservationResult:
        """
        Validate that every entity found in the original context
        is still present in the compressed context.

        Entity comparison is case-insensitive while preserving
        the original entity spelling in the result.
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

        required = self.identify_required_entities(
            original
        )

        compressed_entities = self._extract_entities(
            compressed
        )

        compressed_lookup = {
            entity.casefold()
            for entity in compressed_entities
        }

        preserved = [
            entity
            for entity in required
            if entity.casefold() in compressed_lookup
        ]

        missing = [
            entity
            for entity in required
            if entity.casefold() not in compressed_lookup
        ]

        return EntityPreservationResult(
            required_entities=required,
            preserved_entities=preserved,
            missing_entities=missing,
            is_preserved=len(missing) == 0,
        )
