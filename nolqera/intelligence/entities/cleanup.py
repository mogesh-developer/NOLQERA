from .detector import DetectedEntity


class EntitySpanCleaner:
    """
    Removes overlapping entity spans.

    Priority:
    1. Higher confidence score
    2. Longer span when scores are equal
    3. Earlier span when everything else is equal
    """

    def clean(
        self,
        entities: list[DetectedEntity],
    ) -> list[DetectedEntity]:

        if not isinstance(entities, list):
            raise TypeError("entities must be a list")

        if not entities:
            return []

        for entity in entities:
            if not isinstance(entity, DetectedEntity):
                raise TypeError(
                    "all items must be DetectedEntity instances"
                )

        # Generic priority calculation.
        ordered = sorted(
            entities,
            key=self._priority,
            reverse=True,
        )

        selected: list[DetectedEntity] = []

        for entity in ordered:

            # If this entity overlaps an already
            # selected higher-quality entity,
            # discard it.
            if any(
                self._overlaps(entity, existing)
                for existing in selected
            ):
                continue

            selected.append(entity)

        # Restore document order.
        return sorted(
            selected,
            key=lambda entity: (
                entity.start,
                entity.end,
            ),
        )

    @staticmethod
    def _priority(
        entity: DetectedEntity,
    ) -> tuple[float, int, int]:

        span_length = entity.end - entity.start

        return (
            entity.score,
            span_length,
            -entity.start,
        )

    @staticmethod
    def _overlaps(
        first: DetectedEntity,
        second: DetectedEntity,
    ) -> bool:

        return (
            first.start < second.end
            and second.start < first.end
        )