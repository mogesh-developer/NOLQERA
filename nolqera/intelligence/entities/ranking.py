from .detector import DetectedEntity


class EntityRanker:
    """Rank detected entities by confidence and span quality."""

    def rank(
        self,
        entities: list[DetectedEntity],
    ) -> list[DetectedEntity]:
        """Return entities ordered by descending importance."""

        if not isinstance(entities, list):
            raise TypeError("entities must be a list")

        if not entities:
            raise ValueError("entities cannot be empty")

        for entity in entities:
            if not isinstance(
                entity,
                DetectedEntity,
            ):
                raise TypeError(
                    "all items must be DetectedEntity instances"
                )

            if not 0.0 <= entity.score <= 1.0:
                raise ValueError(
                    "entity score must be between 0 and 1"
                )

        return sorted(
            entities,
            key=lambda entity: (
                entity.score,
                entity.candidate.length,
            ),
            reverse=True,
        )

    def top_k(
        self,
        entities: list[DetectedEntity],
        k: int,
    ) -> list[DetectedEntity]:
        """Return the top k detected entities."""

        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError(
                "k must be greater than zero"
            )

        ranked = self.rank(entities)

        return ranked[:k]