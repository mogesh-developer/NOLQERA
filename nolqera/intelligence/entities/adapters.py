from .models import EntityResult


class EntityAdapter:
    @staticmethod
    def to_result(entity: dict) -> EntityResult:
        if not isinstance(entity, dict):
            raise TypeError("entity must be a dictionary")

        required_fields = {
            "text",
            "entity_type",
            "score",
            "start",
            "end",
        }
        missing = required_fields - entity.keys()

        if missing:
            raise ValueError(
                f"entity is missing required fields: {sorted(missing)}"
            )

        return EntityResult(
            text=entity["text"],
            entity_type=entity["entity_type"],
            score=float(entity["score"]),
            start=int(entity["start"]),
            end=int(entity["end"]),
        )

    @classmethod
    def to_results(cls, entities: list[dict]) -> list[EntityResult]:
        if not isinstance(entities, list):
            raise TypeError("entities must be a list")

        return [cls.to_result(entity) for entity in entities]