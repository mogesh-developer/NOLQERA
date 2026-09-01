from nolqera.intelligence.entities.engine import EntityEngine


class EntityAnalyzer:
    """
    Phase 4 pipeline adapter for entity analysis.

    This class does NOT implement entity extraction itself.
    It delegates entity analysis to the existing EntityEngine.
    """

    def __init__(self, entity_engine: EntityEngine):
        if not isinstance(entity_engine, EntityEngine):
            raise TypeError("entity_engine must be an EntityEngine")

        self._entity_engine = entity_engine

    def analyze(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        return self._entity_engine.analyze(text)