from nolqera.intelligence.importance.engine import ImportanceEngine


class ImportanceAnalyzer:
    """
    Phase 4 adapter for the existing importance intelligence layer.

    Reuses the full importance engine instead of calculating
    sentence importance independently.
    """

    def __init__(self, engine: ImportanceEngine):
        if not isinstance(engine, ImportanceEngine):
            raise TypeError(
                "engine must be an ImportanceEngine"
            )

        self.engine = engine

    def analyze(
        self,
        sentences: list[str],
    ) -> list[dict]:
        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list")

        if any(not isinstance(sentence, str) for sentence in sentences):
            raise TypeError(
                "sentences must contain only strings"
            )

        if any(not sentence.strip() for sentence in sentences):
            raise ValueError(
                "sentences cannot contain empty strings"
            )

        results = self.engine.analyze(sentences)

        scores = {res.sentence: res.score for res in results}

        return [
            {
                "index": index,
                "text": sentence,
                "score": scores.get(sentence, 0.0),
            }
            for index, sentence in enumerate(sentences)
        ]