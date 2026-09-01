from nolqera.intelligence.keyphrase.engine import KeyphraseEngine


class KeywordAnalyzer:
    """
    Phase 4 adapter for the existing keyphrase intelligence layer.

    Reuses the existing KeyphraseEngine instead of implementing
    keyword/keyphrase scoring again.
    """

    def __init__(self, engine: KeyphraseEngine):
        if not isinstance(engine, KeyphraseEngine):
            raise TypeError(
                "engine must be a KeyphraseEngine"
            )

        self.engine = engine

    def analyze(
        self,
        text: str,
        top_k: int = 5,
    ) -> dict:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        result = self.engine.analyze(
            text,
            top_k=top_k,
        )

        return result