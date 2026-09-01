from nolqera.intelligence.intent.engine import IntentEngine


class IntentAnalyzer:
    """
    Phase 4 pipeline adapter for intent analysis.

    Delegates intent analysis to the existing IntentEngine.
    No intent detection logic is duplicated here.
    """

    def __init__(self, intent_engine: IntentEngine):
        if not isinstance(intent_engine, IntentEngine):
            raise TypeError(
                "intent_engine must be an IntentEngine"
            )

        self._intent_engine = intent_engine

    def analyze(self, text: str):
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "text cannot be empty"
            )

        return self._intent_engine.analyze(text)