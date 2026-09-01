import pytest

from nolqera.intelligence.intent.engine import IntentEngine
from nolqera.intelligence.pipeline.intent_analyzer import IntentAnalyzer


def test_intent_analyzer_accepts_intent_engine():
    engine = IntentEngine()

    analyzer = IntentAnalyzer(engine)

    assert analyzer is not None


def test_intent_analyzer_rejects_invalid_engine():
    with pytest.raises(
        TypeError,
        match="intent_engine must be an IntentEngine",
    ):
        IntentAnalyzer(object())


def test_intent_analyzer_rejects_non_string_text():
    analyzer = IntentAnalyzer(IntentEngine())

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        analyzer.analyze(123)


def test_intent_analyzer_rejects_empty_text():
    analyzer = IntentAnalyzer(IntentEngine())

    with pytest.raises(
        ValueError,
        match="text cannot be empty",
    ):
        analyzer.analyze("")


def test_intent_analyzer_rejects_whitespace_text():
    analyzer = IntentAnalyzer(IntentEngine())

    with pytest.raises(
        ValueError,
        match="text cannot be empty",
    ):
        analyzer.analyze("   ")


def test_intent_analyzer_delegates_to_intent_engine():
    class FakeIntentEngine:

        def analyze(self, text):
            return [
                {
                    "intent": "question",
                }
            ]

    analyzer = IntentAnalyzer.__new__(IntentAnalyzer)
    analyzer._intent_engine = FakeIntentEngine()

    result = analyzer.analyze(
        "What is Python?"
    )

    assert result == [
        {
            "intent": "question",
        }
    ]


def test_intent_analyzer_returns_exact_intent_result():
    analyzer = IntentAnalyzer(IntentEngine())

    result = analyzer.analyze(
        "What is Python?"
    )

    assert len(result) == 1

    assert result[0].intent == "question"
    assert result[0].score == pytest.approx(1.0)
    assert result[0].evidence_count == 2