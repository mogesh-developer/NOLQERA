import pytest

from nolqera.intelligence.intent.engine import (
    IntentEngine,
)
from nolqera.intelligence.intent.models import (
    IntentResult,
)


def test_intent_engine_detects_question():

    engine = IntentEngine()

    results = engine.analyze(
        "How does FastAPI work?"
    )

    print("\n--- NOLQERA Intent Engine ---")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result.intent:<15} "
            f"{result.score:.4f} "
            f"| evidence={result.evidence_count}"
        )

    assert results

    assert all(
        isinstance(result, IntentResult)
        for result in results
    )

    assert results[0].intent == "question"

    assert 0.0 <= results[0].score <= 1.0


def test_intent_engine_handles_statement():

    engine = IntentEngine()

    results = engine.analyze(
        "FastAPI is a Python framework."
    )

    assert isinstance(results, list)


def test_intent_engine_rejects_empty_text():

    engine = IntentEngine()

    with pytest.raises(ValueError):
        engine.analyze("")


def test_intent_engine_rejects_non_string():

    engine = IntentEngine()

    with pytest.raises(TypeError):
        engine.analyze(123)