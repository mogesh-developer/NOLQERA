import pytest

from nolqera.intelligence.relevance.engine import RelevanceEngine


def test_engine_rejects_empty_query():
    engine = RelevanceEngine()

    with pytest.raises(ValueError):
        engine.analyze(
            query="",
            sentences=["Some sentence."],
        )


def test_engine_rejects_empty_sentences():
    engine = RelevanceEngine()

    with pytest.raises(ValueError):
        engine.analyze(
            query="What is Python?",
            sentences=[],
        )


def test_engine_rejects_empty_sentence():
    engine = RelevanceEngine()

    with pytest.raises(ValueError):
        engine.analyze(
            query="What is Python?",
            sentences=[""],
        )


def test_engine_handles_single_sentence():
    engine = RelevanceEngine(
        relevant_threshold=0.05,
        weak_threshold=0.01,
    )

    results = engine.analyze(
        query="What database is used?",
        sentences=[
            "The application uses MongoDB."
        ],
    )

    assert len(results) == 1
    assert results[0].rank == 1


def test_engine_handles_unrelated_content():
    engine = RelevanceEngine()

    results = engine.analyze(
        query="What database is used?",
        sentences=[
            "I travelled to Chennai yesterday."
        ],
    )

    assert len(results) == 1
    assert results[0].score == 0.0
    assert results[0].label == "irrelevant"