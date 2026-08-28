import pytest

from nolqera.intelligence.importance.engine import (
    ImportanceEngine,
)


def test_importance_engine_ranks_real_document():
    engine = ImportanceEngine()

    sentences = [
        "The application uses FastAPI.",
        "FastAPI provides REST API endpoints.",
        "The application uses MongoDB for data storage.",
        "I travelled to Chennai yesterday.",
    ]

    results = engine.analyze(sentences)

    print("\n--- Importance Scores ---")

    for result in results:
        print(
            f"{result.score:.4f} | "
            f"Rank {result.rank} | "
            f"{result.sentence}"
        )

    print("\n--- Most Important Sentence ---")
    print(results[0].sentence)
    print(f"Score: {results[0].score:.4f}")

    assert len(results) == 4

    assert results[0].rank == 1

    assert all(
        0.0 <= result.score <= 1.0
        for result in results
    )

def test_repeated_generic_content_gets_lower_importance():
    engine = ImportanceEngine()

    sentences = [
        "The system architecture uses Python FastAPI MongoDB.",
        "This is a very useful system.",
        "The system is good and useful.",
        "Python FastAPI MongoDB provide backend services.",
    ]

    results = engine.analyze(sentences)

    assert results[0].score > results[-1].score

def test_informative_later_sentence_can_beat_generic_early_sentence():
    engine = ImportanceEngine()

    sentences = [
        "This application is very useful.",
        "The system uses FastAPI with MongoDB for persistent data storage.",
        "This is a good application.",
    ]

    results = engine.analyze(sentences)

    assert results[0].sentence == (
        "The system uses FastAPI with MongoDB "
        "for persistent data storage."
    )


def test_engine_rejects_empty_sentences():
    engine = ImportanceEngine()

    with pytest.raises(ValueError):
        engine.analyze([])


def test_engine_rejects_empty_sentence():
    engine = ImportanceEngine()

    with pytest.raises(ValueError):
        engine.analyze([
            "FastAPI is useful.",
            "",
            "MongoDB stores data.",
        ])


def test_engine_rejects_non_string_sentence():
    engine = ImportanceEngine()

    with pytest.raises(TypeError):
        engine.analyze([
            "FastAPI is useful.",
            123,
        ])


def test_engine_handles_single_sentence():
    engine = ImportanceEngine()

    results = engine.analyze([
        "FastAPI is a web framework."
    ])

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].score == 1.0