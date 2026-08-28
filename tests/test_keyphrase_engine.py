from nolqera.intelligence.keyphrase.engine import (
    KeyphraseEngine,
)


def test_keyphrase_engine_extracts_real_concepts():
    engine = KeyphraseEngine()

    text = (
        "The application uses FastAPI for REST APIs. "
        "MongoDB is used for persistent data storage. "
        "FastAPI and MongoDB form the backend architecture."
    )

    results = engine.extract(
        text,
        top_k=5,
    )

    print("\n--- Keyphrase Results ---")

    for result in results:
        print(
            f"{result.score:.4f} | "
            f"Rank {result.rank} | "
            f"{result.phrase}"
        )

    assert results

    assert len(results) <= 5

    assert all(
        0.0 <= result.score <= 1.0
        for result in results
    )

    assert all(
        result.rank == index
        for index, result in enumerate(
            results,
            start=1,
        )
    )