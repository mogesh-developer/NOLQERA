import pytest

from nolqera.intelligence.semantic_similarity.engine import (
    SemanticSimilarityEngine,
)
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)


@pytest.fixture
def engine():

    provider = TFIDFEmbeddingProvider()

    provider.fit(
        [
            ["fastapi", "backend", "api"],
            ["mongodb", "database", "storage"],
            ["fastapi", "rest", "api"],
        ]
    )

    return SemanticSimilarityEngine(
        provider
    )


def test_engine_uses_embedding_provider(engine):

    result = engine.compare(
        ["fastapi", "backend"],
        ["fastapi", "api"],
    )

    print("\n--- Semantic Similarity ---")
    print(
        f"{result.score:.4f} | "
        f"{result.text_a} <-> {result.text_b}"
    )

    assert 0.0 <= result.score <= 1.0


def test_engine_returns_high_similarity_for_related_tokens(
    engine,
):

    result = engine.compare(
        ["fastapi", "backend"],
        ["fastapi", "api"],
    )

    assert result.score > 0.0


def test_engine_rejects_empty_tokens(engine):

    with pytest.raises(ValueError):

        engine.compare(
            [],
            ["fastapi"],
        )


def test_engine_rejects_invalid_tokens(engine):

    with pytest.raises(TypeError):

        engine.compare(
            "fastapi",
            ["backend"],
        )