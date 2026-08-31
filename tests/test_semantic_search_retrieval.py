import pytest

from nolqera.intelligence.semantic_search.engine import (
    SemanticSearchEngine,
)
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)


@pytest.fixture
def engine():

    documents = [
        ["fastapi", "python", "backend", "api"],
        ["python", "machine", "learning"],
        ["react", "frontend", "javascript"],
        ["mongodb", "database", "storage"],
    ]

    provider = TFIDFEmbeddingProvider()
    provider.fit(documents)

    return SemanticSearchEngine(provider)


def test_search_supports_top_k(engine):

    documents = [
        "fastapi python backend api",
        "python machine learning",
        "react frontend javascript",
        "mongodb database storage",
    ]

    results = engine.search(
        "python backend",
        documents,
        top_k=2,
    )

    assert len(results) == 2


def test_search_top_k_returns_highest_scores_first(
    engine,
):

    documents = [
        "mongodb database storage",
        "fastapi python backend api",
        "python machine learning",
        "react frontend javascript",
    ]

    results = engine.search(
        "python backend",
        documents,
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].score >= results[1].score


def test_search_supports_min_score(engine):

    documents = [
        "fastapi python backend api",
        "python machine learning",
        "react frontend javascript",
        "mongodb database storage",
    ]

    results = engine.search(
        "fastapi backend",
        documents,
        min_score=0.5,
    )

    assert all(
        result.score >= 0.5
        for result in results
    )


def test_search_rejects_invalid_top_k(engine):

    with pytest.raises(TypeError):
        engine.search(
            "python",
            ["python backend"],
            top_k="2",
        )


def test_search_rejects_zero_top_k(engine):

    with pytest.raises(ValueError):
        engine.search(
            "python",
            ["python backend"],
            top_k=0,
        )


def test_search_rejects_negative_top_k(engine):

    with pytest.raises(ValueError):
        engine.search(
            "python",
            ["python backend"],
            top_k=-1,
        )


def test_search_rejects_invalid_min_score(engine):

    with pytest.raises(
        ValueError,
    ):
        engine.search(
            "python",
            ["python backend"],
            min_score=1.5,
        )