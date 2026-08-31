import pytest

from nolqera.intelligence.semantic_search.engine import (
    SemanticSearchEngine,
)
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)


@pytest.fixture
def provider():

    documents = [
        ["fastapi", "backend", "api"],
        ["fastapi", "rest", "api"],
        ["mongodb", "database", "storage"],
    ]

    provider = TFIDFEmbeddingProvider()
    provider.fit(documents)

    return provider


@pytest.fixture
def engine(provider):

    return SemanticSearchEngine(provider)


def test_engine_accepts_embedding_provider(
    provider,
):

    engine = SemanticSearchEngine(provider)

    assert engine is not None


def test_search_returns_results(
    engine,
):

    results = engine.search(
        "fastapi backend",
        [
            "fastapi backend api",
            "mongodb database",
        ],
    )

    assert isinstance(results, list)
    assert len(results) == 2


def test_search_orders_highest_similarity_first(
    engine,
):

    results = engine.search(
        "fastapi backend",
        [
            "mongodb database storage",
            "fastapi backend api",
            "fastapi rest api",
        ],
    )

    assert results[0].score >= results[1].score
    assert results[1].score >= results[2].score


def test_search_preserves_document_index(
    engine,
):

    documents = [
        "mongodb database storage",
        "fastapi backend api",
        "python machine learning",
    ]

    results = engine.search(
        "fastapi backend",
        documents,
    )

    assert results[0].index == 1


def test_search_preserves_document_text(
    engine,
):

    documents = [
        "fastapi backend api",
        "mongodb database",
    ]

    results = engine.search(
        "fastapi backend",
        documents,
    )

    returned_texts = [
        result.text
        for result in results
    ]

    assert set(returned_texts) == set(documents)


def test_query_must_be_string(
    engine,
):

    with pytest.raises(TypeError):
        engine.search(
            123,
            ["fastapi backend"],
        )


def test_query_cannot_be_empty(
    engine,
):

    with pytest.raises(ValueError):
        engine.search(
            "",
            ["fastapi backend"],
        )


def test_documents_must_be_list(
    engine,
):

    with pytest.raises(TypeError):
        engine.search(
            "fastapi",
            "fastapi backend",
        )


def test_documents_cannot_be_empty(
    engine,
):

    with pytest.raises(ValueError):
        engine.search(
            "fastapi",
            [],
        )


def test_documents_must_contain_strings(
    engine,
):

    with pytest.raises(TypeError):
        engine.search(
            "fastapi",
            [
                "fastapi backend",
                123,
            ],
        )


def test_documents_cannot_contain_empty_strings(
    engine,
):

    with pytest.raises(ValueError):
        engine.search(
            "fastapi",
            [
                "fastapi backend",
                "",
            ],
        )


def test_search_scores_are_valid(
    engine,
):

    results = engine.search(
        "fastapi backend",
        [
            "fastapi backend api",
            "mongodb database",
        ],
    )

    for result in results:
        assert 0.0 <= result.score <= 1.0