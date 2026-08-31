import pytest

from nolqera.intelligence.semantic_search.index import (
    SemanticSearchIndex,
)
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)


@pytest.fixture
def index():

    documents = [
        "fastapi python backend api",
        "python machine learning",
        "react frontend javascript",
        "mongodb database storage",
    ]

    provider = TFIDFEmbeddingProvider()
    provider.fit(
        [
            document.split()
            for document in documents
        ]
    )

    return SemanticSearchIndex(
        embedding_provider=provider,
        documents=documents,
    )


def test_index_accepts_documents(index):

    assert index.document_count == 4


def test_index_search_returns_results(index):

    results = index.search(
        "python backend",
    )

    assert results
    assert results[0].text == (
        "fastapi python backend api"
    )


def test_index_search_supports_top_k(index):

    results = index.search(
        "python",
        top_k=2,
    )

    assert len(results) == 2


def test_index_search_supports_min_score(index):

    results = index.search(
        "fastapi backend",
        min_score=0.5,
    )

    assert results
    assert all(
        result.score >= 0.5
        for result in results
    )


def test_index_rejects_empty_documents():

    provider = TFIDFEmbeddingProvider()

    with pytest.raises(ValueError):

        SemanticSearchIndex(
            embedding_provider=provider,
            documents=[],
        )


def test_index_rejects_non_list_documents():

    provider = TFIDFEmbeddingProvider()

    with pytest.raises(TypeError):

        SemanticSearchIndex(
            embedding_provider=provider,
            documents="invalid",
        )


def test_index_rejects_empty_document():

    provider = TFIDFEmbeddingProvider()

    with pytest.raises(ValueError):

        SemanticSearchIndex(
            embedding_provider=provider,
            documents=[
                "fastapi backend",
                "",
            ],
        )


def test_index_preserves_document_order(index):

    assert index.documents == [
        "fastapi python backend api",
        "python machine learning",
        "react frontend javascript",
        "mongodb database storage",
    ]