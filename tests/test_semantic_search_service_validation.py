import pytest

from nolqera.intelligence.semantic_search.service import (
    SemanticSearchService,
)
from nolqera.intelligence.semantic_search.index import (
    SemanticSearchIndex,
)
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)


@pytest.fixture
def service():

    documents = [
        "fastapi python backend api",
        "python machine learning",
        "mongodb database storage",
    ]

    provider = TFIDFEmbeddingProvider()
    provider.fit(
        [
            document.split()
            for document in documents
        ]
    )

    index = SemanticSearchIndex(
        embedding_provider=provider,
        documents=documents,
    )

    return SemanticSearchService(index)


def test_service_rejects_invalid_query(service):

    with pytest.raises(TypeError):
        service.search(123)


def test_service_rejects_empty_query(service):

    with pytest.raises(ValueError):
        service.search("")


def test_service_supports_top_k(service):

    results = service.search(
        "python",
        top_k=2,
    )

    assert len(results) <= 2


def test_service_supports_min_score(service):

    results = service.search(
        "python",
        min_score=0.5,
    )

    assert all(
        result.score >= 0.5
        for result in results
    )


def test_service_rejects_invalid_top_k(service):

    with pytest.raises(TypeError):
        service.search(
            "python",
            top_k="2",
        )


def test_service_rejects_zero_top_k(service):

    with pytest.raises(ValueError):
        service.search(
            "python",
            top_k=0,
        )


def test_service_rejects_negative_top_k(service):

    with pytest.raises(ValueError):
        service.search(
            "python",
            top_k=-1,
        )


def test_service_rejects_invalid_min_score(service):

    with pytest.raises(TypeError):
        service.search(
            "python",
            min_score="0.5",
        )


def test_service_rejects_out_of_range_min_score(service):

    with pytest.raises(ValueError):
        service.search(
            "python",
            min_score=1.5,
        )