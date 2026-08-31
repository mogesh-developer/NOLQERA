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
def index():

    documents = [
        "fastapi python backend api",
        "python machine learning",
        "mongodb database storage",
    ]

    # Fit TF-IDF on these document terms to ensure proper vocabulary coverage
    fit_documents = [
        ["fastapi", "python", "backend", "api"],
        ["python", "machine", "learning"],
        ["mongodb", "database", "storage"],
        ["django", "python", "framework"],
        ["docker", "container"],
    ]

    provider = TFIDFEmbeddingProvider()
    provider.fit(fit_documents)

    return SemanticSearchIndex(
        embedding_provider=provider,
        documents=documents,
    )


def test_service_accepts_index(index):

    service = SemanticSearchService(index)

    assert service.index is index


def test_service_search(index):

    service = SemanticSearchService(index)

    results = service.search(
        "python backend",
        top_k=1,
    )

    assert len(results) == 1


def test_service_supports_add_document(index):

    service = SemanticSearchService(index)

    count = service.document_count

    service.add_document(
        "django python framework"
    )

    assert service.document_count == count + 1


def test_service_supports_add_documents(index):

    service = SemanticSearchService(index)

    count = service.document_count

    service.add_documents(
        [
            "docker container",
            "mongodb database",
        ]
    )

    assert service.document_count == count + 2


def test_service_supports_update(index):

    service = SemanticSearchService(index)

    service.update_document(
        0,
        "django python framework",
    )

    assert service.documents[0] == (
        "django python framework"
    )


def test_service_supports_remove(index):

    service = SemanticSearchService(index)

    count = service.document_count

    service.remove_document(0)

    assert service.document_count == count - 1


def test_service_supports_clear(index):

    service = SemanticSearchService(index)

    service.clear()

    assert service.document_count == 0


def test_service_rejects_invalid_index():

    with pytest.raises(TypeError):
        SemanticSearchService(None)