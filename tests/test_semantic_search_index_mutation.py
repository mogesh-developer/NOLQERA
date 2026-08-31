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


def test_add_document(index):

    index.add_document(
        "django python web framework"
    )

    assert index.document_count == 4

    assert (
        "django python web framework"
        in index.documents
    )


def test_add_documents(index):

    index.add_documents(
        [
            "mongodb database",
            "docker container",
        ]
    )

    assert index.document_count == 5


def test_added_document_can_be_searched(index):

    index.add_document(
        "django python web framework"
    )

    results = index.search(
        "python framework",
        top_k=1,
    )

    assert results[0].text == (
        "django python web framework"
    )


def test_add_document_rejects_non_string(index):

    with pytest.raises(TypeError):

        index.add_document(123)


def test_add_document_rejects_empty_string(index):

    with pytest.raises(ValueError):

        index.add_document("")


def test_add_documents_rejects_non_list(index):

    with pytest.raises(TypeError):

        index.add_documents(
            "invalid"
        )


def test_add_documents_rejects_empty_list(index):

    with pytest.raises(ValueError):

        index.add_documents([])


def test_clear_removes_all_documents(index):

    index.clear()

    assert index.document_count == 0

    assert index.documents == []