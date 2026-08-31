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


def test_remove_document(index):

    index.remove_document(1)

    assert index.documents == [
        "fastapi python backend api",
        "mongodb database storage",
    ]

    assert index.document_count == 2


def test_remove_document_updates_search(index):

    index.remove_document(1)

    results = index.search(
        "python backend",
    )

    assert all(
        result.text != "python machine learning"
        for result in results
    )


def test_remove_document_preserves_remaining_order(index):

    index.remove_document(1)

    assert index.documents == [
        "fastapi python backend api",
        "mongodb database storage",
    ]


def test_remove_document_rejects_non_integer(index):

    with pytest.raises(TypeError):
        index.remove_document("1")


def test_remove_document_rejects_negative_index(index):

    with pytest.raises(ValueError):
        index.remove_document(-1)


def test_remove_document_rejects_out_of_range(index):

    with pytest.raises(IndexError):
        index.remove_document(100)