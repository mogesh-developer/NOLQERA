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
            [
                "fastapi",
                "python",
                "backend",
                "api",
            ],
            [
                "python",
                "machine",
                "learning",
            ],
            [
                "mongodb",
                "database",
                "storage",
            ],
            [
                "django",
                "python",
                "web",
                "framework",
            ],
            [
                "updated",
                "document",
            ],
        ]
    )

    return SemanticSearchIndex(
        provider,
        documents,
    )

def test_update_document(index):
    index.update_document(
        0,
        "django python web framework",
    )

    assert index.documents[0] == (
        "django python web framework"
    )


def test_update_document_preserves_index(index):
    index.update_document(
        1,
        "updated document",
    )

    results = index.search(
        "updated document",
        top_k=1,
    )

    assert results[0].index == 1


def test_update_document_changes_search_embedding(index):
    index.update_document(
        0,
        "django python web framework",
    )

    results = index.search(
        "django python",
        top_k=1,
    )

    assert results[0].index == 0


def test_update_document_rejects_non_integer_index(index):
    with pytest.raises(TypeError):
        index.update_document(
            "0",
            "updated",
        )


def test_update_document_rejects_negative_index(index):
    with pytest.raises(ValueError):
        index.update_document(
            -1,
            "updated",
        )


def test_update_document_rejects_out_of_range(index):
    with pytest.raises(IndexError):
        index.update_document(
            999,
            "updated",
        )


def test_update_document_rejects_non_string(index):
    with pytest.raises(TypeError):
        index.update_document(
            0,
            123,
        )


def test_update_document_rejects_empty_string(index):
    with pytest.raises(ValueError):
        index.update_document(
            0,
            "",
        )