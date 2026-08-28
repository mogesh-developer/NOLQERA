import pytest

from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)


def test_tfidf_provider_fits_and_embeds():

    provider = TFIDFEmbeddingProvider()

    documents = [
        ["fastapi", "backend"],
        ["mongodb", "database"],
        ["fastapi", "api"],
    ]

    provider.fit(documents)

    vector = provider.embed(
        ["fastapi", "backend"]
    )

    assert isinstance(vector, list)
    assert len(vector) == len(
        provider.vocabulary
    )


def test_tfidf_provider_embeds_many():

    provider = TFIDFEmbeddingProvider()

    documents = [
        ["fastapi", "backend"],
        ["mongodb", "database"],
        ["fastapi", "api"],
    ]

    provider.fit(documents)

    vectors = provider.embed_many(documents)

    assert len(vectors) == 3

    vocabulary_size = len(
        provider.vocabulary
    )

    assert all(
        len(vector) == vocabulary_size
        for vector in vectors
    )


def test_tfidf_provider_exposes_vocabulary():

    provider = TFIDFEmbeddingProvider()

    provider.fit(
        [
            ["fastapi", "backend"],
            ["mongodb", "database"],
        ]
    )

    vocabulary = provider.vocabulary

    assert isinstance(vocabulary, list)
    assert "fastapi" in vocabulary
    assert "mongodb" in vocabulary


def test_embedding_before_fit_is_rejected():

    provider = TFIDFEmbeddingProvider()

    with pytest.raises(RuntimeError):
        provider.embed(["fastapi"])


def test_empty_documents_are_rejected():

    provider = TFIDFEmbeddingProvider()

    with pytest.raises(ValueError):
        provider.fit([])


def test_invalid_documents_are_rejected():

    provider = TFIDFEmbeddingProvider()

    with pytest.raises(TypeError):
        provider.fit("invalid")
