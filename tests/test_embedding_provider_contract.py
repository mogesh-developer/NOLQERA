import pytest

from nolqera.intelligence.semantic_similarity.embeddings.base import (
    EmbeddingProvider,
)
from nolqera.intelligence.semantic_similarity.embeddings.transformer import (
    TransformerEmbeddingProvider,
)


MODEL_NAME = "all-MiniLM-L6-v2"


def test_transformer_provider_implements_embedding_provider():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    assert isinstance(provider, EmbeddingProvider)


def test_transformer_provider_has_embed():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    assert callable(provider.embed)


def test_transformer_provider_has_embed_many():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    assert callable(provider.embed_many)


def test_transformer_provider_has_dimension():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    assert isinstance(provider.dimension, int)
    assert provider.dimension > 0


def test_embed_returns_list_of_floats():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    vector = provider.embed(
        "FastAPI is a backend framework"
    )

    assert isinstance(vector, list)
    assert len(vector) == provider.dimension
    assert all(
        isinstance(value, float)
        for value in vector
    )


def test_embed_many_returns_correct_shape():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    texts = [
        "FastAPI is a backend framework",
        "MongoDB is a database",
    ]

    vectors = provider.embed_many(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)

    for vector in vectors:
        assert isinstance(vector, list)
        assert len(vector) == provider.dimension


def test_embed_rejects_invalid_text():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    with pytest.raises(TypeError):
        provider.embed(123)


def test_embed_rejects_empty_text():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    with pytest.raises(ValueError):
        provider.embed("")


def test_embed_many_rejects_invalid_input():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    with pytest.raises(TypeError):
        provider.embed_many("not a list")


def test_embed_many_rejects_empty_list():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    with pytest.raises(ValueError):
        provider.embed_many([])


def test_embed_many_rejects_empty_string():
    provider = TransformerEmbeddingProvider(MODEL_NAME)

    with pytest.raises(ValueError):
        provider.embed_many([
            "FastAPI",
            "",
        ])