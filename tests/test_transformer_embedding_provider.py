from sentence_transformers import multi_vector_encoder
import pytest

from nolqera.intelligence.semantic_similarity.embeddings.transformer import (
    TransformerEmbeddingProvider,
)


MODEL_NAME = "all-MiniLM-L6-v2"


@pytest.fixture
def provider():
    return TransformerEmbeddingProvider(
        model_name=MODEL_NAME
    )


def test_transformer_provider_creates_provider():
    provider = TransformerEmbeddingProvider(
        model_name=MODEL_NAME
    )

    assert provider.model_name == MODEL_NAME


def test_transformer_provider_embeds_text(provider):
    vector = provider.embed(
        "FastAPI is a backend framework"
    )

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(
        isinstance(value, float)
        for value in vector
    )


def test_transformer_provider_embeds_many_texts(provider):
    texts = [
        "FastAPI is a backend framework",
        "MongoDB is a database",
    ]

    vectors = provider.embed_many(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == 2

    assert all(
        isinstance(vector, list)
        for vector in vectors
    )

    assert all(
        len(vector) == provider.dimension
        for vector in vectors
    )


def test_transformer_provider_dimension(provider):
    vector = provider.embed(
        "FastAPI backend"
    )

    assert provider.dimension == len(vector)
    assert provider.dimension > 0


def test_transformer_provider_preserves_input_order(provider):
    texts = [
        "FastAPI backend framework",
        "MongoDB database",
        "Python programming language",
    ]

    vectors = provider.embed_many(texts)

    assert len(vectors) == len(texts)

    for vector in vectors:
        assert len(vector) == provider.dimension


def test_transformer_provider_rejects_empty_text(provider):
    with pytest.raises(ValueError):
        provider.embed("")


def test_transformer_provider_rejects_non_string(provider):
    with pytest.raises(TypeError):
        provider.embed(123)


def test_transformer_provider_rejects_empty_list(provider):
    with pytest.raises(ValueError):
        provider.embed_many([])


def test_transformer_provider_rejects_invalid_list(provider):
    with pytest.raises(TypeError):
        provider.embed_many(
            [
                "FastAPI backend",
                123,
            ]
        )


def test_transformer_provider_rejects_empty_string_in_list(provider):
    with pytest.raises(ValueError):
        provider.embed_many(
            [
                "FastAPI backend",
                "",
            ]
        )


def test_transformer_provider_rejects_invalid_model_name():
    with pytest.raises(TypeError):
        TransformerEmbeddingProvider(
            model_name=123
        )


def test_transformer_provider_rejects_empty_model_name():
    with pytest.raises(ValueError):
        TransformerEmbeddingProvider(
            model_name=""
        )


def test_transformer_provider_rejects_whitespace_model_name():
    with pytest.raises(ValueError):
        TransformerEmbeddingProvider(
            model_name="   "
        )
    
def test_transformer_provider_embeds_text(provider):
    text = "FastAPI is a backend framework"

    vector = provider.embed(text)

    print("\n--- Transformer Embedding ---")
    print(f"Text      : {text}")
    print(f"Dimension : {len(vector)}")
    print(f"Vector    : {vector}")

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(
        isinstance(value, float)
        for value in vector
    )

def test_transformer_provider_embeds_many_texts(provider):
    texts = [
        "FastAPI is a backend framework",
        "MongoDB is a database",
    ]

    vectors = provider.embed_many(texts)

    print("\n--- Transformer Embeddings ---")

    for index, (text, vector) in enumerate(
        zip(texts, vectors),
        start=1,
    ):
        print(f"\nText {index}: {text}")
        print(f"Dimension: {len(vector)}")
        print(f"Vector: {vector}")

    assert isinstance(vectors, list)
    assert len(vectors) == 2

    assert all(
        isinstance(vector, list)
        for vector in vectors
    )

    assert all(
        len(vector) == provider.dimension
        for vector in vectors
    )
    print(f"Vector[:10]: {vector[:10]}")
    print(f"Dimension : {len(vector)}")