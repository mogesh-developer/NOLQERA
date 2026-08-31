import pytest
from nolqera.intelligence.semantic_similarity.engine import (
    SemanticSimilarityEngine,
)
from nolqera.intelligence.semantic_similarity.embeddings.transformer import (
    TransformerEmbeddingProvider,
)

MODEL_NAME = "all-MiniLM-L6-v2"


def test_semantic_similarity_integrates_with_transformer_provider():

    provider = TransformerEmbeddingProvider(model_name=MODEL_NAME)

    engine = SemanticSimilarityEngine(provider)

    similar_result = engine.compare(
        ["fastapi", "backend", "api"],
        ["fastapi", "rest", "api"],
    )

    unrelated_result = engine.compare(
        ["fastapi", "backend"],
        ["mongodb", "database"],
    )

    print("\n--- Transformer Semantic Similarity ---")

    print(
        f"Similar   : {similar_result.score:.4f}"
    )

    print(
        f"Unrelated : {unrelated_result.score:.4f}"
    )

    assert similar_result.score > unrelated_result.score