from nolqera.intelligence.semantic_similarity.engine import (
    SemanticSimilarityEngine,
)
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)


def test_semantic_similarity_integrates_with_tfidf_provider():

    documents = [
        ["fastapi", "backend", "api"],
        ["fastapi", "rest", "api"],
        ["mongodb", "database", "storage"],
    ]

    provider = TFIDFEmbeddingProvider()
    provider.fit(documents)

    engine = SemanticSimilarityEngine(provider)

    similar_result = engine.compare(
        ["fastapi", "backend"],
        ["fastapi", "api"],
    )

    unrelated_result = engine.compare(
        ["fastapi", "backend"],
        ["mongodb", "database"],
    )

    print("\n--- Semantic Similarity Integration ---")

    print(
        f"Similar   : {similar_result.score:.4f}"
    )

    print(
        f"Unrelated : {unrelated_result.score:.4f}"
    )

    assert similar_result.score > unrelated_result.score