from nolqera.intelligence.semantic_search.engine import (
    SemanticSearchEngine,
)
from nolqera.intelligence.semantic_similarity.embeddings.transformer import (
    TransformerEmbeddingProvider,
)


def test_semantic_search_integrates_with_transformer_provider():

    provider = TransformerEmbeddingProvider()

    engine = SemanticSearchEngine(provider)

    documents = [
        "FastAPI is a Python backend framework",
        "MongoDB is a NoSQL database",
        "React is a frontend library",
        "Python is used for machine learning",
    ]

    results = engine.search(
        "Python backend development",
        documents,
    )

    print("\n--- Transformer Semantic Search ---")

    for result in results:
        print(
            f"{result.score:.4f} | "
            f"{result.text}"
        )

    assert len(results) == 4

    assert results[0].score >= results[1].score
    assert results[1].score >= results[2].score
    assert results[2].score >= results[3].score

    assert results[0].text == (
        "FastAPI is a Python backend framework"
    )

    assert results[0].score > results[-1].score