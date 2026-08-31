from nolqera.intelligence.semantic_search.engine import (
    SemanticSearchEngine,
)
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)


def test_semantic_search_integrates_with_tfidf_provider():

    documents = [
        ["fastapi", "python", "backend", "api"],
        ["mongodb", "database", "storage"],
        ["react", "frontend", "javascript"],
        ["python", "machine", "learning"],
    ]

    provider = TFIDFEmbeddingProvider()
    provider.fit(documents)

    engine = SemanticSearchEngine(provider)

    search_documents = [
        "fastapi python backend api",
        "mongodb database storage",
        "react frontend javascript",
        "python machine learning",
    ]

    results = engine.search(
        "fastapi python backend",
        search_documents,
    )

    print("\n--- TF-IDF Semantic Search ---")

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
        "fastapi python backend api"
    )

    assert results[0].score > results[-1].score