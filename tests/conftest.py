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

    fit_documents = [
        ["fastapi", "python", "backend", "api"],
        ["python", "machine", "learning"],
        ["mongodb", "database", "storage"],
        ["django", "python", "framework"],
        ["docker", "container"],
    ]

    provider = TFIDFEmbeddingProvider()
    provider.fit(fit_documents)

    return SemanticSearchIndex(
        embedding_provider=provider,
        documents=documents,
    )