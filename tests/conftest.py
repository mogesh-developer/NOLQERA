import pytest

from nolqera.intelligence.semantic_search.index import (
    SemanticSearchIndex,
)
from nolqera.intelligence.semantic_similarity.embeddings.base import (
    EmbeddingProvider,
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


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str):
        return [1.0, 0.0, 0.0]


@pytest.fixture
def mock_components():
    from nolqera.intelligence.context_optimization.context_ranking import (
        ContextRanker,
    )
    from nolqera.intelligence.context_optimization.noise_detection import (
        NoiseDetector,
    )
    from nolqera.intelligence.entities.engine import EntityEngine
    from nolqera.intelligence.importance.engine import ImportanceEngine
    from nolqera.intelligence.intent.engine import IntentEngine
    from nolqera.intelligence.keyphrase.engine import KeyphraseEngine
    from nolqera.intelligence.pipeline.context_compressor import (
        ContextCompressor,
    )
    from nolqera.intelligence.pipeline.context_ranker import (
        ContextRankingAnalyzer,
    )
    from nolqera.intelligence.pipeline.noise_remover import (
        NoiseRemover,
    )
    from nolqera.intelligence.semantic_search.engine import (
        SemanticSearchEngine,
    )

    return {
        "semantic_search_engine": SemanticSearchEngine(
            embedding_provider=FakeEmbeddingProvider()
        ),
        "importance_engine": ImportanceEngine(),
        "keyphrase_engine": KeyphraseEngine(),
        "entity_engine": EntityEngine(),
        "intent_engine": IntentEngine(),
        "noise_remover": NoiseRemover(NoiseDetector()),
        "context_ranker": ContextRankingAnalyzer(ContextRanker()),
        "context_compressor": ContextCompressor(),
    }