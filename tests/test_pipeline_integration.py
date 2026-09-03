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
from nolqera.intelligence.pipeline.config import PipelineConfig
from nolqera.intelligence.pipeline.context_compressor import (
    ContextCompressor,
)
from nolqera.intelligence.pipeline.context_ranker import (
    ContextRankingAnalyzer,
)
from nolqera.intelligence.pipeline.integration import (
    NOLQERAEngine,
)
from nolqera.intelligence.pipeline.noise_remover import (
    NoiseRemover,
)
from nolqera.intelligence.pipeline.orchestrator import (
    NOLQERAPipeline,
)
from nolqera.intelligence.semantic_search.engine import (
    SemanticSearchEngine,
)
from nolqera.intelligence.semantic_similarity.embeddings.base import (
    EmbeddingProvider,
)


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str):
        return [1.0, 0.0, 0.0]


def create_pipeline(config: PipelineConfig | None = None):
    semantic_search_engine = SemanticSearchEngine(
        embedding_provider=FakeEmbeddingProvider()
    )

    importance_engine = ImportanceEngine()

    keyphrase_engine = KeyphraseEngine()

    entity_engine = EntityEngine()

    intent_engine = IntentEngine()

    noise_remover = NoiseRemover(
        NoiseDetector()
    )

    context_ranker = ContextRankingAnalyzer(
        ContextRanker()
    )

    context_compressor = ContextCompressor()

    return NOLQERAPipeline(
        semantic_search_engine=semantic_search_engine,
        importance_engine=importance_engine,
        keyphrase_engine=keyphrase_engine,
        entity_engine=entity_engine,
        intent_engine=intent_engine,
        noise_remover=noise_remover,
        context_ranker=context_ranker,
        context_compressor=context_compressor,
        config=config,
    )


def test_nolqera_engine_process():
    pipeline = create_pipeline()

    engine = NOLQERAEngine(pipeline)

    result = engine.process(
        query="Python API",
        raw_input="Python API development with FastAPI",
    )

    assert result is not None