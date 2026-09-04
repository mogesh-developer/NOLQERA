import pytest
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
from nolqera.intelligence.pipeline.models import PipelineResult
from nolqera.intelligence.pipeline.integration import (
    create_engine,
)
from nolqera.intelligence.pipeline.integration import (
    NOLQERAEngine,
)
from nolqera.intelligence.pipeline.integration import (
    create_engine,
)
def create_test_engine():
    from nolqera.intelligence.pipeline.integration import create_engine

    return create_engine(
        semantic_search_engine=SemanticSearchEngine(
            embedding_provider=FakeEmbeddingProvider()
        ),
        importance_engine=ImportanceEngine(),
        keyphrase_engine=KeyphraseEngine(),
        entity_engine=EntityEngine(),
        intent_engine=IntentEngine(),
        noise_remover=NoiseRemover(
            NoiseDetector()
        ),
        context_ranker=ContextRankingAnalyzer(
            ContextRanker()
        ),
        context_compressor=ContextCompressor(),
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



def test_create_engine_returns_public_engine():
    from nolqera.intelligence.pipeline.integration import (
        NOLQERAEngine,
        create_engine,
    )

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

    engine = create_engine(
        semantic_search_engine=semantic_search_engine,
        importance_engine=importance_engine,
        keyphrase_engine=keyphrase_engine,
        entity_engine=entity_engine,
        intent_engine=intent_engine,
        noise_remover=noise_remover,
        context_ranker=context_ranker,
        context_compressor=context_compressor,
    )

    assert isinstance(engine, NOLQERAEngine)

    result = engine.process(
        query="Python API",
        raw_input="Python API development with FastAPI",
    )

    assert isinstance(result, PipelineResult)

def test_public_engine_rejects_invalid_query():
    from nolqera.intelligence.pipeline.integration import create_engine

    engine = create_test_engine()

    with pytest.raises(TypeError):
        engine.process(
            query=123,
            raw_input="Python is a programming language.",
        )


def test_public_engine_rejects_empty_query():
    from nolqera.intelligence.pipeline.integration import create_engine

    engine = create_test_engine()

    with pytest.raises(ValueError):
        engine.process(
            query="   ",
            raw_input="Python is a programming language.",
        )


def test_public_engine_rejects_invalid_input():
    from nolqera.intelligence.pipeline.integration import create_engine

    engine = create_test_engine()

    with pytest.raises(TypeError):
        engine.process(
            query="Python",
            raw_input=123,
        )


def test_public_engine_rejects_empty_input():
    from nolqera.intelligence.pipeline.integration import create_engine

    engine = create_test_engine()

    with pytest.raises(ValueError):
        engine.process(
            query="Python",
            raw_input="   ",
        )

def test_public_engine_returns_unified_result():
    from nolqera.intelligence.pipeline.integration import create_engine

    engine = create_test_engine()

    result = engine.process(
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python framework."
        ),
    )

    assert isinstance(result, PipelineResult)

    assert isinstance(result.input_text, str)
    assert isinstance(result.normalized_text, str)
    assert isinstance(result.sentences, list)
    assert isinstance(result.relevance, list)
    assert isinstance(result.importance, list)
    assert result.keywords is not None
    assert isinstance(result.entities, list)
    assert isinstance(result.intents, list)
    assert isinstance(result.filtered_results, list)
    assert isinstance(result.ranked_context, list)
    assert isinstance(result.compressed_context, str)
    assert result.metadata is not None

def test_public_root_api_create_engine():
    from nolqera import create_engine
    from nolqera.intelligence.pipeline.integration import NOLQERAEngine

    semantic_search_engine = SemanticSearchEngine(
        embedding_provider=FakeEmbeddingProvider()
    )

    engine = create_engine(
        semantic_search_engine=semantic_search_engine,
        importance_engine=ImportanceEngine(),
        keyphrase_engine=KeyphraseEngine(),
        entity_engine=EntityEngine(),
        intent_engine=IntentEngine(),
        noise_remover=NoiseRemover(
            NoiseDetector()
        ),
        context_ranker=ContextRankingAnalyzer(
            ContextRanker()
        ),
        context_compressor=ContextCompressor(),
    )

    assert isinstance(engine, NOLQERAEngine)

    result = engine.process(
        query="Python API",
        raw_input="Python API development with FastAPI.",
    )

    assert isinstance(result, PipelineResult)
    assert result.compressed_context

def test_e2e_happy_path():
    from nolqera import create_engine

    semantic_search_engine = SemanticSearchEngine(
        embedding_provider=FakeEmbeddingProvider()
    )

    engine = create_engine(
        semantic_search_engine=semantic_search_engine,
        importance_engine=ImportanceEngine(),
        keyphrase_engine=KeyphraseEngine(),
        entity_engine=EntityEngine(),
        intent_engine=IntentEngine(),
        noise_remover=NoiseRemover(
            NoiseDetector()
        ),
        context_ranker=ContextRankingAnalyzer(
            ContextRanker()
        ),
        context_compressor=ContextCompressor(),
    )

    result = engine.process(
        query="Python API development",
        raw_input=(
            "Python is widely used for backend development. "
            "FastAPI is a Python framework for building APIs. "
            "React is commonly used for frontend development."
        ),
    )

    assert isinstance(result, PipelineResult)

    assert result.input_text == (
        "Python is widely used for backend development. "
        "FastAPI is a Python framework for building APIs. "
        "React is commonly used for frontend development."
    )

    assert result.normalized_text

    assert result.sentences

    assert result.relevance
    assert result.importance

    assert result.keywords is not None
    assert result.entities is not None
    assert result.intents is not None

    assert result.filtered_results is not None
    assert result.ranked_context is not None

    assert isinstance(result.compressed_context, str)
    assert result.compressed_context.strip()

    assert result.metadata is not None
    assert result.is_empty is False

def test_e2e_result_validation():
    from nolqera import create_engine

    engine = create_engine(
        semantic_search_engine=SemanticSearchEngine(
            embedding_provider=FakeEmbeddingProvider()
        ),
        importance_engine=ImportanceEngine(),
        keyphrase_engine=KeyphraseEngine(),
        entity_engine=EntityEngine(),
        intent_engine=IntentEngine(),
        noise_remover=NoiseRemover(
            NoiseDetector()
        ),
        context_ranker=ContextRankingAnalyzer(
            ContextRanker()
        ),
        context_compressor=ContextCompressor(),
    )

    result = engine.process(
        query="Python API",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python framework. "
            "React is a frontend library."
        ),
    )

    # Core result contract
    assert isinstance(result, PipelineResult)

    # Input → sentence processing
    assert result.input_text
    assert result.normalized_text
    assert len(result.sentences) > 0

    # Intelligence outputs
    assert len(result.relevance) == len(result.sentences)
    assert len(result.importance) > 0
    assert result.keywords is not None
    assert isinstance(result.entities, list)
    assert isinstance(result.intents, list)

    # Context processing
    assert isinstance(result.filtered_results, list)
    assert isinstance(result.ranked_context, list)
    assert isinstance(result.compressed_context, str)

    # Metadata consistency
    assert result.metadata.input_count == len(result.relevance)
    assert result.metadata.sentence_count == len(result.sentences)
    assert result.metadata.filtered_count == len(result.filtered_results)
    assert result.metadata.ranked_count == len(result.ranked_context)

    # Final state
    assert result.compressed_context.strip()
    assert result.is_empty is False

def test_e2e_component_interaction_flow():
    from nolqera import create_engine

    engine = create_engine(
        semantic_search_engine=SemanticSearchEngine(
            embedding_provider=FakeEmbeddingProvider()
        ),
        importance_engine=ImportanceEngine(),
        keyphrase_engine=KeyphraseEngine(),
        entity_engine=EntityEngine(),
        intent_engine=IntentEngine(),
        noise_remover=NoiseRemover(
            NoiseDetector()
        ),
        context_ranker=ContextRankingAnalyzer(
            ContextRanker()
        ),
        context_compressor=ContextCompressor(),
    )

    result = engine.process(
        query="Python FastAPI",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python framework. "
            "FastAPI is used to build APIs."
        ),
    )

    # Sentence segmentation produced usable input
    assert len(result.sentences) == 3

    # Relevance processed every sentence
    assert len(result.relevance) == len(result.sentences)

    # Importance received sentence-level information
    assert len(result.importance) > 0

    # Downstream intelligence stages produced outputs
    assert result.keywords is not None
    assert result.entities is not None
    assert result.intents is not None

    # Context pipeline produced progressively processed results
    assert isinstance(result.filtered_results, list)
    assert isinstance(result.ranked_context, list)
    assert isinstance(result.compressed_context, str)

    # Final output is actually connected to the pipeline
    assert result.compressed_context.strip()

    # Metadata reflects the actual pipeline flow
    assert result.metadata.sentence_count == 3
    assert result.metadata.input_count == len(result.relevance)
    assert result.metadata.filtered_count == len(result.filtered_results)
    assert result.metadata.ranked_count == len(result.ranked_context)

def test_e2e_invalid_input_flow():
    from nolqera import create_engine

    engine = create_engine(
        semantic_search_engine=SemanticSearchEngine(
            embedding_provider=FakeEmbeddingProvider()
        ),
        importance_engine=ImportanceEngine(),
        keyphrase_engine=KeyphraseEngine(),
        entity_engine=EntityEngine(),
        intent_engine=IntentEngine(),
        noise_remover=NoiseRemover(
            NoiseDetector()
        ),
        context_ranker=ContextRankingAnalyzer(
            ContextRanker()
        ),
        context_compressor=ContextCompressor(),
    )

    with pytest.raises(TypeError):
        engine.process(
            query=123,
            raw_input="Python is a programming language.",
        )

    with pytest.raises(ValueError):
        engine.process(
            query="   ",
            raw_input="Python is a programming language.",
        )

    with pytest.raises(TypeError):
        engine.process(
            query="Python",
            raw_input=123,
        )

    with pytest.raises(ValueError):
        engine.process(
            query="Python",
            raw_input="   ",
        )
def test_pipeline_process_with_external_entity_recognizer():
    from nolqera.intelligence.entities import (
        EntityEngine,
        HuggingFaceEntityRecognizer,
    )
    from nolqera.intelligence.pipeline.integration import create_engine

    engine = create_engine(
        semantic_search_engine=SemanticSearchEngine(
            embedding_provider=FakeEmbeddingProvider()
        ),
        importance_engine=ImportanceEngine(),
        keyphrase_engine=KeyphraseEngine(),
        entity_engine=EntityEngine(),
        intent_engine=IntentEngine(),
        noise_remover=NoiseRemover(
            NoiseDetector()
        ),
        context_ranker=ContextRankingAnalyzer(
            ContextRanker()
        ),
        context_compressor=ContextCompressor(),
        use_external_recognizer=True,
        recognizer=HuggingFaceEntityRecognizer(),
    )

    text = (
        "The application is built using FastAPI. "
        "The application uses MongoDB for data storage. "
        "Python is the main programming language."
    )

    result = engine.process(
        query="What technologies are used?",
        raw_input=text,
    )

    assert isinstance(result, PipelineResult)

    values = [entity.text for entity in result.entities]

    assert "FastAPI" in values
    assert "MongoDB" in values
    assert "Python" in values