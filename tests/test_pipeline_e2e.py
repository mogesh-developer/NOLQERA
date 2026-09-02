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
from nolqera.intelligence.pipeline.integration import (
    create_default_configured_pipeline,
    run_pipeline,
)
from nolqera.intelligence.pipeline.models import (
    PipelineResult,
)
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
from nolqera.intelligence.semantic_similarity.embeddings.base import (
    EmbeddingProvider,
)


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic embedding provider for E2E tests.
    """

    def embed(self, text):
        if isinstance(text, list):
            text = " ".join(text)

        text = text.lower()

        if "python" in text or "fastapi" in text:
            return [1.0, 0.0]

        return [0.0, 1.0]

    def embed_many(self, texts):
        return [
            self.embed(text)
            for text in texts
        ]


def create_e2e_pipeline(
    config=None,
):
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

    return create_default_configured_pipeline(
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


def test_complete_pipeline_returns_result():
    pipeline = create_e2e_pipeline()

    result = run_pipeline(
        pipeline=pipeline,
        query="Python FastAPI",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python web framework."
        ),
    )

    assert isinstance(
        result,
        PipelineResult,
    )


def test_complete_pipeline_preserves_input():
    pipeline = create_e2e_pipeline()

    raw_input = (
        "Python is a programming language. "
        "FastAPI is a Python web framework."
    )

    result = run_pipeline(
        pipeline=pipeline,
        query="Python",
        raw_input=raw_input,
    )

    assert result.input_text == raw_input
    assert result.normalized_text


def test_complete_pipeline_executes_all_major_stages():
    pipeline = create_e2e_pipeline()

    result = run_pipeline(
        pipeline=pipeline,
        query="Python FastAPI",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python web framework."
        ),
    )

    assert result.sentences
    assert result.relevance
    assert result.importance
    assert result.keywords is not None
    assert result.entities is not None
    assert result.intents is not None
    assert isinstance(
        result.filtered_results,
        list,
    )
    assert isinstance(
        result.ranked_context,
        list,
    )
    assert isinstance(
        result.compressed_context,
        str,
    )


def test_complete_pipeline_produces_optimized_context():
    pipeline = create_e2e_pipeline()

    result = run_pipeline(
        pipeline=pipeline,
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python web framework. "
            "React is a frontend library."
        ),
    )

    assert result.compressed_context
    assert result.is_empty is False


def test_complete_pipeline_metadata_is_consistent():
    pipeline = create_e2e_pipeline()

    result = run_pipeline(
        pipeline=pipeline,
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python web framework."
        ),
    )

    assert result.metadata.input_count == (
        len(result.relevance)
    )

    assert result.metadata.sentence_count == (
        len(result.sentences)
    )

    assert result.metadata.filtered_count == (
        len(result.filtered_results)
    )

    assert result.metadata.ranked_count == (
        len(result.ranked_context)
    )


def test_complete_pipeline_respects_configuration():
    config = PipelineConfig(
        keyword_top_k=2,
        max_sentences=1,
    )

    pipeline = create_e2e_pipeline(
        config=config,
    )

    result = run_pipeline(
        pipeline=pipeline,
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python web framework. "
            "Python is useful for backend development."
        ),
    )

    assert pipeline.config is config
    assert result.compressed_context


def test_complete_pipeline_handles_multiple_contexts():
    pipeline = create_e2e_pipeline()

    result = run_pipeline(
        pipeline=pipeline,
        query="Python backend",
        raw_input=(
            "Python is used for backend development. "
            "FastAPI is built with Python. "
            "Django is another Python web framework. "
            "React is commonly used for frontend development."
        ),
    )

    assert len(result.sentences) == 4
    assert result.relevance
    assert result.ranked_context
    assert result.compressed_context


def test_complete_pipeline_rejects_empty_query():
    pipeline = create_e2e_pipeline()

    with pytest.raises(ValueError):
        run_pipeline(
            pipeline=pipeline,
            query="",
            raw_input=(
                "Python is a programming language."
            ),
        )


def test_complete_pipeline_rejects_empty_input():
    pipeline = create_e2e_pipeline()

    with pytest.raises(ValueError):
        run_pipeline(
            pipeline=pipeline,
            query="Python",
            raw_input="",
        )


def test_complete_pipeline_rejects_invalid_pipeline():
    with pytest.raises(TypeError):
        run_pipeline(
            pipeline="invalid",
            query="Python",
            raw_input=(
                "Python is a programming language."
            ),
        )


def test_complete_pipeline_rejects_invalid_query_type():
    pipeline = create_e2e_pipeline()

    with pytest.raises(TypeError):
        run_pipeline(
            pipeline=pipeline,
            query=123,
            raw_input=(
                "Python is a programming language."
            ),
        )


def test_complete_pipeline_rejects_invalid_input_type():
    pipeline = create_e2e_pipeline()

    with pytest.raises(TypeError):
        run_pipeline(
            pipeline=pipeline,
            query="Python",
            raw_input=123,
        )