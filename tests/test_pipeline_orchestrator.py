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
from nolqera.intelligence.pipeline.context_compressor import (
    ContextCompressor,
)
from nolqera.intelligence.pipeline.context_ranker import (
    ContextRankingAnalyzer,
)
from nolqera.intelligence.pipeline.noise_remover import (
    NoiseRemover,
)
from nolqera.intelligence.pipeline.orchestrator import (
    NOLQERAPipeline,
    PipelineResult,
)
from nolqera.intelligence.semantic_search.engine import (
    SemanticSearchEngine,
)
from nolqera.intelligence.semantic_similarity.embeddings.base import (
    EmbeddingProvider,
)
from nolqera.intelligence.pipeline.config import PipelineConfig
from nolqera.intelligence.pipeline.models import PipelineResult

class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic embedding provider for pipeline tests.
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


def create_pipeline():
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
        config=PipelineConfig(
            keyword_top_k=5,
            max_sentences=2,
        ),
    )


def test_pipeline_can_be_created():
    pipeline = create_pipeline()

    assert isinstance(
        pipeline,
        NOLQERAPipeline,
    )


def test_pipeline_returns_pipeline_result():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python FastAPI",
        raw_input=(
            "FastAPI is a Python framework. "
            "Python is widely used for backend development."
        ),
    )

    assert isinstance(
        result,
        PipelineResult,
    )


def test_pipeline_normalizes_input():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python",
        raw_input=(
            "   FastAPI   is a Python framework.   "
        ),
    )

    assert result.normalized_text == (
        "FastAPI is a Python framework."
    )


def test_pipeline_segments_sentences():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python framework."
        ),
    )

    assert result.sentences == [
        "Python is a programming language.",
        "FastAPI is a Python framework.",
    ]


def test_pipeline_runs_relevance_analysis():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "The weather is sunny."
        ),
    )

    assert len(result.relevance) == 2

    assert result.relevance[0]["index"] == 0
    assert result.relevance[1]["index"] == 1


def test_pipeline_runs_importance_analysis():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python framework."
        ),
    )

    assert isinstance(
        result.importance,
        list,
    )

    assert len(result.importance) > 0


def test_pipeline_runs_keyword_analysis():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python framework."
        ),
    )

    assert result.keywords is not None


def test_pipeline_runs_entity_analysis():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python framework."
        ),
    )

    assert isinstance(
        result.entities,
        list,
    )


def test_pipeline_runs_intent_analysis():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="What is Python?",
        raw_input=(
            "Python is a programming language."
        ),
    )

    assert isinstance(
        result.intents,
        list,
    )


def test_pipeline_runs_context_ranking():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python framework."
        ),
    )

    assert isinstance(
        result.ranked_context,
        list,
    )


def test_pipeline_runs_context_compression():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python framework. "
            "React is a frontend library."
        ),
    )

    assert isinstance(
        result.compressed_context,
        str,
    )

    assert result.compressed_context


def test_pipeline_respects_max_sentences():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python",
        raw_input=(
            "Python is a programming language. "
            "FastAPI is a Python framework. "
            "Python is useful for backend systems."
        ),
    )

    compressed_sentences = [
        sentence.strip()
        for sentence in result.compressed_context.split(".")
        if sentence.strip()
    ]

    assert len(compressed_sentences) <= 2


def test_pipeline_reports_empty_state_correctly():
    pipeline = create_pipeline()

    result = pipeline.process(
        query="Python",
        raw_input=(
            "Python is a programming language."
        ),
    )

    assert result.is_empty is False


def test_pipeline_rejects_invalid_query():
    pipeline = create_pipeline()

    with pytest.raises(TypeError):
        pipeline.process(
            query=123,
            raw_input="Python is a language.",
        )


def test_pipeline_rejects_empty_query():
    pipeline = create_pipeline()

    with pytest.raises(ValueError):
        pipeline.process(
            query="   ",
            raw_input="Python is a language.",
        )


def test_pipeline_rejects_invalid_input():
    pipeline = create_pipeline()

    with pytest.raises(TypeError):
        pipeline.process(
            query="Python",
            raw_input=123,
        )


def test_pipeline_rejects_empty_input():
    pipeline = create_pipeline()

    with pytest.raises(ValueError):
        pipeline.process(
            query="Python",
            raw_input="   ",
        )


def test_pipeline_preserves_original_input():
    pipeline = create_pipeline()

    raw_input = (
        "Python is a programming language. "
        "FastAPI is a Python framework."
    )

    result = pipeline.process(
        query="Python",
        raw_input=raw_input,
    )

    assert result.input_text == raw_input