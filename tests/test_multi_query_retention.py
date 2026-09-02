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


class MultiTopicEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic embedding provider for multi-query
    retention validation.

    Each dimension represents a different query topic.
    """

    TOPICS = [
        "benefit",
        "fastapi",
        "ml",
        "performance",
        "platform",
    ]

    TERMS = {
        "benefit": [
            "benefit",
            "productivity",
            "readable",
            "syntax",
            "ecosystem",
        ],
        "fastapi": [
            "fastapi",
            "backend",
            "api",
            "web",
        ],
        "ml": [
            "pytorch",
            "tensorflow",
            "machine learning",
            "artificial intelligence",
            "data science",
            "numpy",
            "pandas",
        ],
        "performance": [
            "limitation",
            "slower",
            "performance",
            "cpu",
            "memory",
            "rust",
            "c++",
        ],
        "platform": [
            "cross-platform",
            "windows",
            "linux",
            "macos",
        ],
    }

    def embed(self, text):
        if isinstance(text, list):
            text = " ".join(text)

        text = text.lower()

        return [
            float(
                sum(
                    term in text
                    for term in self.TERMS[topic]
                )
            )
            for topic in self.TOPICS
        ]

    def embed_many(self, texts):
        return [
            self.embed(text)
            for text in texts
        ]


def build_pipeline():
    return create_default_configured_pipeline(
        semantic_search_engine=SemanticSearchEngine(
            MultiTopicEmbeddingProvider()
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
        config=PipelineConfig(
            max_sentences=16,
        ),
    )


RAW_CONTEXT = """
Python is known for readable syntax and high developer productivity.

FastAPI is a Python web framework used for backend APIs and
asynchronous endpoints.

Django is another Python web framework for backend development.

NumPy and pandas are widely used for data processing and data science.

PyTorch and TensorFlow are used for machine learning and artificial
intelligence.

Python is generally slower than compiled languages such as C++ and
Rust for CPU-intensive workloads.

Python can have higher memory usage in some applications.

Python supports cross-platform development across Windows, Linux,
and macOS.

Python is widely used for automation involving files, APIs, and
databases.

JavaScript is commonly used for browser-based web applications.
""".strip()


MULTI_QUERY_CASES = [
    (
        "What are the main benefits of Python?",
        [
            "readable syntax",
            "developer productivity",
        ],
    ),
    (
        "Which framework is used for backend APIs?",
        [
            "FastAPI",
            "backend APIs",
        ],
    ),
    (
        "Which Python libraries are used for machine learning?",
        [
            "PyTorch",
            "TensorFlow",
        ],
    ),
    (
        "What are Python's performance and memory limitations?",
        [
            "slower",
            "memory usage",
        ],
    ),
    (
        "Which operating systems does Python support?",
        [
            "Windows",
            "Linux",
            "macOS",
        ],
    ),
]


@pytest.mark.parametrize(
    "query, expected_information",
    MULTI_QUERY_CASES,
)
def test_multi_query_retention(
    query,
    expected_information,
):
    pipeline = build_pipeline()

    result = run_pipeline(
        pipeline=pipeline,
        query=query,
        raw_input=RAW_CONTEXT,
    )

    optimized_context = (
        result.compressed_context.casefold()
    )

    missing_information = [
        item
        for item in expected_information
        if item.casefold()
        not in optimized_context
    ]

    assert not missing_information, (
        f"Query failed: {query}\n"
        f"Missing information: "
        f"{missing_information}\n"
        f"Optimized context:\n"
        f"{result.compressed_context}"
    )