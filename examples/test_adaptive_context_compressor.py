from __future__ import annotations

from nolqera import (
    NOLQERAPipeline,
    PipelineConfig,
    create_default_configured_pipeline,
)
from nolqera.intelligence.context_optimization.adaptive_context_compressor import (
    AdaptiveContextCompressor,
)
from nolqera.intelligence.context_optimization.context_ranking import ContextRanker
from nolqera.intelligence.context_optimization.final_context_compressor import (
    FinalContextCompressor,
)
from nolqera.intelligence.context_optimization.noise_detection import NoiseDetector
from nolqera.intelligence.context_optimization.redundancy_aware_compression import (
    RedundancyAwareCompressor,
)
from nolqera.intelligence.context_optimization.token_reduction import (
    TokenReductionStrategy,
)
from nolqera.intelligence.entities.engine import EntityEngine
from nolqera.intelligence.importance.engine import ImportanceEngine
from nolqera.intelligence.intent.engine import IntentEngine
from nolqera.intelligence.keyphrase.engine import KeyphraseEngine
from nolqera.intelligence.pipeline.context_compressor import ContextCompressor
from nolqera.intelligence.pipeline.context_ranker import ContextRankingAnalyzer
from nolqera.intelligence.pipeline.noise_remover import NoiseRemover
from nolqera.intelligence.semantic_search.engine import SemanticSearchEngine
from nolqera.intelligence.semantic_similarity.embeddings.transformer import (
    TransformerEmbeddingProvider,
)


QUERY = (
    "What are the main benefits and limitations of Python, "
    "and when should a developer consider using another language?"
)

RAW_CONTEXT = """
Python is a high-level general-purpose programming language known for its readable syntax and large ecosystem of libraries.

One important benefit of Python is developer productivity.

Its readable syntax also makes Python code easier to understand and maintain.

Python programs can often be written with fewer lines of code than equivalent programs in lower-level languages.

Python has a large ecosystem of third-party packages.

Python also has a large global developer community.

Python is popular in artificial intelligence and data science.

Python is widely used for automation because scripts can interact with files, APIs, databases, operating system processes, and external services.

Python applications can run on Windows, Linux, and macOS with relatively few changes.

Another benefit is cross-platform support.

Python is therefore especially useful when development speed, readability, ecosystem support, automation, and data or AI capabilities are important requirements.

Python is different because it is commonly used for backend development, automation, data science, machine learning, and scripting.

Libraries and frameworks allow developers to build machine learning models, process datasets, perform statistical analysis, and create AI applications without implementing every algorithm from scratch.

Developers can use frameworks such as FastAPI and Django for web development, NumPy and pandas for data processing, and PyTorch and TensorFlow for machine learning.

Developers can find extensive documentation, tutorials, open-source projects, and community support when solving technical problems.

A limitation of Python is that it is generally slower than compiled languages such as C++ or Rust for CPU-intensive workloads.

Python can also have higher memory usage in some applications.

For performance-critical systems, developers may combine Python with optimized native libraries or other programming languages.

Developers should consider another language when requirements strongly prioritize CPU performance, very low memory usage, or platform-specific capabilities that are better served by another technology.

Java has strong performance, mature tooling, and a large ecosystem, but its syntax can require more code for some tasks compared with Python.

Java is another popular programming language used for enterprise applications.

JavaScript is widely used for web development, especially in browser-based applications.

This allows repetitive manual tasks to be automated efficiently.
"""


def exact_duplicate(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def word_counter(text: str) -> int:
    return len(text.split())


def build_nolqera_pipeline(max_sentences: int = 16) -> NOLQERAPipeline:
    embedding_provider = TransformerEmbeddingProvider(
        model_name="all-MiniLM-L6-v2"
    )

    semantic_search_engine = SemanticSearchEngine(
        embedding_provider=embedding_provider
    )

    return create_default_configured_pipeline(
        semantic_search_engine=semantic_search_engine,
        importance_engine=ImportanceEngine(),
        keyphrase_engine=KeyphraseEngine(),
        entity_engine=EntityEngine(),
        intent_engine=IntentEngine(),
        noise_remover=NoiseRemover(NoiseDetector()),
        context_ranker=ContextRankingAnalyzer(ContextRanker()),
        context_compressor=ContextCompressor(),
        config=PipelineConfig(max_sentences=max_sentences),
    )


def build_adaptive_compressor(max_sentences: int = 16) -> AdaptiveContextCompressor:
    redundancy_compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate
    )

    token_reduction_strategy = TokenReductionStrategy(
        token_counter=word_counter
    )

    pipeline = build_nolqera_pipeline(max_sentences=max_sentences)

    def entity_extractor(text: str) -> list[str]:
        if not text or not text.strip():
            return []
        result = pipeline.process(
            query=QUERY,
            raw_input=text,
        )
        entities = []
        for e in result.entities:
            if isinstance(e, str):
                entities.append(e)
            elif hasattr(e, "text"):
                entities.append(e.text)
            elif isinstance(e, dict) and "text" in e:
                entities.append(e["text"])
        return entities

    inner_compressor = FinalContextCompressor(
        redundancy_compressor=redundancy_compressor,
        token_reduction_strategy=token_reduction_strategy,
        entity_extractor=entity_extractor,
        max_sentences=max_sentences,
        importance_threshold=0.70,
        require_preservation=True,
    )

    return AdaptiveContextCompressor(compressor=inner_compressor)


def main() -> None:
    print("=" * 72)
    print("NOLQERA ADAPTIVE CONTEXT COMPRESSOR INTEGRATION TEST")
    print("=" * 72)

    print()
    print("Query:")
    print(QUERY)

    print()
    print("=" * 72)
    print("1. RUNNING NOLQERA PIPELINE")
    print("=" * 72)

    pipeline = build_nolqera_pipeline(max_sentences=16)

    result = pipeline.process(
        query=QUERY,
        raw_input=RAW_CONTEXT,
    )

    ranked_context = result.ranked_context

    print(f"Ranked contexts : {len(ranked_context)}")

    print()
    print("=" * 72)
    print("2. RUNNING ADAPTIVE CONTEXT COMPRESSOR (max_sentences='auto')")
    print("=" * 72)

    adaptive_compressor = build_adaptive_compressor()

    token_budget = 1000

    final_result = adaptive_compressor.compress(
        ranked_context,
        token_budget=token_budget,
        max_sentences="auto",
    )

    print("Adaptive compression completed ✓")

    print()
    print("=" * 72)
    print("3. COMPRESSION RESULT")
    print("=" * 72)

    print(f"Original sentences : {final_result.original_count}")
    print(f"Final sentences    : {final_result.final_count}")

    print(f"Original tokens    : {final_result.original_tokens}")
    print(f"Compressed tokens  : {final_result.compressed_tokens}")

    print(f"Token reduction    : {final_result.token_reduction}")
    print(
        f"Reduction percent  : "
        f"{final_result.reduction_percentage:.2f}%"
    )

    print()
    print("=" * 72)
    print("4. PRESERVATION")
    print("=" * 72)

    print(
        "Information preserved :",
        final_result.information_preservation.is_preserved,
    )

    print(
        "Entities preserved    :",
        final_result.entity_preservation.is_preserved,
    )

    print(
        "Facts preserved       :",
        final_result.fact_preservation.is_preserved,
    )

    print(
        "FINAL PRESERVATION    :",
        final_result.is_preserved,
    )

    print()
    print("=" * 72)
    print("5. ADAPTIVE OPTIMIZED CONTEXT")
    print("=" * 72)

    print(final_result.text)

    print()
    print("=" * 72)
    print("ADAPTIVE INTEGRATION TEST COMPLETE ✓")
    print("=" * 72)


if __name__ == "__main__":
    main()
