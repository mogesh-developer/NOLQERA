from __future__ import annotations

import json
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from nolqera import (
    NOLQERAPipeline,
    run_pipeline,
    create_default_configured_pipeline,
)

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
from nolqera.intelligence.semantic_similarity.embeddings.transformer import (
    TransformerEmbeddingProvider,
)


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"


# ============================================================
# REALISTIC DOCUMENT CONTEXT
# ============================================================

QUERY = (
    "What are the main benefits of Python for modern "
    "software development?"
)

RAW_CONTEXT = """
Python is a high-level general-purpose programming language
known for its readable syntax and large ecosystem of libraries.
It is widely used for web development, automation, data analysis,
machine learning, scientific computing, and scripting.

One important benefit of Python is developer productivity.
Python programs can often be written with fewer lines of code
than equivalent programs in lower-level languages. Its readable
syntax also makes Python code easier to understand and maintain.

Python has a large ecosystem of third-party packages.
Developers can use frameworks such as FastAPI and Django for
web development, NumPy and pandas for data processing, and
PyTorch and TensorFlow for machine learning.

Python is also popular in artificial intelligence and data science.
Libraries and frameworks allow developers to build machine learning
models, process datasets, perform statistical analysis, and create
AI applications without implementing every algorithm from scratch.

Another benefit is cross-platform support. Python applications
can run on Windows, Linux, and macOS with relatively few changes.
This makes Python useful for teams that develop software across
different operating systems.

Python is widely used for automation because scripts can interact
with files, APIs, databases, operating system processes, and
external services. This allows repetitive manual tasks to be
automated efficiently.

Python also has a large global developer community. This means
developers can find extensive documentation, tutorials, open-source
projects, and community support when solving technical problems.

A limitation of Python is that it is generally slower than
compiled languages such as C++ or Rust for CPU-intensive workloads.
Python also has higher memory usage in some applications. For
performance-critical systems, developers may combine Python with
optimized native libraries or other programming languages.

Java is another popular programming language used for enterprise
applications. Java has strong performance, mature tooling, and a
large ecosystem, but its syntax can require more code for some
tasks compared with Python.

JavaScript is widely used for web development, especially in
browser-based applications. Python is different because it is
commonly used for backend development, automation, data science,
machine learning, and scripting.

Python is therefore especially useful when development speed,
readability, ecosystem support, automation, and data or AI
capabilities are important requirements.
""".strip()


# ============================================================
# NOLQERA PIPELINE
# ============================================================

def build_nolqera_pipeline() -> NOLQERAPipeline:
    """
    Build the existing NOLQERA pipeline using its public
    components. No NOLQERA core logic is modified.
    """

    embedding_provider = TransformerEmbeddingProvider(
        model_name="all-MiniLM-L6-v2"
    )

    semantic_search_engine = SemanticSearchEngine(
        embedding_provider=embedding_provider
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
    )


# ============================================================
# OLLAMA CLIENT
# ============================================================

def call_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
) -> tuple[str, float]:
    """
    Send a prompt to the local Ollama server.

    Returns:
        (generated_answer, latency_ms)
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    body = json.dumps(payload).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start = time.perf_counter()

    try:
        with urlopen(request, timeout=300) as response:
            response_data = response.read()

    except HTTPError as exc:
        raise RuntimeError(
            f"Ollama HTTP error: {exc.code} {exc.reason}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama at "
            f"{OLLAMA_URL}. Make sure Ollama is running."
        ) from exc

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    result = json.loads(
        response_data.decode("utf-8")
    )

    answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return answer, latency_ms


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(
    query: str,
    context: str,
) -> str:
    return f"""
You are a helpful technical assistant.

Answer the user's question using ONLY the supplied context.

If the context does not contain enough information,
say that clearly instead of inventing facts.

User question:
{query}

Context:
{context}

Answer:
""".strip()


# ============================================================
# SIMPLE CONTEXT METRICS
# ============================================================

def count_words(text: str) -> int:
    """
    Simple deterministic token proxy.

    This is intentionally called word count, not model token count.
    """

    return len(text.split())


def calculate_reduction(
    raw_count: int,
    optimized_count: int,
) -> float:

    if raw_count == 0:
        return 0.0

    return (
        (raw_count - optimized_count)
        / raw_count
    ) * 100.0


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_separator():
    print("-" * 70)


def print_section(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN REAL-WORLD TEST
# ============================================================

def main():

    print("=" * 70)
    print("NOLQERA + OLLAMA REAL-WORLD VALIDATION")
    print("=" * 70)

    print()
    print(f"Model       : {OLLAMA_MODEL}")
    print(f"Ollama URL  : {OLLAMA_URL}")

    # --------------------------------------------------------
    # 1. Build NOLQERA
    # --------------------------------------------------------

    print_section("1. BUILDING NOLQERA")

    print("Creating NOLQERA pipeline...")

    pipeline = build_nolqera_pipeline()

    print("NOLQERA pipeline ready ✓")

    # --------------------------------------------------------
    # 2. Run actual NOLQERA pipeline
    # --------------------------------------------------------

    print_section("2. RUNNING NOLQERA")

    print("Query:")
    print(QUERY)

    print()
    print("Processing real context...")

    start = time.perf_counter()

    result = run_pipeline(
        pipeline=pipeline,
        query=QUERY,
        raw_input=RAW_CONTEXT,
    )

    nolqera_latency_ms = (
        time.perf_counter() - start
    ) * 1000

    optimized_context = (
        result.compressed_context.strip()
    )

    if not optimized_context:
        raise RuntimeError(
            "NOLQERA produced an empty compressed context."
        )

    print("NOLQERA processing complete ✓")

    # --------------------------------------------------------
    # 3. Display raw context
    # --------------------------------------------------------

    print_section("3. RAW CONTEXT")

    print(RAW_CONTEXT)

    # --------------------------------------------------------
    # 4. Display optimized context
    # --------------------------------------------------------

    print_section("4. NOLQERA OPTIMIZED CONTEXT")

    print(optimized_context)

    # --------------------------------------------------------
    # 5. Context metrics
    # --------------------------------------------------------

    raw_words = count_words(
        RAW_CONTEXT
    )

    optimized_words = count_words(
        optimized_context
    )

    reduction = calculate_reduction(
        raw_words,
        optimized_words,
    )

    compression_ratio = (
        raw_words / optimized_words
        if optimized_words
        else 0.0
    )

    print_section("5. NOLQERA METRICS")

    print(
        f"Raw context words       : {raw_words}"
    )

    print(
        f"Optimized context words : {optimized_words}"
    )

    print(
        f"Word reduction          : {reduction:.2f}%"
    )

    print(
        f"Compression ratio       : {compression_ratio:.2f}x"
    )

    print(
        f"NOLQERA latency         : "
        f"{nolqera_latency_ms:.2f} ms"
    )

    print(
        f"Original sentences      : "
        f"{len(result.sentences)}"
    )

    print(
        f"Retrieved/filtered      : "
        f"{result.metadata.filtered_count}"
    )

    print(
        f"Ranked context          : "
        f"{result.metadata.ranked_count}"
    )

    # --------------------------------------------------------
    # 6. RAW → Ollama
    # --------------------------------------------------------

    print_section("6. RAW CONTEXT → QWEN3")

    raw_prompt = build_prompt(
        query=QUERY,
        context=RAW_CONTEXT,
    )

    print("Sending RAW context to Ollama...")

    raw_answer, raw_llm_latency_ms = call_ollama(
        prompt=raw_prompt
    )

    print()
    print("QWEN3 ANSWER — RAW CONTEXT")
    print_separator()
    print(raw_answer)

    # --------------------------------------------------------
    # 7. NOLQERA → Ollama
    # --------------------------------------------------------

    print_section("7. NOLQERA CONTEXT → QWEN3")

    optimized_prompt = build_prompt(
        query=QUERY,
        context=optimized_context,
    )

    print("Sending NOLQERA optimized context to Ollama...")

    optimized_answer, optimized_llm_latency_ms = (
        call_ollama(
            prompt=optimized_prompt
        )
    )

    print()
    print("QWEN3 ANSWER — NOLQERA CONTEXT")
    print_separator()
    print(optimized_answer)

    # --------------------------------------------------------
    # 8. Final comparison
    # --------------------------------------------------------

    print_section("8. FINAL COMPARISON")

    print(
        f"Raw context words       : {raw_words}"
    )

    print(
        f"NOLQERA context words   : {optimized_words}"
    )

    print(
        f"Context reduction       : {reduction:.2f}%"
    )

    print(
        f"Compression ratio       : {compression_ratio:.2f}x"
    )

    print()

    print(
        f"RAW → Qwen latency      : "
        f"{raw_llm_latency_ms:.2f} ms"
    )

    print(
        f"NOLQERA → Qwen latency  : "
        f"{optimized_llm_latency_ms:.2f} ms"
    )

    print()

    print(
        f"NOLQERA processing      : "
        f"{nolqera_latency_ms:.2f} ms"
    )

    print_separator()

    if reduction > 0:
        print(
            "NOLQERA reduced the context size ✓"
        )
    else:
        print(
            "NOLQERA did not reduce the context size."
        )

    if raw_answer and optimized_answer:
        print(
            "Both LLM runs completed successfully ✓"
        )

    print()
    print("=" * 70)
    print("NOLQERA + OLLAMA REAL-WORLD TEST COMPLETE ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()