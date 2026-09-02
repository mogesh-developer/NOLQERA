from __future__ import annotations

import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from nolqera import (
    NOLQERAPipeline,
    PipelineConfig,
    create_default_configured_pipeline,
    run_pipeline,
)
from nolqera.intelligence.context_optimization.context_ranking import ContextRanker
from nolqera.intelligence.context_optimization.noise_detection import NoiseDetector
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
from nolqera.tokenization.tokenizer import Tokenizer


load_dotenv()

GLM_API_KEY = os.getenv("GLM_API_KEY")

if not GLM_API_KEY:
    raise RuntimeError("GLM_API_KEY not found in environment or .env file.")

GLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL = "glm-4.5-flash"


QUERY = """
What are the main benefits and limitations of Python, and when
should a developer consider using another language?
""".strip()


RAW_CONTEXT = """
Python is a high-level general-purpose programming language known
for its readable syntax and large ecosystem of libraries.

One important benefit of Python is developer productivity.
Python programs can often be written with fewer lines of code than
equivalent programs in lower-level languages. Its readable syntax
also makes Python code easier to understand and maintain.

Python has a large ecosystem of third-party packages. Developers
can use frameworks such as FastAPI and Django for web development,
NumPy and pandas for data processing, and PyTorch and TensorFlow
for machine learning.

Python is popular in artificial intelligence and data science.
Libraries and frameworks allow developers to build machine learning
models, process datasets, perform statistical analysis, and create
AI applications without implementing every algorithm from scratch.

Another benefit is cross-platform support. Python applications can
run on Windows, Linux, and macOS with relatively few changes.

Python is widely used for automation because scripts can interact
with files, APIs, databases, operating system processes, and
external services. This allows repetitive manual tasks to be
automated efficiently.

Python also has a large global developer community. Developers can
find extensive documentation, tutorials, open-source projects, and
community support when solving technical problems.

A limitation of Python is that it is generally slower than compiled
languages such as C++ or Rust for CPU-intensive workloads.

Python can also have higher memory usage in some applications.
For performance-critical systems, developers may combine Python with
optimized native libraries or other programming languages.

Java is another popular programming language used for enterprise
applications. Java has strong performance, mature tooling, and a
large ecosystem, but its syntax can require more code for some tasks
compared with Python.

JavaScript is widely used for web development, especially in
browser-based applications. Python is different because it is
commonly used for backend development, automation, data science,
machine learning, and scripting.

Python is therefore especially useful when development speed,
readability, ecosystem support, automation, and data or AI
capabilities are important requirements.

Developers should consider another language when requirements
strongly prioritize CPU performance, very low memory usage, or
platform-specific capabilities that are better served by another
technology.
""".strip()


# ============================================================
# NOLQERA
# ============================================================

def build_nolqera_pipeline() -> NOLQERAPipeline:
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
        config=PipelineConfig(max_sentences=16),
    )


# ============================================================
# GLM FLASH
# ============================================================

def call_glm(prompt: str) -> tuple[str, float]:
    payload = {
        "model": GLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.0,
    }

    request = Request(
        GLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GLM_API_KEY}",
        },
        method="POST",
    )

    start = time.perf_counter()
    max_retries = 3
    data = {}

    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(request, timeout=180) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )
            break
        except (TimeoutError, URLError) as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"GLM Flash API request timed out after {max_retries} attempts: {exc}"
                ) from exc
            time.sleep(2 * attempt)
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 502, 503, 504) and attempt < max_retries:
                time.sleep(2 * attempt)
                continue
            raise RuntimeError(
                f"GLM HTTP error: {exc.code} {exc.reason}\n"
                f"{error_body}"
            ) from exc

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    answer = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not answer:
        raise RuntimeError(
            "GLM Flash returned an empty response."
        )

    return answer, latency_ms


# ============================================================
# PROMPT
# ============================================================

def build_prompt(context: str) -> str:
    return f"""
You are a technical assistant.

Answer the question using ONLY the supplied context.

The answer MUST cover all three areas:
1. Main benefits of Python
2. Main limitations of Python
3. When another language may be preferable

Do not invent information.

Question:
{QUERY}

Context:
{context}

Answer:
""".strip()


# ============================================================
# REQUIRED INFORMATION
# ============================================================

REQUIRED_CONCEPTS = {
    "developer productivity": [
        "developer productivity",
        "development speed",
        "fewer lines",
        "readable syntax",
    ],
    "ecosystem": [
        "large ecosystem",
        "third-party packages",
        "libraries",
        "frameworks",
    ],
    "ai and data science": [
        "artificial intelligence",
        "machine learning",
        "data science",
        "data processing",
    ],
    "automation": [
        "automation",
        "automate",
        "apis",
        "databases",
    ],
    "cross platform": [
        "cross-platform",
        "windows",
        "linux",
        "macos",
    ],
    "community": [
        "developer community",
        "documentation",
        "tutorials",
        "open-source",
    ],
    "performance limitation": [
        "slower",
        "performance",
        "cpu-intensive",
        "compiled languages",
    ],
    "memory limitation": [
        "memory",
        "memory usage",
    ],
    "other language": [
        "another language",
        "c++",
        "rust",
        "java",
        "javascript",
    ],
}


def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    )


def evaluate_concepts(answer: str) -> dict:
    normalized = normalize(answer)

    results = {}

    for concept, phrases in REQUIRED_CONCEPTS.items():
        results[concept] = any(
            phrase in normalized
            for phrase in phrases
        )

    return results


def calculate_retention(results: dict) -> float:
    if not results:
        return 0.0

    return (
        sum(results.values())
        / len(results)
    ) * 100


# ============================================================
# HELPERS
# ============================================================

TOKENIZER = Tokenizer()


def token_count(text: str) -> int:
    return len(
        TOKENIZER.tokenize(
            text,
            lowercase=False,
        )
    )


def reduction_percentage(
    raw_tokens: int,
    optimized_tokens: int,
) -> float:
    if raw_tokens == 0:
        return 0.0

    return (
        (raw_tokens - optimized_tokens)
        / raw_tokens
    ) * 100


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("NOLQERA REAL-WORLD INFORMATION RETENTION TEST (GLM-4.5-FLASH)")
    print("=" * 72)

    print()
    print(f"Model      : {GLM_MODEL}")
    print(f"Question   : {QUERY}")

    # --------------------------------------------------------
    # NOLQERA
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("1. RUNNING NOLQERA")
    print("=" * 72)

    pipeline = build_nolqera_pipeline()

    start = time.perf_counter()

    result = run_pipeline(
        pipeline=pipeline,
        query=QUERY,
        raw_input=RAW_CONTEXT,
    )

    nolqera_latency = (
        time.perf_counter() - start
    ) * 1000

    optimized_context = (
        result.compressed_context.strip()
    )

    if not optimized_context:
        raise RuntimeError(
            "NOLQERA returned empty context."
        )

    print("NOLQERA completed ✓")

    # --------------------------------------------------------
    # DEBUG: RANKED CONTEXT INSPECTION
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("DEBUG — RANKED CONTEXT")
    print("=" * 72)

    print(f"Filtered results : {len(result.filtered_results)}")
    print(f"Ranked context   : {len(result.ranked_context)}")

    for index, item in enumerate(result.ranked_context, start=1):
        print()
        print(f"[RANK {index}]")

        print("Text:")
        print(item.result.text)

        print("Relevance :", item.relevance_score)
        print("Importance:", item.importance_score)
        print("Final     :", item.ranking_score)

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    raw_tokens = token_count(RAW_CONTEXT)
    optimized_tokens = token_count(optimized_context)

    reduction = reduction_percentage(
        raw_tokens,
        optimized_tokens,
    )

    print()
    print("RAW TOKENS       :", raw_tokens)
    print("OPTIMIZED TOKENS :", optimized_tokens)
    print("REDUCTION       :", f"{reduction:.2f}%")
    print("NOLQERA LATENCY :", f"{nolqera_latency:.2f} ms")

    print()
    print("=" * 72)
    print("2. NOLQERA OPTIMIZED CONTEXT")
    print("=" * 72)

    print(optimized_context)

    # --------------------------------------------------------
    # RAW LLM
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("3. RAW CONTEXT → GLM FLASH")
    print("=" * 72)

    raw_answer, raw_latency = call_glm(
        build_prompt(RAW_CONTEXT)
    )

    print(raw_answer)

    # --------------------------------------------------------
    # OPTIMIZED LLM
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("4. NOLQERA CONTEXT → GLM FLASH")
    print("=" * 72)

    optimized_answer, optimized_latency = call_glm(
        build_prompt(optimized_context)
    )

    print(optimized_answer)

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    raw_results = evaluate_concepts(raw_answer)
    optimized_results = evaluate_concepts(
        optimized_answer
    )

    raw_retention = calculate_retention(
        raw_results
    )

    optimized_retention = calculate_retention(
        optimized_results
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("5. INFORMATION RETENTION REPORT")
    print("=" * 72)

    print()
    print(
        f"{'Concept':<28}"
        f"{'RAW':<10}"
        f"{'NOLQERA':<10}"
    )

    print("-" * 48)

    for concept in REQUIRED_CONCEPTS:
        raw_status = (
            "PASS"
            if raw_results[concept]
            else "MISS"
        )

        optimized_status = (
            "PASS"
            if optimized_results[concept]
            else "MISS"
        )

        print(
            f"{concept:<28}"
            f"{raw_status:<10}"
            f"{optimized_status:<10}"
        )

    print()
    print(
        f"RAW answer coverage       : "
        f"{raw_retention:.2f}%"
    )

    print(
        f"NOLQERA answer coverage   : "
        f"{optimized_retention:.2f}%"
    )

    print()
    print(
        f"Raw → GLM latency         : "
        f"{raw_latency / 1000:.2f} sec"
    )

    print(
        f"NOLQERA → GLM latency     : "
        f"{optimized_latency / 1000:.2f} sec"
    )

    print()

    if optimized_retention >= 80:
        print(
            "RESULT: STRONG RETENTION ✓"
        )
    elif optimized_retention >= 60:
        print(
            "RESULT: MODERATE RETENTION ⚠"
        )
    else:
        print(
            "RESULT: LOW RETENTION ✗"
        )

    print()
    print("=" * 72)
    print("RETENTION TEST COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
