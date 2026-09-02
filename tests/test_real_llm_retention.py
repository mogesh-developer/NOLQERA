import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv(
    "NOLQERA_OLLAMA_MODEL",
    "qwen3:4b",
)


QUERY = (
    "What are the main benefits and limitations of Python, "
    "and when should another language be considered?"
)


RAW_CONTEXT = """
Python is a high-level general-purpose programming language
known for readable syntax and a large ecosystem.

Python provides strong developer productivity because programs
can often be written with fewer lines of code.

Python has a large ecosystem containing libraries and frameworks
such as FastAPI, Django, NumPy, pandas, PyTorch, and TensorFlow.

Python is widely used for artificial intelligence, machine learning,
data science, automation, and scientific computing.

Python supports Windows, Linux, and macOS.

Python has a large developer community with documentation,
tutorials, open-source projects, and community support.

A limitation of Python is that it is generally slower than
compiled languages such as C++ or Rust for CPU-intensive workloads.

Python can also use more memory in some applications.

Developers should consider another language when they need very
high CPU performance, very low memory usage, or platform-specific
capabilities better served by another technology.
""".strip()


REQUIRED_FACTS = [
    "developer productivity",
    "large ecosystem",
    "artificial intelligence",
    "machine learning",
    "data science",
    "automation",
    "windows",
    "linux",
    "macos",
    "developer community",
    "slower",
    "cpu-intensive",
    "memory",
    "c++",
    "rust",
]


REQUIRED_ENTITIES = [
    "Python",
    "FastAPI",
    "Django",
    "NumPy",
    "pandas",
    "PyTorch",
    "TensorFlow",
    "C++",
    "Rust",
]


REQUIRED_SECTIONS = [
    "benefits",
    "limitations",
    "another language",
]


def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def call_ollama(prompt: str) -> tuple[str, float]:

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start = time.perf_counter()

    try:
        with urlopen(
            request,
            timeout=300,
        ) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as exc:

        if exc.code in (404, 405):
            raise RuntimeError(
                "Ollama endpoint/model unavailable."
            ) from exc

        raise RuntimeError(
            f"Ollama HTTP error: "
            f"{exc.code} {exc.reason}"
        ) from exc

    except URLError as exc:

        raise RuntimeError(
            "Ollama is not running at "
            f"{OLLAMA_URL}"
        ) from exc

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    answer = (
        data.get("response", "")
        .strip()
    )

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty answer."
        )

    return answer, latency_ms


def build_prompt(context: str) -> str:

    return f"""
You are a technical assistant.

Answer the question using ONLY the supplied context.

Your answer MUST clearly cover:

1. Main benefits of Python
2. Main limitations of Python
3. When another language may be preferable

Preserve concrete technical facts and technology names.

Do not invent facts that are not present in the context.

Question:
{QUERY}

Context:
{context}

Answer:
""".strip()


def evaluate_facts(answer: str) -> dict:

    normalized = normalize(answer)

    return {
        fact: fact.lower() in normalized
        for fact in REQUIRED_FACTS
    }


def evaluate_entities(answer: str) -> dict:

    normalized = normalize(answer)

    return {
        entity: entity.lower() in normalized
        for entity in REQUIRED_ENTITIES
    }


def evaluate_sections(answer: str) -> dict:

    normalized = normalize(answer)

    return {
        section: section in normalized
        for section in REQUIRED_SECTIONS
    }


def coverage(results: dict) -> float:

    if not results:
        return 0.0

    return (
        sum(results.values())
        / len(results)
    ) * 100.0


@pytest.fixture(scope="module")
def llm_answers():

    raw_answer, raw_latency = call_ollama(
        build_prompt(RAW_CONTEXT)
    )

    # Import NOLQERA only after the test has
    # confirmed Ollama is available.
    from nolqera import (
        PipelineConfig,
        create_default_configured_pipeline,
        run_pipeline,
    )

    from nolqera.intelligence.entities.engine import (
        EntityEngine,
    )
    from nolqera.intelligence.importance.engine import (
        ImportanceEngine,
    )
    from nolqera.intelligence.intent.engine import (
        IntentEngine,
    )
    from nolqera.intelligence.keyphrase.engine import (
        KeyphraseEngine,
    )
    from nolqera.intelligence.context_optimization.context_ranking import (
        ContextRanker,
    )
    from nolqera.intelligence.context_optimization.noise_detection import (
        NoiseDetector,
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
    from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
        TFIDFEmbeddingProvider,
    )
    from nolqera.tokenization import (
        tokenize_words,
    )

    embedding_provider = TFIDFEmbeddingProvider()
    embedding_provider.fit([
        tokenize_words(line)
        for line in RAW_CONTEXT.splitlines()
        if line.strip()
    ])

    semantic_search_engine = SemanticSearchEngine(
        embedding_provider=embedding_provider
    )

    pipeline = create_default_configured_pipeline(
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
        config=PipelineConfig(max_sentences=10),
    )

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

    assert optimized_context

    optimized_answer, optimized_latency = (
        call_ollama(
            build_prompt(
                optimized_context
            )
        )
    )

    return {
        "raw_answer": raw_answer,
        "optimized_answer": optimized_answer,
        "optimized_context": optimized_context,
        "raw_latency": raw_latency,
        "optimized_latency": optimized_latency,
        "nolqera_latency": nolqera_latency,
    }


def test_raw_llm_answer_is_non_empty(llm_answers):

    assert llm_answers["raw_answer"].strip()


def test_nolqera_llm_answer_is_non_empty(llm_answers):

    assert llm_answers["optimized_answer"].strip()


def test_raw_answer_has_reasonable_fact_coverage(
    llm_answers,
):

    results = evaluate_facts(
        llm_answers["raw_answer"]
    )

    score = coverage(results)

    print(
        f"\nRAW fact coverage: {score:.2f}%"
    )

    assert score >= 60.0


def test_nolqera_answer_preserves_detailed_facts(
    llm_answers,
):

    results = evaluate_facts(
        llm_answers["optimized_answer"]
    )

    score = coverage(results)

    print(
        f"\nNOLQERA fact coverage: "
        f"{score:.2f}%"
    )

    assert score >= 60.0


def test_nolqera_answer_preserves_entities(
    llm_answers,
):

    results = evaluate_entities(
        llm_answers["optimized_answer"]
    )

    score = coverage(results)

    print(
        f"\nNOLQERA entity coverage: "
        f"{score:.2f}%"
    )

    assert score >= 55.0


def test_nolqera_answer_is_complete(
    llm_answers,
):

    results = evaluate_sections(
        llm_answers["optimized_answer"]
    )

    score = coverage(results)

    print(
        f"\nNOLQERA answer completeness: "
        f"{score:.2f}%"
    )

    assert score >= 66.0


def test_nolqera_context_is_not_empty(
    llm_answers,
):

    assert len(
        llm_answers["optimized_context"]
    ) > 0


def test_nolqera_context_is_smaller_than_raw(
    llm_answers,
):

    raw_words = len(
        RAW_CONTEXT.split()
    )

    optimized_words = len(
        llm_answers[
            "optimized_context"
        ].split()
    )

    print(
        f"\nRAW words: {raw_words}"
    )

    print(
        f"NOLQERA words: "
        f"{optimized_words}"
    )

    assert optimized_words <= raw_words


def test_llm_latency_is_measured(
    llm_answers,
):

    assert llm_answers["raw_latency"] > 0
    assert llm_answers["optimized_latency"] > 0
    assert llm_answers["nolqera_latency"] > 0