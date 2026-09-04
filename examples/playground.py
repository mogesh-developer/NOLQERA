from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from nolqera import (
    PipelineConfig,
    create_engine,
)
from nolqera.intelligence.context_optimization.context_ranking import (
    ContextRanker,
)
from nolqera.intelligence.context_optimization.noise_detection import (
    NoiseDetector,
)
from nolqera.intelligence.entities import (
    EntityEngine,
    HuggingFaceEntityRecognizer,
)
from nolqera.intelligence.importance.engine import ImportanceEngine
from nolqera.intelligence.intent.engine import IntentEngine
from nolqera.intelligence.keyphrase.engine import KeyphraseEngine
from nolqera.intelligence.pipeline.context_compressor import (
    ContextCompressor,
)
from nolqera.intelligence.pipeline.context_ranker import (
    ContextRankingAnalyzer,
)
from nolqera.intelligence.pipeline.noise_remover import NoiseRemover
from nolqera.intelligence.semantic_search.engine import SemanticSearchEngine
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)
from nolqera.tokenization import tokenize_words


import os
import socket

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

socket.setdefaulttimeout(180)

HOST = "127.0.0.1"
PORT = 8000

GLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL = "glm-4.5-flash"
API_TIMEOUT = 180


def build_engine(raw_input: str):
    """Build a default NOLQERA engine for playground usage."""

    lines = [
        line
        for line in raw_input.splitlines()
        if line.strip()
    ]

    embedding_provider = TFIDFEmbeddingProvider()

    if lines:
        embedding_provider.fit(
            [tokenize_words(line) for line in lines]
        )

    semantic_search_engine = SemanticSearchEngine(
        embedding_provider=embedding_provider
    )

    return create_engine(
        semantic_search_engine=semantic_search_engine,
        importance_engine=ImportanceEngine(),
        keyphrase_engine=KeyphraseEngine(),

        # Keep external NER enabled for the playground.
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

        config=PipelineConfig(),
    )


DEFAULT_KNOWLEDGE_BASE = """
Python is a high-level general-purpose programming language known for its readable syntax and large ecosystem of libraries.

One important benefit of Python is developer productivity. Python programs can often be written with fewer lines of code than equivalent programs in lower-level languages. Its readable syntax also makes Python code easier to understand and maintain.

Python has a large ecosystem of third-party packages. Developers can use frameworks such as FastAPI, Django, and Flask for web development, NumPy and pandas for data processing, and PyTorch and TensorFlow for machine learning.

Python is popular in artificial intelligence and data science. Libraries and frameworks allow developers to build machine learning models, process datasets, perform statistical analysis, and create AI applications without implementing every algorithm from scratch.

Another benefit is cross-platform support. Python applications can run on Windows, Linux, and macOS with relatively few changes.

Python is widely used for automation because scripts can interact with files, APIs, databases (PostgreSQL, MySQL, MongoDB, Redis), operating system processes, and external services. This allows repetitive manual tasks to be automated efficiently.

Python also has a large global developer community. Developers can find extensive documentation, tutorials, open-source projects, and community support when solving technical problems.

A limitation of Python is that it is generally slower than compiled languages such as C++ or Rust for CPU-intensive workloads.

Python can also have higher memory usage in some applications. For performance-critical systems, developers may combine Python with optimized native libraries or other programming languages.

Java is another popular programming language used for enterprise applications. Java has strong performance, mature tooling, and a large ecosystem, but its syntax can require more code for some tasks compared with Python.

JavaScript and TypeScript are widely used for web development, especially in browser-based frontend applications (React, Vue, Angular) and Node.js backend services. Python is different because it is commonly used for backend development, automation, data science, machine learning, and scripting.

Python is therefore especially useful when development speed, readability, ecosystem support, automation, and data or AI capabilities are important requirements.

Developers should consider another language when requirements strongly prioritize CPU performance, very low memory usage, or platform-specific capabilities that are better served by another technology.
""".strip()


def ask_direct_glm(query: str, raw_knowledge: str) -> str:
    """
    Send query AND raw un-optimized knowledge base directly to GLM API (baseline).
    This tests GLM reading the raw full context without NOLQERA optimization.
    """
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GLM_API_KEY environment variable is not set. "
            "Please set GLM_API_KEY in your environment or .env file."
        )

    prompt = f"""
You are an AI assistant answering a user's question.

Use the provided raw knowledge base context to answer the question.
Do not invent facts that are not supported by the context.
If the context does not contain enough information, say so clearly.

User question:
{query}

Raw Knowledge Base Context:
{raw_knowledge}

Answer:
""".strip()

    payload = {
        "model": GLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
    }

    request = Request(
        GLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=API_TIMEOUT) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )
    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8")
        except Exception:
            error_body = ""
        raise RuntimeError(
            f"GLM HTTP error {exc.code}: {error_body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to GLM API at {GLM_URL}."
        ) from exc

    try:
        answer = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "GLM returned an unexpected response structure."
        ) from exc

    if not isinstance(answer, str):
        raise RuntimeError(
            "GLM returned an invalid response."
        )

    return answer.strip()


def ask_glm_llm(
    query: str,
    context: str,
) -> str:
    """
    Send NOLQERA's compressed context to the GLM API.
    """

    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GLM_API_KEY environment variable is not set. "
            "Please set GLM_API_KEY in your environment or .env file."
        )

    ctx_text = context.strip() if context and context.strip() else "No specific context provided."

    prompt = f"""
You are an AI assistant answering a user's question.

Use the provided context to answer the question.
Do not invent facts that are not supported by the context.
If the context does not contain enough information, say so clearly.

User question:
{query}

Relevant context:
{ctx_text}

Answer:
""".strip()

    payload = {
        "model": GLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
    }

    request = Request(
        GLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=API_TIMEOUT) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8")
        except Exception:
            error_body = ""

        raise RuntimeError(
            f"GLM HTTP error {exc.code}: {error_body}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            "Could not connect to GLM API. "
            f"Check connection to {GLM_URL}."
        ) from exc

    try:
        answer = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "GLM returned an unexpected response structure."
        ) from exc

    if not isinstance(answer, str):
        raise RuntimeError(
            "GLM returned an invalid response."
        )

    return answer.strip()


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>NOLQERA Playground</title>
</head>

<body style="font-family: sans-serif; margin: 20px;">
    <h1>NOLQERA Playground</h1>
    <p style="color: #555;">Knowledge Source &rarr; NOLQERA Context Optimization &rarr; LLM</p>

    <form method="POST">
        <label><strong>User Query</strong></label><br>
        <input
            type="text"
            name="query"
            style="width: 100%; max-width: 800px; padding: 8px; margin-top: 5px;"
            placeholder="Enter your query..."
            value="{query_val}"
            required
        >

        <br><br>

        <label><strong>Knowledge Base / Document Context</strong></label><br>
        <textarea
            name="raw_input"
            rows="12"
            style="width: 100%; max-width: 800px; padding: 8px; margin-top: 5px; font-family: monospace;"
            required
        >{raw_input_val}</textarea>

        <br><br>

        <button type="submit" style="padding: 10px 20px; background-color: #0066cc; color: white; border: none; border-radius: 4px; font-weight: bold; cursor: pointer;">
            Run NOLQERA Experiment
        </button>
    </form>

    {result}
</body>
</html>
"""


def render_result(
    result,
    direct_llm_answer: str,
    nolqera_llm_answer: str,
) -> str:
    """Render comparison of Direct GLM (Raw Document) vs NOLQERA + GLM (Compressed Document)."""

    payload = {
        "normalized_text": result.normalized_text,
        "sentences": result.sentences,
        "relevance": result.relevance,
        "importance": result.importance,
        "keywords": result.keywords,
        "entities": result.entities,
        "intents": result.intents,
        "filtered_results": result.filtered_results,
        "ranked_context": result.ranked_context,
        "compressed_context": result.compressed_context,
        "metadata": {
            "input_count": result.metadata.input_count,
            "sentence_count": result.metadata.sentence_count,
            "filtered_count": result.metadata.filtered_count,
            "ranked_count": result.metadata.ranked_count,
        },
    }

    return f"""
    <hr style="margin: 30px 0;">

    <h2 style="font-family: sans-serif;">Direct GLM Answer (Raw Document)</h2>
    <div style="border: 1px solid #ccc; border-radius: 6px; padding: 15px; background-color: #fafafa;">
        <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; margin: 0;">{direct_llm_answer}</pre>
    </div>

    <hr style="margin: 30px 0;">

    <h2 style="font-family: sans-serif; color: #0066cc;">NOLQERA + GLM Answer (Optimized Context)</h2>
    <div style="border: 2px solid #0066cc; border-radius: 6px; padding: 15px; background-color: #f0f7ff;">
        <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; margin: 0;">{nolqera_llm_answer}</pre>
    </div>

    <hr style="margin: 30px 0;">

    <h2 style="font-family: sans-serif;">NOLQERA Compressed Context</h2>
    <div style="border: 1px solid #ddd; border-radius: 6px; padding: 15px; background-color: #f5f5f5;">
        <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; margin: 0;">{result.compressed_context}</pre>
    </div>

    <hr style="margin: 30px 0;">

    <details>
        <summary style="font-family: sans-serif; font-weight: bold; cursor: pointer; padding: 5px 0;">View NOLQERA Full Pipeline JSON Result</summary>
        <pre style="background: #272822; color: #f8f8f2; padding: 15px; border-radius: 6px; overflow-x: auto; margin-top: 10px;">
{json.dumps(payload, indent=2, default=str)}
        </pre>
    </details>
    """


class PlaygroundHandler(BaseHTTPRequestHandler):
    """HTTP handler for the NOLQERA playground."""

    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )
        self.end_headers()

        self.wfile.write(
            HTML_PAGE.format(
                query_val="",
                raw_input_val=DEFAULT_KNOWLEDGE_BASE,
                result="",
            ).encode("utf-8")
        )

    def do_POST(self):
        content_length = int(
            self.headers.get("Content-Length", 0)
        )

        body = self.rfile.read(
            content_length
        ).decode("utf-8")

        form = parse_qs(body)

        query = form.get(
            "query",
            [""],
        )[0].strip()

        raw_input = form.get(
            "raw_input",
            [DEFAULT_KNOWLEDGE_BASE],
        )[0].strip()

        if not query:
            raise ValueError("Query string cannot be empty.")

        try:
            # -------------------------------------------------
            # 1. Baseline Experiment: Direct GLM on Raw Document
            # -------------------------------------------------
            direct_llm_answer = ask_direct_glm(query, raw_input)

            # -------------------------------------------------
            # 2. NOLQERA Experiment: Query + Knowledge Base -> NOLQERA -> GLM
            # -------------------------------------------------
            engine = build_engine(raw_input)

            result = engine.process(
                query=query,
                raw_input=raw_input,
            )

            context = result.compressed_context

            nolqera_llm_answer = ask_glm_llm(
                query=query,
                context=context,
            )

            rendered = render_result(
                result=result,
                direct_llm_answer=direct_llm_answer,
                nolqera_llm_answer=nolqera_llm_answer,
            )

        except Exception as exc:
            rendered = f"""
            <hr>

            <h2>Error</h2>

            <pre>{exc}</pre>
            """

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )
        self.end_headers()

        self.wfile.write(
            HTML_PAGE.format(
                query_val=query,
                raw_input_val=raw_input,
                result=rendered,
            ).encode("utf-8")
        )


def main() -> None:
    server = HTTPServer(
        (HOST, PORT),
        PlaygroundHandler,
    )

    print(
        f"NOLQERA Playground running at "
        f"http://{HOST}:{PORT}"
    )

    print(
        f"LLM: GLM API / {GLM_MODEL}"
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print(
            "\nStopping NOLQERA Playground..."
        )

    finally:
        server.server_close()


if __name__ == "__main__":
    main()