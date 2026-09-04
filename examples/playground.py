from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

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
from nolqera.intelligence.pipeline.noise_remover import NoiseRemover
from nolqera.intelligence.semantic_search.engine import SemanticSearchEngine
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)
from nolqera.tokenization import tokenize_words


HOST = "127.0.0.1"
PORT = 8000


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
        entity_engine=EntityEngine(),
        intent_engine=IntentEngine(),
        noise_remover=NoiseRemover(
            NoiseDetector()
        ),
        context_ranker=ContextRankingAnalyzer(
            ContextRanker()
        ),
        context_compressor=ContextCompressor(),
        config=PipelineConfig(),
    )


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>NOLQERA Playground</title>
</head>

<body>
    <h1>NOLQERA Playground</h1>

    <form method="POST">
        <label>Query</label><br>
        <input
            type="text"
            name="query"
            style="width: 500px;"
            required
        >

        <br><br>

        <label>Raw Input</label><br>
        <textarea
            name="raw_input"
            rows="12"
            cols="70"
            required
        ></textarea>

        <br><br>

        <button type="submit">
            Run NOLQERA
        </button>
    </form>

    {result}
</body>
</html>
"""


def render_result(result) -> str:
    """Render a PipelineResult for the initial playground."""

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
    <hr>

    <h2>Pipeline Result</h2>

    <pre>
{json.dumps(payload, indent=2, default=str)}
    </pre>
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
            HTML_PAGE.format(result="").encode("utf-8")
        )

    def do_POST(self):
        content_length = int(
            self.headers.get("Content-Length", 0)
        )

        body = self.rfile.read(content_length).decode(
            "utf-8"
        )

        form = parse_qs(body)

        query = form.get("query", [""])[0]
        raw_input = form.get("raw_input", [""])[0]

        try:
            engine = build_engine(raw_input)

            result = engine.process(
                query=query,
                raw_input=raw_input,
            )

            rendered = render_result(result)

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
                result=rendered
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

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NOLQERA Playground...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()