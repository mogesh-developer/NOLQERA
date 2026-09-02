"""
CLI runner for NOLQERA context optimization engine.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

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
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)
from nolqera.tokenization import tokenize_words


def build_cli_pipeline(
    raw_context: str,
    config: PipelineConfig | None = None,
) -> NOLQERAPipeline:
    """Build a default CLI processing pipeline."""
    embedding_provider = TFIDFEmbeddingProvider()
    lines = [line for line in raw_context.splitlines() if line.strip()]
    if lines:
        embedding_provider.fit([tokenize_words(line) for line in lines])

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
        config=config or PipelineConfig(),
    )


def main(args: Sequence[str] | None = None) -> int:
    """Main CLI execution entry point."""
    parser = argparse.ArgumentParser(
        prog="nolqera-cli",
        description="NOLQERA Context Optimization CLI Runner",
    )
    parser.add_argument(
        "-q",
        "--query",
        required=True,
        help="Query or topic prompt for context relevance scoring",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        help="Path to raw input text file (reads from stdin if omitted)",
    )
    parser.add_argument(
        "-m",
        "--max-sentences",
        type=int,
        default=3,
        help="Maximum sentence limit for compressed context output",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        help="Path to JSON configuration file",
    )

    parsed = parser.parse_args(args)

    if parsed.input_file:
        try:
            with open(parsed.input_file, "r", encoding="utf-8") as file:
                raw_text = file.read()
        except OSError as exc:
            print(f"Error reading input file: {exc}", file=sys.stderr)
            return 1
    else:
        raw_text = sys.stdin.read()

    if not raw_text.strip():
        print("Error: Empty input context provided.", file=sys.stderr)
        return 1

    if parsed.config_file:
        try:
            with open(parsed.config_file, "r", encoding="utf-8") as file:
                config = PipelineConfig.from_json(file.read())
        except (OSError, ValueError, TypeError) as exc:
            print(f"Error loading configuration file: {exc}", file=sys.stderr)
            return 1
    else:
        config = PipelineConfig(max_sentences=parsed.max_sentences)

    pipeline = build_cli_pipeline(raw_text, config=config)
    result = run_pipeline(
        pipeline=pipeline,
        query=parsed.query,
        raw_input=raw_text,
    )

    print(result.compressed_context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
