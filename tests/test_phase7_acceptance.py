"""
Phase 7 Final Acceptance Test Suite for NOLQERA.
"""

from __future__ import annotations

import json
import pytest

from nolqera import (
    PipelineConfig,
    create_default_configured_pipeline,
    run_pipeline,
)
from nolqera.cli import main as cli_main
from nolqera.evaluation.benchmark import run_benchmark
from nolqera.evaluation.models import EvaluationContext, EvaluationRecord
from nolqera.evaluation.report import generate_evaluation_report
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


SAMPLE_CONTEXT = """
NOLQERA is an advanced context optimization engine for AI pipelines.
NOLQERA uses Python 3.11.9 for its execution environment.
The system provides semantic search, multi-signal ranking, and deduplication.
Context compression reduces token usage by target percentage.
FastAPI and PyTorch are integrated framework dependencies.
""".strip()

SAMPLE_QUERY = "What language and version does NOLQERA use?"


def build_test_pipeline(config: PipelineConfig | None = None):
    provider = TFIDFEmbeddingProvider()
    provider.fit([
        tokenize_words(line)
        for line in SAMPLE_CONTEXT.splitlines()
        if line.strip()
    ])
    search_engine = SemanticSearchEngine(embedding_provider=provider)

    return create_default_configured_pipeline(
        semantic_search_engine=search_engine,
        importance_engine=ImportanceEngine(),
        keyphrase_engine=KeyphraseEngine(),
        entity_engine=EntityEngine(),
        intent_engine=IntentEngine(),
        noise_remover=NoiseRemover(NoiseDetector()),
        context_ranker=ContextRankingAnalyzer(ContextRanker()),
        context_compressor=ContextCompressor(),
        config=config or PipelineConfig(max_sentences=3),
    )


def test_phase7_config_serialization():
    config = PipelineConfig(
        keyword_top_k=7,
        max_sentences=4,
        compression_strategy="adaptive",
    )
    json_str = config.to_json()
    reloaded = PipelineConfig.from_json(json_str)

    assert reloaded == config
    assert reloaded.compression_strategy == "adaptive"


def test_phase7_cli_execution(tmp_path, capsys):
    input_file = tmp_path / "raw.txt"
    input_file.write_text(SAMPLE_CONTEXT, encoding="utf-8")

    exit_code = cli_main(
        [
            "--query",
            SAMPLE_QUERY,
            "--input",
            str(input_file),
            "--max-sentences",
            "2",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Python" in captured.out


def test_phase7_evaluation_reporting_export():
    pipeline = build_test_pipeline()
    pipeline_result = run_pipeline(
        pipeline=pipeline,
        query=SAMPLE_QUERY,
        raw_input=SAMPLE_CONTEXT,
    )

    record = EvaluationRecord(
        query=SAMPLE_QUERY,
        raw_context=EvaluationContext(text=SAMPLE_CONTEXT, token_count=100),
        optimized_context=EvaluationContext(text=pipeline_result.compressed_context, token_count=40),
        expected_information=["Python 3.11.9"],
    )

    benchmark_result = run_benchmark(record)

    report = generate_evaluation_report(benchmark_result)

    json_str = report.to_json()
    json_data = json.loads(json_str)
    assert json_data["raw_tokens"] > 0

    csv_str = report.to_csv()
    assert "raw_tokens" in csv_str


def test_phase7_end_to_end_pipeline_acceptance():
    pipeline = build_test_pipeline(
        config=PipelineConfig(
            max_sentences=3,
            compression_strategy="standard",
        )
    )

    result = run_pipeline(
        pipeline=pipeline,
        query=SAMPLE_QUERY,
        raw_input=SAMPLE_CONTEXT,
    )

    assert result.compressed_context.strip()
    assert "Python" in result.compressed_context
