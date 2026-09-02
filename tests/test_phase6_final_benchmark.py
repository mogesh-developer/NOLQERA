from dataclasses import dataclass

import pytest

from nolqera.evaluation.benchmark import (
    run_benchmark,
)
from nolqera.evaluation.models import (
    EvaluationContext,
    EvaluationRecord,
)
from nolqera.evaluation.report import (
    generate_evaluation_report,
)


@dataclass(frozen=True)
class Phase6Sample:
    name: str
    query: str
    raw: str
    optimized: str
    expected: list[str]
    raw_tokens: int
    optimized_tokens: int


SAMPLES = [
    Phase6Sample(
        name="python benefits",
        query="What are the benefits of Python?",
        raw=(
            "Python has readable syntax. "
            "Python has a large ecosystem. "
            "Python supports automation. "
            "Python is used for machine learning."
        ),
        optimized=(
            "Python has readable syntax. "
            "Python has a large ecosystem. "
            "Python supports automation. "
            "Python is used for machine learning."
        ),
        expected=[
            "Python has readable syntax.",
            "Python has a large ecosystem.",
            "Python supports automation.",
            "Python is used for machine learning.",
        ],
        raw_tokens=100,
        optimized_tokens=100,
    ),

    Phase6Sample(
        name="backend stack",
        query="What backend technologies are used?",
        raw=(
            "NOLQERA uses Python. "
            "NOLQERA uses FastAPI for backend APIs. "
            "NOLQERA uses semantic search. "
            "NOLQERA applies reranking."
        ),
        optimized=(
            "NOLQERA uses Python. "
            "NOLQERA uses FastAPI for backend APIs. "
            "NOLQERA uses semantic search. "
            "NOLQERA applies reranking."
        ),
        expected=[
            "NOLQERA uses Python.",
            "NOLQERA uses FastAPI for backend APIs.",
            "NOLQERA uses semantic search.",
            "NOLQERA applies reranking.",
        ],
        raw_tokens=120,
        optimized_tokens=120,
    ),

    Phase6Sample(
        name="detailed technical facts",
        query="What are the technical configuration facts?",
        raw=(
            "Python version is 3.11.9. "
            "Embedding dimension is 384. "
            "Similarity threshold is 0.82. "
            "Compression target is 40%. "
            "Benchmark size is 128."
        ),
        optimized=(
            "Python version is 3.11.9. "
            "Embedding dimension is 384. "
            "Similarity threshold is 0.82. "
            "Compression target is 40%. "
            "Benchmark size is 128."
        ),
        expected=[
            "Python version is 3.11.9.",
            "Embedding dimension is 384.",
            "Similarity threshold is 0.82.",
            "Compression target is 40%.",
            "Benchmark size is 128.",
        ],
        raw_tokens=100,
        optimized_tokens=100,
    ),

    Phase6Sample(
        name="partial retention",
        query="Which retrieval methods are used?",
        raw=(
            "NOLQERA uses semantic search. "
            "NOLQERA uses keyword retrieval. "
            "NOLQERA uses reranking. "
            "NOLQERA performs deduplication."
        ),
        optimized=(
            "NOLQERA uses semantic search. "
            "NOLQERA uses keyword retrieval. "
            "NOLQERA uses reranking."
        ),
        expected=[
            "NOLQERA uses semantic search.",
            "NOLQERA uses keyword retrieval.",
            "NOLQERA uses reranking.",
            "NOLQERA performs deduplication.",
        ],
        raw_tokens=100,
        optimized_tokens=75,
    ),

    Phase6Sample(
        name="cross platform",
        query="Which platforms are supported?",
        raw=(
            "Python runs on Windows. "
            "Python runs on Linux. "
            "Python runs on macOS."
        ),
        optimized=(
            "Python runs on Windows. "
            "Python runs on Linux. "
            "Python runs on macOS."
        ),
        expected=[
            "Python runs on Windows.",
            "Python runs on Linux.",
            "Python runs on macOS.",
        ],
        raw_tokens=80,
        optimized_tokens=80,
    ),
]


def make_record(
    sample: Phase6Sample,
) -> EvaluationRecord:

    return EvaluationRecord(
        query=sample.query,

        raw_context=EvaluationContext(
            text=sample.raw,
            token_count=sample.raw_tokens,
        ),

        optimized_context=EvaluationContext(
            text=sample.optimized,
            token_count=sample.optimized_tokens,
        ),

        expected_information=sample.expected,
    )


def run_all_benchmarks():

    results = []

    for sample in SAMPLES:

        result = run_benchmark(
            make_record(sample)
        )

        report = generate_evaluation_report(
            result
        )

        results.append(
            (
                sample,
                result,
                report,
            )
        )

    return results


def test_all_phase6_samples_execute():

    results = run_all_benchmarks()

    assert len(results) == len(SAMPLES)

    for sample, result, report in results:

        assert result.query == sample.query
        assert report.query == sample.query


def test_retention_is_correct_for_every_sample():

    results = run_all_benchmarks()

    for sample, result, _ in results:

        expected_retained = sum(
            1
            for item in sample.expected
            if item.lower()
            in sample.optimized.lower()
        )

        assert (
            result.retention_metrics.retained_information
            == expected_retained
        )


def test_full_retention_samples_reach_100_percent():

    results = run_all_benchmarks()

    full_retention_samples = [
        result
        for sample, result, _
        in results
        if sample.name != "partial retention"
    ]

    for result in full_retention_samples:

        assert (
            result.retention_metrics.retention_percentage
            == 100.0
        )


def test_partial_retention_is_detected():

    results = run_all_benchmarks()

    partial = next(
        result
        for sample, result, _
        in results
        if sample.name == "partial retention"
    )

    assert (
        partial.retention_metrics.retention_percentage
        == 75.0
    )

    assert (
        partial.information_loss_metrics.lost_information
        == 1
    )


def test_reduction_is_detected():

    results = run_all_benchmarks()

    partial = next(
        result
        for sample, result, _
        in results
        if sample.name == "partial retention"
    )

    assert (
        partial.reduction_metrics.reduction_percentage
        == 25.0
    )


def test_no_sample_has_invalid_retention():

    results = run_all_benchmarks()

    for _, result, _ in results:

        retention = (
            result.retention_metrics
            .retention_percentage
        )

        assert 0.0 <= retention <= 100.0


def test_final_phase6_average_retention():

    results = run_all_benchmarks()

    retention_scores = [
        result.retention_metrics.retention_percentage
        for _, result, _
        in results
    ]

    average_retention = (
        sum(retention_scores)
        / len(retention_scores)
    )

    print(
        f"\nPhase-6 average retention: "
        f"{average_retention:.2f}%"
    )

    assert average_retention >= 95.0


def test_final_phase6_average_quality():

    results = run_all_benchmarks()

    quality_scores = [
        result.context_quality_metrics.quality_score
        for _, result, _
        in results
    ]

    average_quality = (
        sum(quality_scores)
        / len(quality_scores)
    )

    print(
        f"\nPhase-6 average context quality: "
        f"{average_quality:.2f}%"
    )

    assert average_quality >= 90.0


def test_final_phase6_report_generation():

    results = run_all_benchmarks()

    for _, _, report in results:

        assert report.query
        assert report.raw_tokens > 0
        assert report.optimized_tokens > 0
        assert 0.0 <= report.retention_percentage <= 100.0
        assert (
            0.0
            <= report.information_loss_percentage
            <= 100.0
        )


def test_phase6_final_gate():

    results = run_all_benchmarks()

    average_retention = sum(
        result.retention_metrics.retention_percentage
        for _, result, _
        in results
    ) / len(results)

    average_quality = sum(
        result.context_quality_metrics.quality_score
        for _, result, _
        in results
    ) / len(results)

    assert average_retention >= 95.0
    assert average_quality >= 90.0