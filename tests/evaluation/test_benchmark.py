from nolqera.evaluation.benchmark import (
    BenchmarkResult,
    run_benchmark,
)
from nolqera.evaluation.models import (
    EvaluationContext,
    EvaluationRecord,
)


def create_record() -> EvaluationRecord:
    return EvaluationRecord(
        query="What is Python?",
        raw_context=EvaluationContext(
            text=(
                "Python is a programming language. "
                "Python was created by Guido van Rossum. "
                "Python supports dynamic typing."
            ),
            document_ids=[
                "doc-1",
                "doc-2",
                "doc-3",
            ],
            token_count=100,
        ),
        optimized_context=EvaluationContext(
            text=(
                "Python is a programming language. "
                "Python was created by Guido van Rossum."
            ),
            document_ids=[
                "doc-1",
                "doc-2",
            ],
            token_count=40,
        ),
        expected_information=[
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
            "Python supports dynamic typing.",
        ],
    )


def test_benchmark_result_creation():
    record = create_record()

    result = run_benchmark(record)

    assert isinstance(result, BenchmarkResult)
    assert result.query == "What is Python?"


def test_benchmark_token_metrics():
    result = run_benchmark(create_record())

    assert result.token_metrics.raw_tokens == 100
    assert result.token_metrics.optimized_tokens == 40
    assert result.token_metrics.token_reduction == 60
    assert result.token_metrics.reduction_percentage == 60.0


def test_benchmark_reduction_metrics():
    result = run_benchmark(create_record())

    assert result.reduction_metrics.absolute_reduction == 60
    assert result.reduction_metrics.reduction_percentage == 60.0
    assert result.reduction_metrics.compression_ratio == 2.5


def test_benchmark_retention_metrics():
    result = run_benchmark(create_record())

    assert result.retention_metrics.total_information == 3
    assert result.retention_metrics.retained_information == 2
    assert result.retention_metrics.retention_percentage == (
        2 / 3 * 100
    )


def test_benchmark_information_loss():
    result = run_benchmark(create_record())

    assert result.information_loss_metrics.total_information == 3
    assert result.information_loss_metrics.retained_information == 2
    assert result.information_loss_metrics.lost_information == 1


def test_benchmark_relevance():
    result = run_benchmark(create_record())

    assert result.relevance_metrics.total_expected == 3
    assert result.relevance_metrics.matched_expected == 2


def test_benchmark_context_quality():
    result = run_benchmark(create_record())

    expected_quality = (
        (2 / 3 * 100)
        * (2 / 3 * 100)
        / 100
    )

    assert result.context_quality_metrics.quality_score == (
        expected_quality
    )


def test_benchmark_latency():
    result = run_benchmark(create_record())

    assert result.latency_metrics.latency_ms >= 0


def test_benchmark_is_immutable():
    result = run_benchmark(create_record())

    try:
        result.query = "Modified"
        assert False
    except Exception:
        pass