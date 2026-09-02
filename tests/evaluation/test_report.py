from nolqera.evaluation.benchmark import run_benchmark
from nolqera.evaluation.models import (
    EvaluationContext,
    EvaluationRecord,
)
from nolqera.evaluation.report import (
    EvaluationReport,
    generate_evaluation_report,
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
            token_count=100,
        ),
        optimized_context=EvaluationContext(
            text=(
                "Python is a programming language. "
                "Python was created by Guido van Rossum."
            ),
            token_count=40,
        ),
        expected_information=[
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
            "Python supports dynamic typing.",
        ],
    )


def test_evaluation_report_creation():
    result = run_benchmark(create_record())

    report = generate_evaluation_report(result)

    assert isinstance(report, EvaluationReport)
    assert report.query == "What is Python?"


def test_report_token_metrics():
    result = run_benchmark(create_record())
    report = generate_evaluation_report(result)

    assert report.raw_tokens == 100
    assert report.optimized_tokens == 40
    assert report.token_reduction == 60


def test_report_reduction_metrics():
    result = run_benchmark(create_record())
    report = generate_evaluation_report(result)

    assert report.reduction_percentage == 60.0
    assert report.compression_ratio == 2.5


def test_report_quality_metrics():
    result = run_benchmark(create_record())
    report = generate_evaluation_report(result)

    assert report.relevance_score == (
        2 / 3 * 100
    )

    assert report.retention_percentage == (
        2 / 3 * 100
    )

    assert report.information_loss_percentage == (
        1 / 3 * 100
    )


def test_report_context_quality():
    result = run_benchmark(create_record())
    report = generate_evaluation_report(result)

    expected_quality = (
        (2 / 3 * 100)
        * (2 / 3 * 100)
        / 100
    )

    assert report.context_quality_score == (
        expected_quality
    )


def test_report_latency():
    result = run_benchmark(create_record())
    report = generate_evaluation_report(result)

    assert report.latency_ms >= 0


def test_report_is_immutable():
    result = run_benchmark(create_record())
    report = generate_evaluation_report(result)

    try:
        report.query = "Modified"
        assert False
    except Exception:
        pass