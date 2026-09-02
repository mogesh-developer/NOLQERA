import pytest

from nolqera.evaluation.metrics import (
    RetentionMetrics,
    calculate_retention_metrics,
)
from nolqera.evaluation.models import (
    EvaluationContext,
    EvaluationRecord,
)


def create_record(
    optimized_text: str,
    expected_information: list[str],
) -> EvaluationRecord:
    return EvaluationRecord(
        query="What is Python?",
        raw_context=EvaluationContext(
            text="Raw context",
            token_count=100,
        ),
        optimized_context=EvaluationContext(
            text=optimized_text,
            token_count=40,
        ),
        expected_information=expected_information,
    )


def test_retention_metrics_all_information_retained():
    record = create_record(
        optimized_text=(
            "Python is a programming language. "
            "Python was created by Guido van Rossum."
        ),
        expected_information=[
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
        ],
    )

    metrics = calculate_retention_metrics(record)

    assert metrics.total_information == 2
    assert metrics.retained_information == 2
    assert metrics.retention_percentage == 100.0


def test_retention_metrics_partial_information_retained():
    record = create_record(
        optimized_text=(
            "Python is a programming language."
        ),
        expected_information=[
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
            "Python supports dynamic typing.",
            "Python uses indentation.",
        ],
    )

    metrics = calculate_retention_metrics(record)

    assert metrics.total_information == 4
    assert metrics.retained_information == 1
    assert metrics.retention_percentage == 25.0


def test_retention_metrics_no_information_retained():
    record = create_record(
        optimized_text="Java is a programming language.",
        expected_information=[
            "Python is a programming language.",
            "Python supports dynamic typing.",
        ],
    )

    metrics = calculate_retention_metrics(record)

    assert metrics.total_information == 2
    assert metrics.retained_information == 0
    assert metrics.retention_percentage == 0.0


def test_retention_metrics_is_case_insensitive():
    record = create_record(
        optimized_text="PYTHON IS A PROGRAMMING LANGUAGE.",
        expected_information=[
            "python is a programming language.",
        ],
    )

    metrics = calculate_retention_metrics(record)

    assert metrics.retained_information == 1
    assert metrics.retention_percentage == 100.0


def test_retention_metrics_rejects_empty_information():
    record = create_record(
        optimized_text="Python is a programming language.",
        expected_information=[],
    )

    with pytest.raises(
        ValueError,
        match="expected_information cannot be empty",
    ):
        calculate_retention_metrics(record)


def test_retention_metrics_is_immutable():
    metrics = RetentionMetrics(
        total_information=10,
        retained_information=9,
        retention_percentage=90.0,
    )

    with pytest.raises(Exception):
        metrics.retained_information = 10