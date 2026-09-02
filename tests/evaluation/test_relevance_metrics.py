import pytest

from nolqera.evaluation.metrics import (
    RelevanceMetrics,
    calculate_relevance_metrics,
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


def test_relevance_metrics_all_information_matched():
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

    metrics = calculate_relevance_metrics(record)

    assert metrics.total_expected == 2
    assert metrics.matched_expected == 2
    assert metrics.relevance_score == 100.0


def test_relevance_metrics_partial_information_matched():
    record = create_record(
        optimized_text=(
            "Python is a programming language."
        ),
        expected_information=[
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
            "Python supports dynamic typing.",
        ],
    )

    metrics = calculate_relevance_metrics(record)

    assert metrics.total_expected == 3
    assert metrics.matched_expected == 1
    assert metrics.relevance_score == pytest.approx(
        33.3333333333
    )


def test_relevance_metrics_no_information_matched():
    record = create_record(
        optimized_text="Java is a programming language.",
        expected_information=[
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
        ],
    )

    metrics = calculate_relevance_metrics(record)

    assert metrics.total_expected == 2
    assert metrics.matched_expected == 0
    assert metrics.relevance_score == 0.0


def test_relevance_metrics_is_case_insensitive():
    record = create_record(
        optimized_text="PYTHON IS A PROGRAMMING LANGUAGE.",
        expected_information=[
            "python is a programming language.",
        ],
    )

    metrics = calculate_relevance_metrics(record)

    assert metrics.matched_expected == 1
    assert metrics.relevance_score == 100.0


def test_relevance_metrics_rejects_empty_expected_information():
    record = create_record(
        optimized_text="Python is a programming language.",
        expected_information=[],
    )

    with pytest.raises(
        ValueError,
        match="expected_information cannot be empty",
    ):
        calculate_relevance_metrics(record)


def test_relevance_metrics_is_immutable():
    metrics = RelevanceMetrics(
        total_expected=3,
        matched_expected=2,
        relevance_score=66.6666666667,
    )

    with pytest.raises(Exception):
        metrics.matched_expected = 3