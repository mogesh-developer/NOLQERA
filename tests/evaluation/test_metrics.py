import pytest

from nolqera.evaluation.metrics import (
    TokenMetrics,
    calculate_token_metrics,
)
from nolqera.evaluation.models import (
    EvaluationContext,
    EvaluationRecord,
)


def create_record(
    raw_tokens: int | None,
    optimized_tokens: int | None,
) -> EvaluationRecord:
    return EvaluationRecord(
        query="What is Python?",
        raw_context=EvaluationContext(
            text="Raw context",
            token_count=raw_tokens,
        ),
        optimized_context=EvaluationContext(
            text="Optimized context",
            token_count=optimized_tokens,
        ),
    )


def test_token_metrics_calculation():
    record = create_record(
        raw_tokens=100,
        optimized_tokens=40,
    )

    metrics = calculate_token_metrics(record)

    assert metrics.raw_tokens == 100
    assert metrics.optimized_tokens == 40
    assert metrics.token_reduction == 60
    assert metrics.reduction_percentage == 60.0


def test_token_metrics_partial_reduction():
    record = create_record(
        raw_tokens=200,
        optimized_tokens=150,
    )

    metrics = calculate_token_metrics(record)

    assert metrics.raw_tokens == 200
    assert metrics.optimized_tokens == 150
    assert metrics.token_reduction == 50
    assert metrics.reduction_percentage == 25.0


def test_token_metrics_no_reduction():
    record = create_record(
        raw_tokens=100,
        optimized_tokens=100,
    )

    metrics = calculate_token_metrics(record)

    assert metrics.token_reduction == 0
    assert metrics.reduction_percentage == 0.0


def test_token_metrics_requires_raw_token_count():
    record = create_record(
        raw_tokens=None,
        optimized_tokens=40,
    )

    with pytest.raises(ValueError, match="raw_context.token_count"):
        calculate_token_metrics(record)


def test_token_metrics_requires_optimized_token_count():
    record = create_record(
        raw_tokens=100,
        optimized_tokens=None,
    )

    with pytest.raises(
        ValueError,
        match="optimized_context.token_count",
    ):
        calculate_token_metrics(record)


def test_token_metrics_rejects_zero_raw_tokens():
    record = create_record(
        raw_tokens=0,
        optimized_tokens=0,
    )

    with pytest.raises(
        ValueError,
        match="raw token count must be greater than zero",
    ):
        calculate_token_metrics(record)


def test_token_metrics_can_detect_token_increase():
    record = create_record(
        raw_tokens=100,
        optimized_tokens=120,
    )

    metrics = calculate_token_metrics(record)

    assert metrics.token_reduction == -20
    assert metrics.reduction_percentage == -20.0


def test_token_metrics_is_immutable():
    metrics = TokenMetrics(
        raw_tokens=100,
        optimized_tokens=40,
        token_reduction=60,
        reduction_percentage=60.0,
    )

    with pytest.raises(Exception):
        metrics.raw_tokens = 200