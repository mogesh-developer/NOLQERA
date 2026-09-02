import pytest

from nolqera.evaluation.metrics import (
    ReductionMetrics,
    TokenMetrics,
    calculate_reduction_metrics,
)


def test_reduction_metrics_calculation():
    token_metrics = TokenMetrics(
        raw_tokens=1000,
        optimized_tokens=400,
        token_reduction=600,
        reduction_percentage=60.0,
    )

    metrics = calculate_reduction_metrics(token_metrics)

    assert metrics.absolute_reduction == 600
    assert metrics.reduction_percentage == 60.0
    assert metrics.compression_ratio == 2.5


def test_reduction_metrics_no_reduction():
    token_metrics = TokenMetrics(
        raw_tokens=100,
        optimized_tokens=100,
        token_reduction=0,
        reduction_percentage=0.0,
    )

    metrics = calculate_reduction_metrics(token_metrics)

    assert metrics.absolute_reduction == 0
    assert metrics.reduction_percentage == 0.0
    assert metrics.compression_ratio == 1.0


def test_reduction_metrics_partial_reduction():
    token_metrics = TokenMetrics(
        raw_tokens=200,
        optimized_tokens=150,
        token_reduction=50,
        reduction_percentage=25.0,
    )

    metrics = calculate_reduction_metrics(token_metrics)

    assert metrics.absolute_reduction == 50
    assert metrics.reduction_percentage == 25.0
    assert metrics.compression_ratio == pytest.approx(1.3333333333)


def test_reduction_metrics_rejects_zero_raw_tokens():
    token_metrics = TokenMetrics(
        raw_tokens=0,
        optimized_tokens=0,
        token_reduction=0,
        reduction_percentage=0.0,
    )

    with pytest.raises(
        ValueError,
        match="raw token count must be greater than zero",
    ):
        calculate_reduction_metrics(token_metrics)


def test_reduction_metrics_rejects_zero_optimized_tokens():
    token_metrics = TokenMetrics(
        raw_tokens=100,
        optimized_tokens=0,
        token_reduction=100,
        reduction_percentage=100.0,
    )

    with pytest.raises(
        ValueError,
        match="optimized token count must be greater than zero",
    ):
        calculate_reduction_metrics(token_metrics)


def test_reduction_metrics_handles_token_increase():
    token_metrics = TokenMetrics(
        raw_tokens=100,
        optimized_tokens=125,
        token_reduction=-25,
        reduction_percentage=-25.0,
    )

    metrics = calculate_reduction_metrics(token_metrics)

    assert metrics.absolute_reduction == -25
    assert metrics.reduction_percentage == -25.0
    assert metrics.compression_ratio == 0.8


def test_reduction_metrics_is_immutable():
    metrics = ReductionMetrics(
        absolute_reduction=600,
        reduction_percentage=60.0,
        compression_ratio=2.5,
    )

    with pytest.raises(Exception):
        metrics.absolute_reduction = 500