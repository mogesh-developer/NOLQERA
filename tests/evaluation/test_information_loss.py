import pytest

from nolqera.evaluation.metrics import (
    InformationLossMetrics,
    RetentionMetrics,
    calculate_information_loss,
)


def test_information_loss_calculation():
    retention_metrics = RetentionMetrics(
        total_information=10,
        retained_information=9,
        retention_percentage=90.0,
    )

    metrics = calculate_information_loss(
        retention_metrics
    )

    assert metrics.total_information == 10
    assert metrics.retained_information == 9
    assert metrics.lost_information == 1
    assert metrics.loss_percentage == 10.0


def test_information_loss_zero():
    retention_metrics = RetentionMetrics(
        total_information=10,
        retained_information=10,
        retention_percentage=100.0,
    )

    metrics = calculate_information_loss(
        retention_metrics
    )

    assert metrics.lost_information == 0
    assert metrics.loss_percentage == 0.0


def test_information_loss_complete():
    retention_metrics = RetentionMetrics(
        total_information=10,
        retained_information=0,
        retention_percentage=0.0,
    )

    metrics = calculate_information_loss(
        retention_metrics
    )

    assert metrics.lost_information == 10
    assert metrics.loss_percentage == 100.0


def test_information_loss_rejects_zero_total_information():
    retention_metrics = RetentionMetrics(
        total_information=0,
        retained_information=0,
        retention_percentage=0.0,
    )

    with pytest.raises(
        ValueError,
        match="total information must be greater than zero",
    ):
        calculate_information_loss(retention_metrics)


def test_information_loss_rejects_negative_retained_information():
    retention_metrics = RetentionMetrics(
        total_information=10,
        retained_information=-1,
        retention_percentage=-10.0,
    )

    with pytest.raises(
        ValueError,
        match="retained information cannot be negative",
    ):
        calculate_information_loss(retention_metrics)


def test_information_loss_rejects_retained_over_total():
    retention_metrics = RetentionMetrics(
        total_information=10,
        retained_information=11,
        retention_percentage=110.0,
    )

    with pytest.raises(
        ValueError,
        match="retained information cannot exceed total information",
    ):
        calculate_information_loss(retention_metrics)


def test_information_loss_is_immutable():
    metrics = InformationLossMetrics(
        total_information=10,
        retained_information=9,
        lost_information=1,
        loss_percentage=10.0,
    )

    with pytest.raises(Exception):
        metrics.lost_information = 0