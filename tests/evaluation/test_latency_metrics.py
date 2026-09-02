import time

import pytest

from nolqera.evaluation.metrics import (
    LatencyMetrics,
    measure_latency,
)


def test_latency_measurement():
    def operation():
        pass

    metrics = measure_latency(operation)

    assert isinstance(metrics, LatencyMetrics)
    assert metrics.latency_ms >= 0


def test_latency_measurement_with_delay():
    def operation():
        time.sleep(0.01)

    metrics = measure_latency(operation)

    assert metrics.latency_ms >= 10


def test_latency_measurement_is_float():
    def operation():
        pass

    metrics = measure_latency(operation)

    assert isinstance(metrics.latency_ms, float)


def test_latency_rejects_non_callable():
    with pytest.raises(
        TypeError,
        match="operation must be callable",
    ):
        measure_latency("not callable")


def test_latency_propagates_operation_exception():
    def operation():
        raise RuntimeError("operation failed")

    with pytest.raises(
        RuntimeError,
        match="operation failed",
    ):
        measure_latency(operation)


def test_latency_metrics_is_immutable():
    metrics = LatencyMetrics(
        latency_ms=10.5,
    )

    with pytest.raises(Exception):
        metrics.latency_ms = 20.0