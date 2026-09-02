import pytest

from nolqera.evaluation.metrics import (
    ContextQualityMetrics,
    RelevanceMetrics,
    RetentionMetrics,
    calculate_context_quality,
)


def test_context_quality_calculation():
    retention_metrics = RetentionMetrics(
        total_information=100,
        retained_information=95,
        retention_percentage=95.0,
    )

    relevance_metrics = RelevanceMetrics(
        total_expected=100,
        matched_expected=90,
        relevance_score=90.0,
    )

    metrics = calculate_context_quality(
        retention_metrics,
        relevance_metrics,
    )

    assert metrics.retention_percentage == 95.0
    assert metrics.relevance_score == 90.0
    assert metrics.quality_score == 85.5


def test_context_quality_perfect():
    retention_metrics = RetentionMetrics(
        total_information=10,
        retained_information=10,
        retention_percentage=100.0,
    )

    relevance_metrics = RelevanceMetrics(
        total_expected=10,
        matched_expected=10,
        relevance_score=100.0,
    )

    metrics = calculate_context_quality(
        retention_metrics,
        relevance_metrics,
    )

    assert metrics.quality_score == 100.0


def test_context_quality_zero_retention():
    retention_metrics = RetentionMetrics(
        total_information=10,
        retained_information=0,
        retention_percentage=0.0,
    )

    relevance_metrics = RelevanceMetrics(
        total_expected=10,
        matched_expected=10,
        relevance_score=100.0,
    )

    metrics = calculate_context_quality(
        retention_metrics,
        relevance_metrics,
    )

    assert metrics.quality_score == 0.0


def test_context_quality_zero_relevance():
    retention_metrics = RetentionMetrics(
        total_information=10,
        retained_information=10,
        retention_percentage=100.0,
    )

    relevance_metrics = RelevanceMetrics(
        total_expected=10,
        matched_expected=0,
        relevance_score=0.0,
    )

    metrics = calculate_context_quality(
        retention_metrics,
        relevance_metrics,
    )

    assert metrics.quality_score == 0.0


def test_context_quality_rejects_invalid_retention():
    retention_metrics = RetentionMetrics(
        total_information=10,
        retained_information=10,
        retention_percentage=110.0,
    )

    relevance_metrics = RelevanceMetrics(
        total_expected=10,
        matched_expected=10,
        relevance_score=100.0,
    )

    with pytest.raises(
        ValueError,
        match="retention percentage must be between 0 and 100",
    ):
        calculate_context_quality(
            retention_metrics,
            relevance_metrics,
        )


def test_context_quality_rejects_invalid_relevance():
    retention_metrics = RetentionMetrics(
        total_information=10,
        retained_information=10,
        retention_percentage=100.0,
    )

    relevance_metrics = RelevanceMetrics(
        total_expected=10,
        matched_expected=10,
        relevance_score=110.0,
    )

    with pytest.raises(
        ValueError,
        match="relevance score must be between 0 and 100",
    ):
        calculate_context_quality(
            retention_metrics,
            relevance_metrics,
        )


def test_context_quality_is_immutable():
    metrics = ContextQualityMetrics(
        retention_percentage=95.0,
        relevance_score=90.0,
        quality_score=85.5,
    )

    with pytest.raises(Exception):
        metrics.quality_score = 90.0