from dataclasses import dataclass

from .metrics import (
    ContextQualityMetrics,
    InformationLossMetrics,
    LatencyMetrics,
    ReductionMetrics,
    RelevanceMetrics,
    RetentionMetrics,
    TokenMetrics,
    calculate_context_quality,
    calculate_information_loss,
    calculate_reduction_metrics,
    calculate_relevance_metrics,
    calculate_retention_metrics,
    calculate_token_metrics,
    measure_latency,
)
from .models import EvaluationRecord


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Complete evaluation result for one benchmark sample.
    """

    query: str

    token_metrics: TokenMetrics
    reduction_metrics: ReductionMetrics
    relevance_metrics: RelevanceMetrics
    retention_metrics: RetentionMetrics
    information_loss_metrics: InformationLossMetrics
    context_quality_metrics: ContextQualityMetrics
    latency_metrics: LatencyMetrics


def run_benchmark(
    record: EvaluationRecord,
) -> BenchmarkResult:
    """
    Evaluate a single raw-vs-optimized context pair.
    """

    token_metrics = calculate_token_metrics(record)

    reduction_metrics = calculate_reduction_metrics(
        token_metrics
    )

    relevance_metrics = calculate_relevance_metrics(
        record
    )

    retention_metrics = calculate_retention_metrics(
        record
    )

    information_loss_metrics = calculate_information_loss(
        retention_metrics
    )

    context_quality_metrics = calculate_context_quality(
        retention_metrics,
        relevance_metrics,
    )

    latency_metrics = measure_latency(
        lambda: (
            calculate_token_metrics(record),
            calculate_reduction_metrics(token_metrics),
            calculate_relevance_metrics(record),
            calculate_retention_metrics(record),
            calculate_information_loss(retention_metrics),
            calculate_context_quality(
                retention_metrics,
                relevance_metrics,
            ),
        )
    )

    return BenchmarkResult(
        query=record.query,
        token_metrics=token_metrics,
        reduction_metrics=reduction_metrics,
        relevance_metrics=relevance_metrics,
        retention_metrics=retention_metrics,
        information_loss_metrics=information_loss_metrics,
        context_quality_metrics=context_quality_metrics,
        latency_metrics=latency_metrics,
    )