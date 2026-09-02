from dataclasses import dataclass

from .benchmark import BenchmarkResult


@dataclass(frozen=True)
class EvaluationReport:
    """
    Human-readable evaluation report for one benchmark result.
    """

    query: str

    raw_tokens: int
    optimized_tokens: int

    token_reduction: int
    reduction_percentage: float
    compression_ratio: float

    relevance_score: float
    retention_percentage: float

    information_loss_percentage: float

    context_quality_score: float

    latency_ms: float


def generate_evaluation_report(
    result: BenchmarkResult,
) -> EvaluationReport:
    """
    Convert a benchmark result into a structured evaluation report.
    """

    return EvaluationReport(
        query=result.query,

        raw_tokens=result.token_metrics.raw_tokens,
        optimized_tokens=result.token_metrics.optimized_tokens,

        token_reduction=result.token_metrics.token_reduction,
        reduction_percentage=(
            result.reduction_metrics.reduction_percentage
        ),
        compression_ratio=(
            result.reduction_metrics.compression_ratio
        ),

        relevance_score=(
            result.relevance_metrics.relevance_score
        ),
        retention_percentage=(
            result.retention_metrics.retention_percentage
        ),

        information_loss_percentage=(
            result.information_loss_metrics.loss_percentage
        ),

        context_quality_score=(
            result.context_quality_metrics.quality_score
        ),

        latency_ms=result.latency_metrics.latency_ms,
    )