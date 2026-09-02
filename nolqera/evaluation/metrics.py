from dataclasses import dataclass
from time import perf_counter


from .models import EvaluationRecord


@dataclass(frozen=True)
class TokenMetrics:
    """
    Token usage metrics for a single evaluation record.
    """

    raw_tokens: int
    optimized_tokens: int
    token_reduction: int
    reduction_percentage: float


def calculate_token_metrics(
    record: EvaluationRecord,
) -> TokenMetrics:
    """
    Calculate token reduction metrics from an evaluation record.
    """

    raw_tokens = record.raw_context.token_count
    optimized_tokens = record.optimized_context.token_count

    if raw_tokens is None:
        raise ValueError(
            "raw_context.token_count is required"
        )

    if optimized_tokens is None:
        raise ValueError(
            "optimized_context.token_count is required"
        )

    if raw_tokens == 0:
        raise ValueError(
            "raw token count must be greater than zero"
        )

    token_reduction = raw_tokens - optimized_tokens

    reduction_percentage = (
        token_reduction / raw_tokens
    ) * 100

    return TokenMetrics(
        raw_tokens=raw_tokens,
        optimized_tokens=optimized_tokens,
        token_reduction=token_reduction,
        reduction_percentage=reduction_percentage,
    )

@dataclass(frozen=True)
class ReductionMetrics:
    """
    Context reduction statistics derived from token metrics.
    """

    absolute_reduction: int
    reduction_percentage: float
    compression_ratio: float


def calculate_reduction_metrics(
    token_metrics: TokenMetrics,
) -> ReductionMetrics:
    """
    Calculate reduction statistics from token metrics.
    """

    if token_metrics.raw_tokens <= 0:
        raise ValueError(
            "raw token count must be greater than zero"
        )

    if token_metrics.optimized_tokens <= 0:
        raise ValueError(
            "optimized token count must be greater than zero"
        )

    absolute_reduction = (
        token_metrics.raw_tokens
        - token_metrics.optimized_tokens
    )

    reduction_percentage = (
        absolute_reduction
        / token_metrics.raw_tokens
    ) * 100

    compression_ratio = (
        token_metrics.raw_tokens
        / token_metrics.optimized_tokens
    )

    return ReductionMetrics(
        absolute_reduction=absolute_reduction,
        reduction_percentage=reduction_percentage,
        compression_ratio=compression_ratio,
    )

@dataclass(frozen=True)
class RelevanceMetrics:
    """
    Relevance statistics for an optimized context.
    """

    total_expected: int
    matched_expected: int
    relevance_score: float


def calculate_relevance_metrics(
    record: EvaluationRecord,
) -> RelevanceMetrics:
    """
    Calculate how much expected information is present
    in the optimized context.
    """

    expected_information = record.expected_information

    if not expected_information:
        raise ValueError(
            "expected_information cannot be empty"
        )

    context = record.optimized_context.text.lower()

    matched_expected = sum(
        1
        for item in expected_information
        if item.strip().lower() in context
    )

    total_expected = len(expected_information)

    relevance_score = (
        matched_expected / total_expected
    ) * 100

    return RelevanceMetrics(
        total_expected=total_expected,
        matched_expected=matched_expected,
        relevance_score=relevance_score,
    )

@dataclass(frozen=True)
class RetentionMetrics:
    """
    Information retention statistics.
    """

    total_information: int
    retained_information: int
    retention_percentage: float


def calculate_retention_metrics(
    record: EvaluationRecord,
) -> RetentionMetrics:
    """
    Calculate the percentage of expected information
    retained in the optimized context.
    """

    expected_information = record.expected_information

    if not expected_information:
        raise ValueError(
            "expected_information cannot be empty"
        )

    optimized_context = (
        record.optimized_context.text.lower()
    )

    retained_information = sum(
        1
        for item in expected_information
        if item.strip().lower() in optimized_context
    )

    total_information = len(expected_information)

    retention_percentage = (
        retained_information / total_information
    ) * 100

    return RetentionMetrics(
        total_information=total_information,
        retained_information=retained_information,
        retention_percentage=retention_percentage,
    )

@dataclass(frozen=True)
class InformationLossMetrics:
    """
    Information loss statistics.
    """

    total_information: int
    retained_information: int
    lost_information: int
    loss_percentage: float


def calculate_information_loss(
    retention_metrics: RetentionMetrics,
) -> InformationLossMetrics:
    """
    Calculate information lost during context optimization.
    """

    total_information = retention_metrics.total_information
    retained_information = retention_metrics.retained_information

    if total_information <= 0:
        raise ValueError(
            "total information must be greater than zero"
        )

    if retained_information < 0:
        raise ValueError(
            "retained information cannot be negative"
        )

    if retained_information > total_information:
        raise ValueError(
            "retained information cannot exceed total information"
        )

    lost_information = (
        total_information - retained_information
    )

    loss_percentage = (
        lost_information / total_information
    ) * 100

    return InformationLossMetrics(
        total_information=total_information,
        retained_information=retained_information,
        lost_information=lost_information,
        loss_percentage=loss_percentage,
    )

@dataclass(frozen=True)
class LatencyMetrics:
    """
    Execution latency statistics for an evaluation operation.
    """

    latency_ms: float


def measure_latency(operation) -> LatencyMetrics:
    """
    Measure the execution time of a callable in milliseconds.
    """

    if not callable(operation):
        raise TypeError(
            "operation must be callable"
        )

    start = perf_counter()

    operation()

    end = perf_counter()

    latency_ms = (end - start) * 1000

    return LatencyMetrics(
        latency_ms=latency_ms,
    )

@dataclass(frozen=True)
class ContextQualityMetrics:
    """
    Overall context quality statistics.
    """

    retention_percentage: float
    relevance_score: float
    quality_score: float


def calculate_context_quality(
    retention_metrics: RetentionMetrics,
    relevance_metrics: RelevanceMetrics,
) -> ContextQualityMetrics:
    """
    Calculate overall context quality from retention
    and relevance metrics.
    """

    retention_percentage = (
        retention_metrics.retention_percentage
    )

    relevance_score = (
        relevance_metrics.relevance_score
    )

    if not 0 <= retention_percentage <= 100:
        raise ValueError(
            "retention percentage must be between 0 and 100"
        )

    if not 0 <= relevance_score <= 100:
        raise ValueError(
            "relevance score must be between 0 and 100"
        )

    quality_score = (
        retention_percentage
        * relevance_score
    ) / 100

    return ContextQualityMetrics(
        retention_percentage=retention_percentage,
        relevance_score=relevance_score,
        quality_score=quality_score,
    )