from nolqera.evaluation import (
    EvaluationContext,
    EvaluationRecord,
    calculate_reduction_metrics,
    calculate_token_metrics,
)


def main() -> None:
    record = EvaluationRecord(
        query="What is Python?",
        raw_context=EvaluationContext(
            text="Raw retrieved context...",
            document_ids=["doc-1", "doc-2", "doc-3"],
            token_count=1000,
        ),
        optimized_context=EvaluationContext(
            text="Optimized relevant context...",
            document_ids=["doc-1"],
            token_count=400,
        ),
        expected_information=[
            "Python is a programming language",
        ],
        metadata={
            "dataset": "demo",
        },
    )

    token_metrics = calculate_token_metrics(record)

    reduction_metrics = calculate_reduction_metrics(
        token_metrics
    )

    print()
    print("=" * 50)
    print("        NOLQERA EVALUATION RESULT")
    print("=" * 50)

    print(f"Query                 : {record.query}")
    print()

    print("TOKEN METRICS")
    print("-" * 50)
    print(f"Raw tokens            : {token_metrics.raw_tokens}")
    print(
        f"Optimized tokens      : "
        f"{token_metrics.optimized_tokens}"
    )
    print(
        f"Tokens reduced        : "
        f"{token_metrics.token_reduction}"
    )
    print(
        f"Reduction percentage  : "
        f"{token_metrics.reduction_percentage:.2f}%"
    )

    print()
    print("REDUCTION METRICS")
    print("-" * 50)
    print(
        f"Absolute reduction    : "
        f"{reduction_metrics.absolute_reduction}"
    )
    print(
        f"Reduction percentage  : "
        f"{reduction_metrics.reduction_percentage:.2f}%"
    )
    print(
        f"Compression ratio     : "
        f"{reduction_metrics.compression_ratio:.2f}x"
    )

    print()
    print("=" * 50)


if __name__ == "__main__":
    main()