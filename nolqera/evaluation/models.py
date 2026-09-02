from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EvaluationContext:
    """
    Represents a context snapshot used during evaluation.
    """

    text: str
    document_ids: List[str] = field(default_factory=list)
    token_count: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")

        if not isinstance(self.document_ids, list):
            raise TypeError("document_ids must be a list")

        if self.token_count is not None:
            if not isinstance(self.token_count, int):
                raise TypeError("token_count must be an integer")

            if self.token_count < 0:
                raise ValueError("token_count cannot be negative")


@dataclass(frozen=True)
class EvaluationRecord:
    """
    Represents a single NOLQERA evaluation sample.
    """

    query: str
    raw_context: EvaluationContext
    optimized_context: EvaluationContext

    expected_information: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")

        if not self.query.strip():
            raise ValueError("query cannot be empty")

        if not isinstance(self.raw_context, EvaluationContext):
            raise TypeError(
                "raw_context must be an EvaluationContext"
            )

        if not isinstance(self.optimized_context, EvaluationContext):
            raise TypeError(
                "optimized_context must be an EvaluationContext"
            )

        if not isinstance(self.expected_information, list):
            raise TypeError(
                "expected_information must be a list"
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary"
            )