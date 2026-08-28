from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IntentResult:
    """
    Final structured representation of an intent.
    """

    intent: str
    score: float
    evidence_count: int
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:

        if not isinstance(self.intent, str):
            raise TypeError(
                "intent must be a string"
            )

        if not self.intent.strip():
            raise ValueError(
                "intent cannot be empty"
            )

        if not isinstance(
            self.score,
            (int, float),
        ):
            raise TypeError(
                "score must be numeric"
            )

        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(
                "score must be between 0 and 1"
            )

        if not isinstance(
            self.evidence_count,
            int,
        ):
            raise TypeError(
                "evidence_count must be an integer"
            )

        if self.evidence_count < 1:
            raise ValueError(
                "evidence_count must be at least 1"
            )

        if self.metadata is not None:
            if not isinstance(
                self.metadata,
                dict,
            ):
                raise TypeError(
                    "metadata must be a dictionary"
                )

    def to_dict(self) -> dict[str, Any]:
        """
        Return a serializable representation.
        """

        return {
            "intent": self.intent,
            "score": float(self.score),
            "evidence_count": self.evidence_count,
            "metadata": self.metadata,
        }