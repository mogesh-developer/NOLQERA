from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EntityResult:
    """
    Final structured representation of a detected entity.
    """

    text: str
    entity_type: str
    score: float
    start: int
    end: int
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")

        if not self.text.strip():
            raise ValueError("text cannot be empty")

        if not isinstance(self.entity_type, str):
            raise TypeError(
                "entity_type must be a string"
            )

        if not self.entity_type.strip():
            raise ValueError(
                "entity_type cannot be empty"
            )

        if not isinstance(self.score, (int, float)):
            raise TypeError(
                "score must be numeric"
            )

        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(
                "score must be between 0 and 1"
            )

        if not isinstance(self.start, int):
            raise TypeError(
                "start must be an integer"
            )

        if not isinstance(self.end, int):
            raise TypeError(
                "end must be an integer"
            )

        if self.start < 0:
            raise ValueError(
                "start cannot be negative"
            )

        if self.end <= self.start:
            raise ValueError(
                "end must be greater than start"
            )

        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise TypeError(
                    "metadata must be a dictionary"
                )

    @property
    def length(self) -> int:
        """Return entity span length."""

        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""

        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "score": float(self.score),
            "start": self.start,
            "end": self.end,
            "metadata": self.metadata,
        }