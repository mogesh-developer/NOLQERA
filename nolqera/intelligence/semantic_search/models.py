from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticSearchResult:
    text: str
    score: float
    index: int

    def __post_init__(self) -> None:

        if not isinstance(self.text, str):
            raise TypeError(
                "text must be a string"
            )

        if not self.text.strip():
            raise ValueError(
                "text cannot be empty"
            )

        if not isinstance(
            self.score,
            (int, float),
        ):
            raise TypeError(
                "score must be numeric"
            )

        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                "score must be between 0 and 1"
            )

        if not isinstance(self.index, int):
            raise TypeError(
                "index must be an integer"
            )

        if self.index < 0:
            raise ValueError(
                "index cannot be negative"
            )

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": float(self.score),
            "index": self.index,
        }