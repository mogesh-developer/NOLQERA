from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticSimilarityResult:
    text_a: str
    text_b: str
    score: float

    def __post_init__(self) -> None:

        if not isinstance(self.text_a, str):
            raise TypeError("text_a must be a string")

        if not isinstance(self.text_b, str):
            raise TypeError("text_b must be a string")

        if not self.text_a.strip():
            raise ValueError("text_a cannot be empty")

        if not self.text_b.strip():
            raise ValueError("text_b cannot be empty")

        if not isinstance(
            self.score,
            (int, float),
        ):
            raise TypeError("score must be numeric")

        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                "score must be between 0 and 1"
            )

    def to_dict(self) -> dict:
        return {
            "text_a": self.text_a,
            "text_b": self.text_b,
            "score": float(self.score),
        }