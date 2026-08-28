from dataclasses import dataclass


@dataclass(frozen=True)
class ImportanceResult:
    """Store the complete importance analysis of a sentence."""

    sentence: str
    score: float
    rank: int