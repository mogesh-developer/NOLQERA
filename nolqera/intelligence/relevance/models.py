from dataclasses import dataclass


@dataclass(frozen=True)
class RelevanceResult:
    """Represent the relevance analysis of one sentence."""

    sentence: str
    score: float
    label: str
    rank: int