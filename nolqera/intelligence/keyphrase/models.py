from dataclasses import dataclass


@dataclass(frozen=True)
class KeyphraseResult:
    """Store the final analysis of a keyphrase."""

    phrase: str
    score: float
    rank: int