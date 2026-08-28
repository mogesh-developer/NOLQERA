from dataclasses import dataclass


@dataclass(frozen=True)
class RelevanceScore:
    """Represent a relevance score and its classification."""

    score: float
    label: str


class RelevanceScorer:
    """Convert similarity scores into relevance classifications."""

    def __init__(
        self,
        relevant_threshold: float = 0.50,
        weak_threshold: float = 0.20,
    ):
        if not 0.0 <= weak_threshold <= 1.0:
            raise ValueError(
                "weak_threshold must be between 0.0 and 1.0."
            )

        if not 0.0 <= relevant_threshold <= 1.0:
            raise ValueError(
                "relevant_threshold must be between 0.0 and 1.0."
            )

        if weak_threshold > relevant_threshold:
            raise ValueError(
                "weak_threshold cannot exceed relevant_threshold."
            )

        self.relevant_threshold = relevant_threshold
        self.weak_threshold = weak_threshold

    def score(self, similarity: float) -> RelevanceScore:
        """Classify a similarity value by relevance."""

        if not isinstance(similarity, (int, float)):
            raise TypeError(
                "similarity must be a numeric value."
            )

        if not 0.0 <= similarity <= 1.0:
            raise ValueError(
                "similarity must be between 0.0 and 1.0."
            )

        if similarity >= self.relevant_threshold:
            label = "relevant"
        elif similarity >= self.weak_threshold:
            label = "weak"
        else:
            label = "irrelevant"

        return RelevanceScore(
            score=float(similarity),
            label=label,
        )