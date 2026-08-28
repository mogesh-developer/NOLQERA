from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticScore:
    score: float
    label: str


class SemanticSimilarityScorer:

    def __init__(
        self,
        high_threshold: float = 0.75,
        medium_threshold: float = 0.45,
    ) -> None:

        if not (
            0.0
            <= medium_threshold
            <= high_threshold
            <= 1.0
        ):
            raise ValueError(
                "thresholds must satisfy "
                "0 <= medium <= high <= 1"
            )

        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def score(
        self,
        similarity: float,
    ) -> SemanticScore:

        if not isinstance(
            similarity,
            (int, float),
        ):
            raise TypeError(
                "similarity must be numeric"
            )

        if not 0.0 <= similarity <= 1.0:
            raise ValueError(
                "similarity must be between 0 and 1"
            )

        if similarity >= self.high_threshold:
            label = "high"

        elif similarity >= self.medium_threshold:
            label = "medium"

        else:
            label = "low"

        return SemanticScore(
            score=float(similarity),
            label=label,
        )