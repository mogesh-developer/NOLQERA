class KeyphraseScorer:
    """Score keyphrase candidates using classical NLP signals."""

    def __init__(
        self,
        tfidf_weight: float = 0.6,
        frequency_weight: float = 0.3,
        length_weight: float = 0.1,
    ):
        weights = (
            tfidf_weight,
            frequency_weight,
            length_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError("Weights cannot be negative.")

        total = sum(weights)

        if total <= 0:
            raise ValueError(
                "At least one weight must be greater than zero."
            )

        self.tfidf_weight = tfidf_weight / total
        self.frequency_weight = frequency_weight / total
        self.length_weight = length_weight / total

    def score(
        self,
        tfidf_score: float,
        frequency_score: float,
        length_score: float,
    ) -> float:
        """Combine keyphrase signals into one score."""

        scores = (
            tfidf_score,
            frequency_score,
            length_score,
        )

        if any(
            not isinstance(score, (int, float))
            for score in scores
        ):
            raise TypeError(
                "All scores must be numeric."
            )

        if any(
            score < 0 or score > 1
            for score in scores
        ):
            raise ValueError(
                "All scores must be between 0 and 1."
            )

        return (
            self.tfidf_weight * tfidf_score
            + self.frequency_weight * frequency_score
            + self.length_weight * length_score
        )