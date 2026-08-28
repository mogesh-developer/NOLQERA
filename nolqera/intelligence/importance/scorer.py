class ImportanceScorer:
    """Calculate sentence importance from multiple signals."""

    def __init__(
        self,
        tfidf_weight: float = 0.6,
        position_weight: float = 0.2,
        density_weight: float = 0.2,
    ):
        weights = (
            tfidf_weight,
            position_weight,
            density_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError("Weights cannot be negative.")

        total = sum(weights)

        if total <= 0:
            raise ValueError(
                "At least one weight must be greater than zero."
            )

        self.tfidf_weight = tfidf_weight / total
        self.position_weight = position_weight / total
        self.density_weight = density_weight / total

    def score(
        self,
        tfidf_score: float,
        position_score: float,
        density_score: float,
    ) -> float:
        """Combine importance signals into one score."""

        scores = (
            tfidf_score,
            position_score,
            density_score,
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
            + self.position_weight * position_score
            + self.density_weight * density_score
        )