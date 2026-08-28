from dataclasses import dataclass


@dataclass(frozen=True)
class RankedItem:
    """A value with its ranking score and original position."""

    index: int
    score: float


class RelevanceRanker:
    """Rank items based on relevance scores."""

    def rank(
        self,
        scores: list[float],
    ) -> list[RankedItem]:
        """Return scores ordered from highest to lowest."""

        if not isinstance(scores, list):
            raise TypeError("scores must be a list")

        if not scores:
            raise ValueError("scores cannot be empty")

        if any(
            not isinstance(score, (int, float))
            for score in scores
        ):
            raise TypeError(
                "scores must contain only numeric values"
            )

        if any(
            not 0.0 <= score <= 1.0
            for score in scores
        ):
            raise ValueError(
                "scores must be between 0.0 and 1.0"
            )

        ranked = [
            RankedItem(
                index=index,
                score=float(score),
            )
            for index, score in enumerate(scores)
        ]

        ranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return ranked

    def top_k(
        self,
        scores: list[float],
        k: int,
    ) -> list[RankedItem]:
        """Return the top-k highest scoring items."""

        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError("k must be greater than zero")

        ranked = self.rank(scores)

        return ranked[:k]