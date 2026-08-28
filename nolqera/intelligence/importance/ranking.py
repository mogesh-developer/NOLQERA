from dataclasses import dataclass


@dataclass(frozen=True)
class RankedImportance:
    """A sentence index paired with its importance score."""

    index: int
    score: float
    rank: int


class ImportanceRanker:
    """Rank sentences by importance score."""

    def rank(
        self,
        scores: list[float],
    ) -> list[RankedImportance]:
        """Return scores ordered from highest to lowest."""

        if not isinstance(scores, list):
            raise TypeError("scores must be a list")

        if not scores:
            raise ValueError("scores cannot be empty")

        for score in scores:
            if not isinstance(score, (int, float)):
                raise TypeError(
                    "scores must contain only numeric values"
                )

            if score < 0 or score > 1:
                raise ValueError(
                    "scores must be between 0 and 1"
                )

        ordered = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            RankedImportance(
                index=index,
                score=score,
                rank=rank,
            )
            for rank, (index, score) in enumerate(
                ordered,
                start=1,
            )
        ]

    def top_k(
        self,
        scores: list[float],
        k: int,
    ) -> list[RankedImportance]:
        """Return the top-k most important sentences."""

        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError("k must be greater than zero")

        ranked = self.rank(scores)

        return ranked[:k]