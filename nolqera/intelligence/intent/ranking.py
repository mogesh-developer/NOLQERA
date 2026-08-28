from .scorer import IntentScore


class IntentRanker:

    def rank(
        self,
        scores: list[IntentScore],
    ) -> list[IntentScore]:

        if not isinstance(scores, list):
            raise TypeError(
                "scores must be a list"
            )

        if not scores:
            raise ValueError(
                "scores cannot be empty"
            )

        for score in scores:
            if not isinstance(
                score,
                IntentScore,
            ):
                raise TypeError(
                    "all scores must be IntentScore"
                )

            if not 0.0 <= score.score <= 1.0:
                raise ValueError(
                    "intent score must be between 0 and 1"
                )

        return sorted(
            scores,
            key=lambda item: item.score,
            reverse=True,
        )

    def top_k(
        self,
        scores: list[IntentScore],
        k: int,
    ) -> list[IntentScore]:

        if not isinstance(k, int):
            raise TypeError(
                "k must be an integer"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero"
            )

        ranked = self.rank(scores)

        return ranked[:k]