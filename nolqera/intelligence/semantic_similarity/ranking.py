from .models import SemanticSimilarityResult


class SemanticSimilarityRanker:

    def rank(
        self,
        results: list[SemanticSimilarityResult],
    ) -> list[SemanticSimilarityResult]:

        if not isinstance(results, list):
            raise TypeError(
                "results must be a list"
            )

        if not results:
            raise ValueError(
                "results cannot be empty"
            )

        for result in results:
            if not isinstance(
                result,
                SemanticSimilarityResult,
            ):
                raise TypeError(
                    "all results must be "
                    "SemanticSimilarityResult"
                )

        return sorted(
            results,
            key=lambda item: item.score,
            reverse=True,
        )

    def top_k(
        self,
        results: list[SemanticSimilarityResult],
        k: int,
    ) -> list[SemanticSimilarityResult]:

        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError(
                "k must be greater than zero"
            )

        return self.rank(results)[:k]