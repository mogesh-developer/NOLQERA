from dataclasses import dataclass


@dataclass(frozen=True)
class RankedKeyphrase:
    """A keyphrase paired with its score and rank."""

    phrase: str
    score: float
    rank: int


class KeyphraseRanker:
    """Rank keyphrases by score."""

    def rank(
        self,
        scores: dict[str, float],
    ) -> list[RankedKeyphrase]:
        """Return keyphrases ordered by descending score."""

        if not isinstance(scores, dict):
            raise TypeError("scores must be a dictionary")

        if not scores:
            raise ValueError("scores cannot be empty")

        for phrase, score in scores.items():
            if not isinstance(phrase, str):
                raise TypeError(
                    "keyphrase names must be strings"
                )

            if not phrase.strip():
                raise ValueError(
                    "keyphrase names cannot be empty"
                )

            if not isinstance(score, (int, float)):
                raise TypeError(
                    "scores must contain numeric values"
                )

            if score < 0 or score > 1:
                raise ValueError(
                    "scores must be between 0 and 1"
                )

        ordered = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            RankedKeyphrase(
                phrase=phrase,
                score=score,
                rank=rank,
            )
            for rank, (phrase, score)
            in enumerate(ordered, start=1)
        ]

    def top_k(
        self,
        scores: dict[str, float],
        k: int,
    ) -> list[RankedKeyphrase]:
        """Return the top-k keyphrases."""

        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k <= 0:
            raise ValueError(
                "k must be greater than zero"
            )

        return self.rank(scores)[:k]

    def remove_overlapping(
        self,
        ranked: list[RankedKeyphrase],
    ) -> list[RankedKeyphrase]:
        """Remove redundant overlapping keyphrases.

        When one phrase is fully contained inside another phrase,
        keep the phrase with the stronger score.
        """

        if not isinstance(ranked, list):
            raise TypeError("ranked must be a list")

        kept: list[RankedKeyphrase] = []

        for candidate in ranked:
            candidate_tokens = candidate.phrase.split()
            candidate_set = set(candidate_tokens)

            redundant = False

            for existing in kept:
                existing_tokens = existing.phrase.split()
                existing_set = set(existing_tokens)

                # Candidate is contained inside an already kept
                # stronger phrase.
                if (
                    len(candidate_tokens) < len(existing_tokens)
                    and candidate_set.issubset(existing_set)
                ):
                    redundant = True
                    break

                # Existing phrase is contained inside the current
                # candidate, and current candidate is weaker.
                if (
                    len(existing_tokens) < len(candidate_tokens)
                    and existing_set.issubset(candidate_set)
                ):
                    redundant = True
                    break

            if not redundant:
                kept.append(candidate)

        return [
            RankedKeyphrase(
                phrase=item.phrase,
                score=item.score,
                rank=index,
            )
            for index, item in enumerate(
                kept,
                start=1,
            )
        ]