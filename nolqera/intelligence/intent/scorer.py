from dataclasses import dataclass

from .classifier import IntentClassification


@dataclass(frozen=True)
class IntentScore:
    intent: str
    score: float
    evidence_count: int


class IntentScorer:

    def score(
        self,
        classifications: list[IntentClassification],
    ) -> list[IntentScore]:

        if not isinstance(classifications, list):
            raise TypeError(
                "classifications must be a list"
            )

        if not classifications:
            return []

        grouped: dict[str, list[float]] = {}

        for classification in classifications:

            if not isinstance(
                classification,
                IntentClassification,
            ):
                raise TypeError(
                    "all classifications must be "
                    "IntentClassification"
                )

            grouped.setdefault(
                classification.intent,
                [],
            ).append(
                float(classification.score)
            )

        results: list[IntentScore] = []

        for intent, scores in grouped.items():

            # Multiple independent signals should
            # strengthen confidence, but the final
            # score must remain within [0, 1].
            combined_score = 1.0

            for score in scores:
                combined_score *= (1.0 - score)

            combined_score = 1.0 - combined_score

            results.append(
                IntentScore(
                    intent=intent,
                    score=min(
                        1.0,
                        combined_score,
                    ),
                    evidence_count=len(scores),
                )
            )

        return results