from dataclasses import dataclass

from .candidates import IntentCandidate


@dataclass(frozen=True)
class IntentClassification:
    intent: str
    score: float


class IntentClassifier:

    def classify(
        self,
        candidates: list[IntentCandidate],
    ) -> list[IntentClassification]:

        if not isinstance(candidates, list):
            raise TypeError(
                "candidates must be a list"
            )

        if not candidates:
            return []

        results: list[IntentClassification] = []

        for candidate in candidates:

            if not isinstance(
                candidate,
                IntentCandidate,
            ):
                raise TypeError(
                    "all candidates must be IntentCandidate"
                )

            intent = self._map_signal_to_intent(
                candidate.signal
            )

            results.append(
                IntentClassification(
                    intent=intent,
                    score=float(candidate.score),
                )
            )

        return results

    @staticmethod
    def _map_signal_to_intent(
        signal: str,
    ) -> str:

        if not isinstance(signal, str):
            raise TypeError(
                "signal must be a string"
            )

        # Generic signal categories.
        # Final semantic intelligence will
        # be expanded later.
        signal_groups = {
            "question_form": "question",
            "interrogative": "question",
        }

        return signal_groups.get(
            signal,
            "unknown",
        )