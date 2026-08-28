from dataclasses import dataclass


@dataclass(frozen=True)
class IntentCandidate:
    text: str
    signal: str
    score: float


class IntentCandidateExtractor:

    def extract(
        self,
        text: str,
    ) -> list[IntentCandidate]:

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        normalized = text.strip()

        candidates: list[IntentCandidate] = []

        # Question-form signal
        if normalized.endswith("?"):
            candidates.append(
                IntentCandidate(
                    text=normalized,
                    signal="question_form",
                    score=1.0,
                )
            )

        # Imperative/action signal.
        # Use the first meaningful token as the
        # candidate signal rather than assigning
        # a final intent label.
        tokens = normalized.split()

        if tokens:
            first_token = tokens[0].lower()

            if first_token in {
                "how",
                "why",
                "what",
                "when",
                "where",
                "which",
                "who",
            }:
                candidates.append(
                    IntentCandidate(
                        text=normalized,
                        signal="interrogative",
                        score=0.9,
                    )
                )

        return candidates