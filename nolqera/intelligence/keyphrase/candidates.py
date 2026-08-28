import re


class KeyphraseCandidateExtractor:
    """Generate meaningful n-gram keyphrase candidates."""

    def __init__(
        self,
        min_n: int = 1,
        max_n: int = 3,
        stopwords: set[str] | None = None,
    ):
        if not isinstance(min_n, int):
            raise TypeError("min_n must be an integer")

        if not isinstance(max_n, int):
            raise TypeError("max_n must be an integer")

        if min_n <= 0:
            raise ValueError("min_n must be greater than zero")

        if max_n < min_n:
            raise ValueError(
                "max_n must be greater than or equal to min_n"
            )

        self.min_n = min_n
        self.max_n = max_n

        self.stopwords = stopwords or {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "then",
            "than",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "to",
            "of",
            "for",
            "from",
            "in",
            "on",
            "at",
            "by",
            "with",
            "as",
            "into",
            "through",
            "about",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "use",
            "uses",
            "used",
        }

    def _is_valid_token(self, token: str) -> bool:
        """Return True when a token can participate in a keyphrase."""

        if not isinstance(token, str):
            return False

        token = token.strip().lower()

        if not token:
            return False

        if token in self.stopwords:
            return False

        # Reject punctuation-only tokens.
        if not re.search(r"[a-zA-Z0-9]", token):
            return False

        return True

    def extract(
        self,
        tokens: list[str],
    ) -> list[str]:
        """Generate filtered n-gram candidates."""

        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list")

        if not tokens:
            raise ValueError("tokens cannot be empty")

        if any(
            not isinstance(token, str)
            for token in tokens
        ):
            raise TypeError(
                "tokens must contain only strings"
            )

        candidates = []
        seen = set()

        # Keep valid tokens while preserving their original order.
        valid_tokens = [
            token.strip().lower()
            for token in tokens
            if self._is_valid_token(token)
        ]

        for n in range(
            self.min_n,
            self.max_n + 1,
        ):
            for index in range(
                len(valid_tokens) - n + 1
            ):
                phrase_tokens = valid_tokens[
                    index:index + n
                ]

                # Don't allow a phrase made entirely from stopwords.
                if all(
                    token in self.stopwords
                    for token in phrase_tokens
                ):
                    continue

                phrase = " ".join(phrase_tokens)

                if phrase not in seen:
                    seen.add(phrase)
                    candidates.append(phrase)

        return candidates