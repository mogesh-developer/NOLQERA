from collections.abc import Iterable


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


class StopwordRemover:
    """Remove stopwords from a token sequence."""

    def __init__(
        self,
        stopwords: Iterable[str] | None = None,
    ):
        if stopwords is None:
            self.stopwords = set(DEFAULT_STOPWORDS)
        else:
            self.stopwords = {
                word.lower()
                for word in stopwords
            }

    def remove(self, tokens: list[str]) -> list[str]:
        """Remove configured stopwords from tokens."""

        return [
            token
            for token in tokens
            if token.lower() not in self.stopwords
        ]