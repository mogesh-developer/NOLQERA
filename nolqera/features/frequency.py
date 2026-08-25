from collections import Counter


class FrequencyAnalyzer:
    """Analyze token frequency across documents."""

    def __init__(self):
        self.frequencies: Counter[str] = Counter()
        self.document_frequencies: Counter[str] = Counter()
        self.total_tokens = 0
        self.total_documents = 0

    def fit(self, documents: list[list[str]]) -> None:
        """Calculate corpus and document frequencies."""

        if not isinstance(documents, list):
            raise TypeError("documents must be a list")

        for document in documents:
            if not isinstance(document, list):
                raise TypeError(
                    "each document must be a list"
                )

            for token in document:
                if not isinstance(token, str):
                    raise TypeError(
                        "tokens must be strings"
                    )

        self.frequencies = Counter(
            token
            for document in documents
            for token in document
        )

        self.document_frequencies = Counter(
            token
            for document in documents
            for token in set(document)
        )

        self.total_tokens = sum(
            self.frequencies.values()
        )

        self.total_documents = len(documents)

    def count(self, token: str) -> int:
        """Return total corpus frequency."""

        if not isinstance(token, str):
            raise TypeError("token must be a string")

        return self.frequencies[token]

    def document_frequency(self, token: str) -> int:
        """Return number of documents containing the token."""

        if not isinstance(token, str):
            raise TypeError("token must be a string")

        return self.document_frequencies[token]

    def unique_count(self) -> int:
        """Return number of unique tokens."""

        return len(self.frequencies)

    def most_common(
        self,
        n: int | None = None,
    ) -> list[tuple[str, int]]:
        """Return most frequent tokens."""

        return self.frequencies.most_common(n)

    def frequency(self, token: str) -> float:
        """Return relative corpus frequency."""

        if not isinstance(token, str):
            raise TypeError("token must be a string")

        if self.total_tokens == 0:
            return 0.0

        return (
            self.frequencies[token]
            / self.total_tokens
        )

    def document_frequency_ratio(
        self,
        token: str,
    ) -> float:
        """Return the proportion of documents containing the token."""

        if not isinstance(token, str):
            raise TypeError("token must be a string")

        if self.total_documents == 0:
            return 0.0

        return (
            self.document_frequencies[token]
            / self.total_documents
        )

    def summary(self) -> dict[str, int]:
        """Return corpus frequency statistics."""

        return {
            "total_tokens": self.total_tokens,
            "unique_tokens": self.unique_count(),
            "total_documents": self.total_documents,
        }
    def to_dict(self) -> dict[str, int]:
        """Return frequency statistics as a dictionary."""

        return self.summary()