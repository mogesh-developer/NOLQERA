from __future__ import annotations


class QueryPreprocessor:
    """
    Preprocesses search queries before candidate retrieval.

    Current responsibilities:
    - Validate query input
    - Normalize surrounding whitespace
    - Collapse repeated whitespace
    - Preserve the semantic content of the query
    """

    def preprocess(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        # Remove leading/trailing whitespace
        query = query.strip()

        # Collapse repeated whitespace
        query = " ".join(query.split())

        return query