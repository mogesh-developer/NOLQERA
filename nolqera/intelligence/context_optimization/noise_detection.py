from __future__ import annotations

import re
from typing import List

from nolqera.intelligence.semantic_search.models import SemanticSearchResult


class NoiseDetector:
    """
    Detects low-information / noisy retrieval results.

    A result is considered noisy when its text contains no meaningful
    lexical content after normalization.
    """

    def __init__(self, min_meaningful_tokens: int = 2):
        if not isinstance(min_meaningful_tokens, int):
            raise TypeError("min_meaningful_tokens must be an integer")

        if min_meaningful_tokens <= 0:
            raise ValueError(
                "min_meaningful_tokens must be greater than zero"
            )

        self.min_meaningful_tokens = min_meaningful_tokens

    @staticmethod
    def _meaningful_tokens(text: str) -> List[str]:
        """
        Extract meaningful alphanumeric tokens from text.
        """
        return re.findall(r"\b[a-zA-Z0-9]+\b", text)

    def is_noise(self, result: SemanticSearchResult) -> bool:
        """
        Return True when a result is considered noisy.
        """
        if not isinstance(result, SemanticSearchResult):
            raise TypeError(
                "result must be a SemanticSearchResult"
            )

        tokens = self._meaningful_tokens(result.text)

        return len(tokens) < self.min_meaningful_tokens

    def filter(self, results: List[SemanticSearchResult]) -> List[SemanticSearchResult]:
        """
        Remove noisy results while preserving order and result objects.
        """
        if not isinstance(results, list):
            raise TypeError("results must be a list")

        for result in results:
            if not isinstance(result, SemanticSearchResult):
                raise TypeError(
                    "all results must be SemanticSearchResult instances"
                )

        return [
            result
            for result in results
            if not self.is_noise(result)
        ]