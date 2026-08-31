from __future__ import annotations

from typing import List

from nolqera.intelligence.semantic_search.models import SemanticSearchResult


def deduplicate_results(
    results: List[SemanticSearchResult],
) -> List[SemanticSearchResult]:
    """
    Remove exact duplicate search results while preserving order.

    The first occurrence of each unique result text is kept.
    Result objects themselves are preserved.
    """

    if not isinstance(results, list):
        raise TypeError("results must be a list")

    for result in results:
        if not isinstance(result, SemanticSearchResult):
            raise TypeError(
                "results must contain SemanticSearchResult objects"
            )

    seen_texts: set[str] = set()
    unique_results: List[SemanticSearchResult] = []

    for result in results:
        if result.text in seen_texts:
            continue

        seen_texts.add(result.text)
        unique_results.append(result)

    return unique_results