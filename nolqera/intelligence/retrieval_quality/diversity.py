from __future__ import annotations

from typing import List

from nolqera.intelligence.semantic_search.models import SemanticSearchResult


def _tokenize(text: str) -> set[str]:
    return set(text.lower().split())


def _jaccard_similarity(
    first: set[str],
    second: set[str],
) -> float:
    if not first and not second:
        return 1.0

    if not first or not second:
        return 0.0

    return len(first & second) / len(first | second)


def diversify_results(
    results: List[SemanticSearchResult],
    similarity_threshold: float = 0.8,
) -> List[SemanticSearchResult]:
    """
    Reduce highly similar results while preserving relevance order.

    Results are processed in their existing order, so the first
    occurrence is preferred when multiple results contain very
    similar information.

    similarity_threshold:
        Maximum allowed Jaccard similarity between a candidate
        and an already-selected result.

        A candidate is removed when its similarity is greater than
        or equal to this threshold.
    """

    if not isinstance(results, list):
        raise TypeError("results must be a list")

    if not isinstance(similarity_threshold, (int, float)):
        raise TypeError(
            "similarity_threshold must be numeric"
        )

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be between 0 and 1"
        )

    for result in results:
        if not isinstance(result, SemanticSearchResult):
            raise TypeError(
                "results must contain SemanticSearchResult objects"
            )

    if not results:
        return []

    selected: List[SemanticSearchResult] = []
    selected_tokens: List[set[str]] = []

    for result in results:
        candidate_tokens = _tokenize(result.text)

        is_too_similar = any(
            _jaccard_similarity(
                candidate_tokens,
                existing_tokens,
            )
            >= similarity_threshold
            for existing_tokens in selected_tokens
        )

        if is_too_similar:
            continue

        selected.append(result)
        selected_tokens.append(candidate_tokens)

    return selected