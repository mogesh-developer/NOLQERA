from __future__ import annotations

from typing import List

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def _tokenize(text: str) -> set[str]:
    return set(text.lower().split())


def _keyword_overlap(
    query_tokens: set[str],
    document_tokens: set[str],
) -> float:
    if not query_tokens:
        return 0.0

    return len(query_tokens & document_tokens) / len(query_tokens)


def rerank_results(
    query: str,
    results: List[SemanticSearchResult],
    relevance_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[SemanticSearchResult]:
    """
    Re-rank retrieval results using the original relevance score
    and query keyword overlap.

    final_score =
        relevance_weight * result.score
        + keyword_weight * keyword_overlap
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if not query.strip():
        raise ValueError("query cannot be empty")

    if not isinstance(results, list):
        raise TypeError("results must be a list")

    if not isinstance(relevance_weight, (int, float)):
        raise TypeError(
            "relevance_weight must be numeric"
        )

    if not isinstance(keyword_weight, (int, float)):
        raise TypeError(
            "keyword_weight must be numeric"
        )

    if relevance_weight < 0 or relevance_weight > 1:
        raise ValueError(
            "relevance_weight must be between 0 and 1"
        )

    if keyword_weight < 0 or keyword_weight > 1:
        raise ValueError(
            "keyword_weight must be between 0 and 1"
        )

    if abs(
        (relevance_weight + keyword_weight) - 1.0
    ) > 1e-9:
        raise ValueError(
            "weights must sum to 1"
        )

    for result in results:
        if not isinstance(result, SemanticSearchResult):
            raise TypeError(
                "results must contain "
                "SemanticSearchResult objects"
            )

    if not results:
        return []

    query_tokens = _tokenize(query)

    scored_results = []

    for position, result in enumerate(results):
        document_tokens = _tokenize(result.text)

        keyword_score = _keyword_overlap(
            query_tokens,
            document_tokens,
        )

        final_score = (
            relevance_weight * result.score
            + keyword_weight * keyword_score
        )

        scored_results.append(
            (
                final_score,
                position,
                result,
            )
        )

    # Higher final score first.
    # Original position is used as deterministic tie-breaker.
    scored_results.sort(
        key=lambda item: (-item[0], item[1])
    )

    return [
        result
        for _, _, result in scored_results
    ]