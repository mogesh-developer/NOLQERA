from __future__ import annotations

import re
from typing import List

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def _normalize_text(text: str) -> str:
    """
    Normalize text for near-duplicate comparison.

    Case and punctuation differences are ignored.
    """

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    return " ".join(text.split())


def _tokenize(text: str) -> set[str]:
    return set(
        _normalize_text(text).split()
    )


def _jaccard_similarity(
    first: set[str],
    second: set[str],
) -> float:
    """
    Calculate token-level Jaccard similarity.
    """

    if not first and not second:
        return 1.0

    if not first or not second:
        return 0.0

    return len(first & second) / len(first | second)


def is_near_duplicate(
    first: str,
    second: str,
    similarity_threshold: float = 0.8,
) -> bool:
    """
    Determine whether two texts are near duplicates.

    Texts are considered near duplicates when their normalized
    token sets have Jaccard similarity greater than or equal to
    the configured threshold.
    """

    if not isinstance(first, str):
        raise TypeError(
            "first must be a string"
        )

    if not isinstance(second, str):
        raise TypeError(
            "second must be a string"
        )

    if not first.strip():
        raise ValueError(
            "first cannot be empty"
        )

    if not second.strip():
        raise ValueError(
            "second cannot be empty"
        )

    if not isinstance(
        similarity_threshold,
        (int, float),
    ):
        raise TypeError(
            "similarity_threshold must be numeric"
        )

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be between 0 and 1"
        )

    first_tokens = _tokenize(first)
    second_tokens = _tokenize(second)

    similarity = _jaccard_similarity(
        first_tokens,
        second_tokens,
    )

    return similarity >= similarity_threshold


def remove_near_duplicates(
    results: List[SemanticSearchResult],
    similarity_threshold: float = 0.8,
) -> List[SemanticSearchResult]:
    """
    Remove near-duplicate search results.

    Results are processed in their existing order.
    The first occurrence is preserved and later near-duplicates
    are removed.

    Result objects themselves are preserved.
    """

    if not isinstance(results, list):
        raise TypeError(
            "results must be a list"
        )

    if not isinstance(
        similarity_threshold,
        (int, float),
    ):
        raise TypeError(
            "similarity_threshold must be numeric"
        )

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be between 0 and 1"
        )

    for result in results:
        if not isinstance(
            result,
            SemanticSearchResult,
        ):
            raise TypeError(
                "results must contain "
                "SemanticSearchResult objects"
            )

    if not results:
        return []

    selected: List[
        SemanticSearchResult
    ] = []

    for candidate in results:

        duplicate = any(
            is_near_duplicate(
                candidate.text,
                existing.text,
                similarity_threshold,
            )
            for existing in selected
        )

        if duplicate:
            continue

        selected.append(candidate)

    return selected