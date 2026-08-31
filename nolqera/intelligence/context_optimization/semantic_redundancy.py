from __future__ import annotations

from typing import List

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)

from nolqera.intelligence.semantic_similarity.engine import (
    SemanticSimilarityEngine,
)


from nolqera.tokenization.word_tokenizer import tokenize_words


def detect_semantic_redundancy(
    first: str,
    second: str,
    similarity_engine: SemanticSimilarityEngine,
    similarity_threshold: float = 0.85,
) -> bool:
    """
    Detect whether two texts contain semantically redundant
    information.

    Unlike lexical near-duplicate detection, this compares the
    meaning represented by embeddings.
    """

    if not isinstance(first, str):
        raise TypeError("first must be a string")

    if not isinstance(second, str):
        raise TypeError("second must be a string")

    if not first.strip():
        raise ValueError("first cannot be empty")

    if not second.strip():
        raise ValueError("second cannot be empty")

    if not isinstance(
        similarity_engine,
        SemanticSimilarityEngine,
    ):
        raise TypeError(
            "similarity_engine must be a SemanticSimilarityEngine"
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

    tokens_a = tokenize_words(first)
    tokens_b = tokenize_words(second)

    result = similarity_engine.compare(
        tokens_a,
        tokens_b,
    )

    return result.score >= similarity_threshold


def remove_semantic_redundancy(
    results: List[SemanticSearchResult],
    similarity_engine: SemanticSimilarityEngine,
    similarity_threshold: float = 0.85,
) -> List[SemanticSearchResult]:
    """
    Remove semantically redundant search results.

    Results are processed in their existing order.
    The first occurrence is preserved.

    The first/highest-ranked result is therefore preferred,
    while later semantically redundant results are removed.
    """

    if not isinstance(results, list):
        raise TypeError("results must be a list")

    if not isinstance(
        similarity_engine,
        SemanticSimilarityEngine,
    ):
        raise TypeError(
            "similarity_engine must be a SemanticSimilarityEngine"
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
                "results must contain SemanticSearchResult objects"
            )

    if not results:
        return []

    selected: List[SemanticSearchResult] = []

    for candidate in results:

        redundant = any(
            detect_semantic_redundancy(
                candidate.text,
                existing.text,
                similarity_engine,
                similarity_threshold,
            )
            for existing in selected
        )

        if redundant:
            continue

        selected.append(candidate)

    return selected