from __future__ import annotations

from typing import List

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)

from nolqera.intelligence.semantic_similarity.engine import (
    SemanticSimilarityEngine,
)


from nolqera.tokenization.word_tokenizer import tokenize_words


def collapse_redundant_information(
    results: List[SemanticSearchResult],
    similarity_engine: SemanticSimilarityEngine,
    similarity_threshold: float = 0.85,
) -> List[SemanticSearchResult]:
    """
    Collapse semantically redundant retrieval results.

    Results are processed in ranking order. The first result in a
    redundant group is retained and later semantically redundant
    results are removed.

    This preserves the highest-ranked representation while reducing
    repeated information.
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

    collapsed: List[SemanticSearchResult] = []

    for candidate in results:

        is_redundant = False

        for retained in collapsed:

            tokens_a = tokenize_words(candidate.text)
            tokens_b = tokenize_words(retained.text)

            similarity = similarity_engine.compare(
                tokens_a,
                tokens_b,
            )

            if similarity.score >= similarity_threshold:
                is_redundant = True
                break

        if not is_redundant:
            collapsed.append(candidate)

    return collapsed