from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def make_ranked_context(
    text: str,
    index: int,
    relevance: float = 0.90,
    importance: float = 0.90,
    ranking: float = 0.90,
) -> RankedContext:

    result = SemanticSearchResult(
        text=text,
        score=relevance,
        index=index,
    )

    return RankedContext(
        result=result,
        relevance_score=relevance,
        importance_score=importance,
        ranking_score=ranking,
    )


def calculate_completeness(
    expected_facts,
    compressed_text: str,
) -> float:

    if not expected_facts:
        return 1.0

    preserved = sum(
        1
        for fact in expected_facts
        if fact.lower() in compressed_text.lower()
    )

    return preserved / len(expected_facts)


def test_complete_answer_is_preserved():

    expected_facts = [
        "semantic search",
        "keyword retrieval",
        "reranking",
    ]

    compressed = [
        make_ranked_context(
            "NOLQERA uses semantic search "
            "for meaning-based retrieval.",
            0,
        ),
        make_ranked_context(
            "NOLQERA uses keyword retrieval "
            "for exact term matching.",
            1,
        ),
        make_ranked_context(
            "NOLQERA applies reranking to improve "
            "candidate ordering.",
            2,
        ),
    ]

    compressed_text = " ".join(
        context.result.text
        for context in compressed
    )

    score = calculate_completeness(
        expected_facts,
        compressed_text,
    )

    assert score == 1.0


def test_partial_answer_is_detected():

    expected_facts = [
        "semantic search",
        "keyword retrieval",
        "reranking",
    ]

    compressed = [
        make_ranked_context(
            "NOLQERA uses semantic search "
            "for meaning-based retrieval.",
            0,
        ),
        make_ranked_context(
            "NOLQERA uses keyword retrieval "
            "for exact term matching.",
            1,
        ),
    ]

    compressed_text = " ".join(
        context.result.text
        for context in compressed
    )

    score = calculate_completeness(
        expected_facts,
        compressed_text,
    )

    assert score == 2 / 3
    assert score < 1.0


def test_zero_completeness_is_detected():

    expected_facts = [
        "semantic search",
        "keyword retrieval",
        "reranking",
    ]

    compressed = [
        make_ranked_context(
            "The system performs context compression.",
            0,
        ),
    ]

    compressed_text = " ".join(
        context.result.text
        for context in compressed
    )

    score = calculate_completeness(
        expected_facts,
        compressed_text,
    )

    assert score == 0.0


def test_completeness_threshold_can_be_evaluated():

    expected_facts = [
        "semantic search",
        "keyword retrieval",
        "reranking",
        "deduplication",
    ]

    compressed = [
        make_ranked_context(
            "NOLQERA uses semantic search.",
            0,
        ),
        make_ranked_context(
            "NOLQERA uses keyword retrieval.",
            1,
        ),
        make_ranked_context(
            "NOLQERA applies reranking.",
            2,
        ),
    ]

    compressed_text = " ".join(
        context.result.text
        for context in compressed
    )

    score = calculate_completeness(
        expected_facts,
        compressed_text,
    )

    assert score == 0.75
    assert score >= 0.75
    assert score < 1.0


def test_empty_expected_facts_are_complete():

    score = calculate_completeness(
        [],
        "Any compressed context.",
    )

    assert score == 1.0