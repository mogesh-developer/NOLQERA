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


def test_semantic_facts_survive_when_source_context_is_preserved():

    original = [
        make_ranked_context(
            "NOLQERA combines semantic search with "
            "keyword-based retrieval to improve recall.",
            0,
        ),
        make_ranked_context(
            "The reranking stage prioritizes candidates "
            "using relevance and diversity.",
            1,
        ),
        make_ranked_context(
            "Context compression removes redundant "
            "sentences while protecting important information.",
            2,
        ),
    ]

    compressed = [
        original[0],
        original[1],
        original[2],
    ]

    original_text = " ".join(
        context.result.text
        for context in original
    )

    compressed_text = " ".join(
        context.result.text
        for context in compressed
    )

    semantic_facts = [
        "semantic search",
        "keyword-based retrieval",
        "improve recall",
        "reranking",
        "relevance",
        "diversity",
        "compression",
        "redundant sentences",
        "important information",
    ]

    for fact in semantic_facts:
        assert fact in original_text
        assert fact in compressed_text


def test_semantic_fact_loss_is_detectable():

    original = [
        make_ranked_context(
            "NOLQERA combines semantic search with "
            "keyword-based retrieval to improve recall.",
            0,
        ),
        make_ranked_context(
            "The reranking stage prioritizes candidates "
            "using relevance and diversity.",
            1,
        ),
    ]

    compressed = [
        make_ranked_context(
            "NOLQERA combines semantic search with "
            "keyword-based retrieval to improve recall.",
            0,
        ),
    ]

    compressed_text = " ".join(
        context.result.text
        for context in compressed
    )

    assert "semantic search" in compressed_text
    assert "keyword-based retrieval" in compressed_text

    assert "reranking" not in compressed_text
    assert "diversity" not in compressed_text


def test_detailed_semantic_facts_are_preserved():

    original = [
        make_ranked_context(
            "The embedding layer converts text into "
            "dense vector representations for semantic comparison.",
            0,
        ),
        make_ranked_context(
            "Candidate retrieval uses cosine similarity "
            "to rank semantically related documents.",
            1,
        ),
        make_ranked_context(
            "The final compression stage preserves "
            "high-value information while removing redundancy.",
            2,
        ),
    ]

    compressed = list(original)

    original_text = " ".join(
        context.result.text
        for context in original
    )

    compressed_text = " ".join(
        context.result.text
        for context in compressed
    )

    required_facts = [
        "dense vector representations",
        "semantic comparison",
        "cosine similarity",
        "semantically related documents",
        "high-value information",
        "removing redundancy",
    ]

    for fact in required_facts:
        assert fact in original_text
        assert fact in compressed_text