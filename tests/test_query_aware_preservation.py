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


def test_query_relevant_information_is_preserved():

    query = "What retrieval methods does NOLQERA use?"

    original = [
        make_ranked_context(
            "NOLQERA uses semantic search for "
            "meaning-based retrieval.",
            0,
        ),
        make_ranked_context(
            "NOLQERA also uses keyword retrieval "
            "for exact term matching.",
            1,
        ),
        make_ranked_context(
            "The system applies reranking to improve "
            "the ordering of retrieved candidates.",
            2,
        ),
        make_ranked_context(
            "The compression layer reduces redundant "
            "context before sending it to the model.",
            3,
        ),
    ]

    compressed = [
        original[0],
        original[1],
        original[2],
    ]

    compressed_text = " ".join(
        context.result.text
        for context in compressed
    )

    query_required_information = [
        "semantic search",
        "keyword retrieval",
        "reranking",
    ]

    for fact in query_required_information:
        assert fact in compressed_text


def test_query_irrelevant_information_can_be_removed():

    query = "What retrieval methods does NOLQERA use?"

    original = [
        make_ranked_context(
            "NOLQERA uses semantic search for "
            "meaning-based retrieval.",
            0,
        ),
        make_ranked_context(
            "NOLQERA uses keyword retrieval "
            "for exact term matching.",
            1,
        ),
        make_ranked_context(
            "The compression layer reduces redundant "
            "context before model inference.",
            2,
        ),
    ]

    compressed = [
        original[0],
        original[1],
    ]

    compressed_text = " ".join(
        context.result.text
        for context in compressed
    )

    assert "semantic search" in compressed_text
    assert "keyword retrieval" in compressed_text

    # This information is not required to answer
    # the retrieval-method query.
    assert "compression layer" not in compressed_text


def test_query_relevant_fact_loss_is_detected():

    query = "What retrieval methods does NOLQERA use?"

    original = [
        make_ranked_context(
            "NOLQERA uses semantic search for "
            "meaning-based retrieval.",
            0,
        ),
        make_ranked_context(
            "NOLQERA also uses keyword retrieval "
            "for exact term matching.",
            1,
        ),
    ]

    compressed = [
        original[0],
    ]

    compressed_text = " ".join(
        context.result.text
        for context in compressed
    )

    assert "semantic search" in compressed_text

    # Query-relevant information was lost.
    assert "keyword retrieval" not in compressed_text


def test_different_queries_require_different_information():

    contexts = [
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
            "NOLQERA uses reranking to improve "
            "candidate ordering.",
            2,
        ),
        make_ranked_context(
            "NOLQERA compression removes "
            "redundant context.",
            3,
        ),
    ]

    retrieval_query = "What retrieval methods are used?"

    retrieval_compressed = [
        contexts[0],
        contexts[1],
        contexts[2],
    ]

    retrieval_text = " ".join(
        context.result.text
        for context in retrieval_compressed
    )

    assert "semantic search" in retrieval_text
    assert "keyword retrieval" in retrieval_text
    assert "reranking" in retrieval_text

    compression_query = "How does NOLQERA reduce context?"

    compression_compressed = [
        contexts[3],
    ]

    compression_text = " ".join(
        context.result.text
        for context in compression_compressed
    )

    assert "compression" in compression_text
    assert "redundant context" in compression_text