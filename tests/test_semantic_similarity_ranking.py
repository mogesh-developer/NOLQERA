from nolqera.intelligence.semantic_similarity.models import (
    SemanticSimilarityResult,
)

from nolqera.intelligence.semantic_similarity.ranking import (
    SemanticSimilarityRanker,
)


def test_rank_orders_highest_first():

    ranker = SemanticSimilarityRanker()

    results = [
        SemanticSimilarityResult(
            text_a="A",
            text_b="B",
            score=0.4,
        ),
        SemanticSimilarityResult(
            text_a="A",
            text_b="C",
            score=0.9,
        ),
        SemanticSimilarityResult(
            text_a="A",
            text_b="D",
            score=0.7,
        ),
    ]

    ranked = ranker.rank(results)

    assert [
        result.score
        for result in ranked
    ] == [0.9, 0.7, 0.4]


def test_top_k_returns_requested_items():

    ranker = SemanticSimilarityRanker()

    results = [
        SemanticSimilarityResult(
            text_a="A",
            text_b="B",
            score=0.9,
        ),
        SemanticSimilarityResult(
            text_a="A",
            text_b="C",
            score=0.7,
        ),
    ]

    top = ranker.top_k(results, 1)

    assert len(top) == 1
    assert top[0].score == 0.9
