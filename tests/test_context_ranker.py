import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    ContextRanker,
    RankedContext,
)
from nolqera.intelligence.pipeline.context_ranker import (
    ContextRankingAnalyzer,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def make_result(
    text: str,
    score: float,
    index: int,
) -> SemanticSearchResult:
    return SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )


def test_context_ranking_analyzer_accepts_ranker():
    ranker = ContextRanker()

    analyzer = ContextRankingAnalyzer(ranker)

    assert isinstance(
        analyzer,
        ContextRankingAnalyzer,
    )


def test_context_ranking_analyzer_rejects_invalid_ranker():
    with pytest.raises(
        TypeError,
        match="context_ranker must be a ContextRanker",
    ):
        ContextRankingAnalyzer(object())


def test_rank_returns_ranked_context_objects():
    analyzer = ContextRankingAnalyzer(
        ContextRanker()
    )

    results = [
        make_result("Python backend", 0.9, 0),
        make_result("Python frontend", 0.7, 1),
    ]

    ranked = analyzer.rank(
        results,
        [0.8, 0.6],
    )

    assert all(
        isinstance(item, RankedContext)
        for item in ranked
    )


def test_rank_returns_exact_ranking_scores():
    analyzer = ContextRankingAnalyzer(
        ContextRanker(
            relevance_weight=0.7,
            importance_weight=0.3,
        )
    )

    results = [
        make_result("Python backend", 0.9, 0),
        make_result("Python frontend", 0.7, 1),
    ]

    ranked = analyzer.rank(
        results,
        [0.8, 0.6],
    )

    assert len(ranked) == 2

    assert ranked[0].result is results[0]
    assert ranked[0].relevance_score == pytest.approx(0.9)
    assert ranked[0].importance_score == pytest.approx(0.8)
    assert ranked[0].ranking_score == pytest.approx(0.87)

    assert ranked[1].result is results[1]
    assert ranked[1].relevance_score == pytest.approx(0.7)
    assert ranked[1].importance_score == pytest.approx(0.6)
    assert ranked[1].ranking_score == pytest.approx(0.67)


def test_rank_orders_highest_ranking_score_first():
    analyzer = ContextRankingAnalyzer(
        ContextRanker()
    )

    results = [
        make_result("Low relevance", 0.4, 0),
        make_result("High relevance", 0.9, 1),
        make_result("Medium relevance", 0.6, 2),
    ]

    ranked = analyzer.rank(
        results,
        [0.4, 0.8, 0.6],
    )

    assert [
        item.result.text
        for item in ranked
    ] == [
        "High relevance",
        "Medium relevance",
        "Low relevance",
    ]


def test_rank_uses_importance_with_relevance():
    analyzer = ContextRankingAnalyzer(
        ContextRanker(
            relevance_weight=0.5,
            importance_weight=0.5,
        )
    )

    results = [
        make_result("Highly relevant", 0.9, 0),
        make_result("Highly important", 0.7, 1),
    ]

    ranked = analyzer.rank(
        results,
        [0.5, 1.0],
    )

    assert ranked[0].result.text == "Highly important"
    assert ranked[0].ranking_score == pytest.approx(0.85)

    assert ranked[1].result.text == "Highly relevant"
    assert ranked[1].ranking_score == pytest.approx(0.7)



def test_rank_rejects_mismatched_importance_scores():
    analyzer = ContextRankingAnalyzer(
        ContextRanker()
    )

    results = [
        make_result("Python backend", 0.9, 0),
        make_result("Python API", 0.8, 1),
    ]

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        analyzer.rank(results, [0.8])


def test_rank_preserves_result_objects():
    analyzer = ContextRankingAnalyzer(
        ContextRanker()
    )

    first = make_result(
        "FastAPI backend",
        0.95,
        4,
    )

    ranked = analyzer.rank(
        [first],
        [0.9],
    )

    assert ranked[0].result is first
    assert ranked[0].result.text == "FastAPI backend"
    assert ranked[0].result.index == 4


def test_rank_empty_results():
    analyzer = ContextRankingAnalyzer(
        ContextRanker()
    )

    assert analyzer.rank([], []) == []