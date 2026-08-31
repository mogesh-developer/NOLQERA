import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    ContextRanker,
    RankedContext,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def make_result(
    text: str,
    score: float,
    index: int,
):
    return SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )


def test_ranker_accepts_weights():

    ranker = ContextRanker(
        relevance_weight=0.7,
        importance_weight=0.3,
    )

    assert ranker.relevance_weight == pytest.approx(0.7)
    assert ranker.importance_weight == pytest.approx(0.3)


def test_weights_are_normalized():

    ranker = ContextRanker(
        relevance_weight=7,
        importance_weight=3,
    )

    assert ranker.relevance_weight == pytest.approx(0.7)
    assert ranker.importance_weight == pytest.approx(0.3)


def test_ranker_rejects_non_numeric_weight():

    with pytest.raises(TypeError):
        ContextRanker(
            relevance_weight="0.7"
        )


def test_ranker_rejects_negative_weight():

    with pytest.raises(ValueError):
        ContextRanker(
            relevance_weight=-0.1
        )


def test_ranker_rejects_zero_total_weight():

    with pytest.raises(ValueError):
        ContextRanker(
            relevance_weight=0,
            importance_weight=0,
        )


def test_rank_returns_ranked_context():

    ranker = ContextRanker()

    results = [
        make_result(
            "FastAPI authentication",
            0.9,
            0,
        )
    ]

    ranked = ranker.rank(
        results,
        [0.8],
    )

    assert len(ranked) == 1
    assert isinstance(
        ranked[0],
        RankedContext,
    )


def test_ranking_score_is_calculated_correctly():

    ranker = ContextRanker(
        relevance_weight=0.7,
        importance_weight=0.3,
    )

    result = make_result(
        "FastAPI authentication",
        0.9,
        0,
    )

    ranked = ranker.rank(
        [result],
        [0.8],
    )

    assert ranked[0].ranking_score == pytest.approx(
        0.87
    )


def test_highest_ranking_score_comes_first():

    ranker = ContextRanker()

    results = [
        make_result(
            "Result A",
            0.8,
            0,
        ),
        make_result(
            "Result B",
            0.95,
            1,
        ),
    ]

    ranked = ranker.rank(
        results,
        [0.5, 0.9],
    )

    assert ranked[0].result.text == "Result B"
    assert ranked[1].result.text == "Result A"


def test_importance_can_change_final_ranking():

    ranker = ContextRanker(
        relevance_weight=0.7,
        importance_weight=0.3,
    )

    result_a = make_result(
        "Result A",
        0.90,
        0,
    )

    result_b = make_result(
        "Result B",
        0.85,
        1,
    )

    ranked = ranker.rank(
        [result_a, result_b],
        [0.80, 0.95],
    )

    assert ranked[0].result.text == "Result B"


def test_rank_preserves_result_objects():

    ranker = ContextRanker()

    result = make_result(
        "FastAPI authentication",
        0.9,
        0,
    )

    ranked = ranker.rank(
        [result],
        [0.8],
    )

    assert ranked[0].result is result


def test_rank_preserves_scores():

    ranker = ContextRanker()

    result = make_result(
        "FastAPI authentication",
        0.9,
        0,
    )

    ranked = ranker.rank(
        [result],
        [0.8],
    )

    assert ranked[0].relevance_score == 0.9
    assert ranked[0].importance_score == 0.8


def test_empty_results_return_empty_list():

    ranker = ContextRanker()

    assert ranker.rank([], []) == []


def test_rejects_non_list_results():

    ranker = ContextRanker()

    with pytest.raises(TypeError):
        ranker.rank(
            None,
            [],
        )


def test_rejects_non_list_importance_scores():

    ranker = ContextRanker()

    with pytest.raises(TypeError):
        ranker.rank(
            [],
            None,
        )


def test_rejects_mismatched_lengths():

    ranker = ContextRanker()

    results = [
        make_result(
            "FastAPI",
            0.8,
            0,
        )
    ]

    with pytest.raises(ValueError):
        ranker.rank(
            results,
            [],
        )


def test_rejects_invalid_relevance_score():

    ranker = ContextRanker()

    result = object.__new__(SemanticSearchResult)
    object.__setattr__(result, "text", "FastAPI")
    object.__setattr__(result, "score", 1.5)
    object.__setattr__(result, "index", 0)

    with pytest.raises(ValueError):
        ranker.rank(
            [result],
            [0.5],
        )


def test_rejects_invalid_importance_score():

    ranker = ContextRanker()

    result = make_result(
        "FastAPI",
        0.8,
        0,
    )

    with pytest.raises(ValueError):
        ranker.rank(
            [result],
            [1.5],
        )