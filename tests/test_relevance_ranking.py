import pytest

from nolqera.intelligence.relevance.ranking import (
    RelevanceRanker,
)


def test_rank_orders_scores_highest_first():
    ranker = RelevanceRanker()

    result = ranker.rank(
        [0.20, 0.90, 0.50, 0.10]
    )

    assert [item.index for item in result] == [
        1, 2, 0, 3
    ]

    assert [item.score for item in result] == [
        0.90, 0.50, 0.20, 0.10
    ]


def test_top_k_returns_only_requested_items():
    ranker = RelevanceRanker()

    result = ranker.top_k(
        [0.20, 0.90, 0.50, 0.10],
        k=2,
    )

    assert [item.index for item in result] == [1, 2]


def test_equal_scores_are_handled():
    ranker = RelevanceRanker()

    result = ranker.rank(
        [0.50, 0.50, 0.20]
    )

    assert [item.index for item in result] == [
        0, 1, 2
    ]


def test_empty_scores_are_rejected():
    ranker = RelevanceRanker()

    with pytest.raises(ValueError):
        ranker.rank([])


def test_invalid_score_is_rejected():
    ranker = RelevanceRanker()

    with pytest.raises(ValueError):
        ranker.rank([0.5, 1.2])


def test_invalid_k_is_rejected():
    ranker = RelevanceRanker()

    with pytest.raises(ValueError):
        ranker.top_k([0.2, 0.8], 0)