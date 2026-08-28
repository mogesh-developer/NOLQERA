import pytest

from nolqera.intelligence.importance.ranking import (
    ImportanceRanker,
)


def test_rank_orders_scores_highest_first():
    ranker = ImportanceRanker()

    results = ranker.rank(
        [0.42, 0.91, 0.18, 0.76]
    )

    assert [item.index for item in results] == [
        1, 3, 0, 2
    ]

    assert [item.rank for item in results] == [
        1, 2, 3, 4
    ]


def test_top_k_returns_requested_items():
    ranker = ImportanceRanker()

    results = ranker.top_k(
        [0.42, 0.91, 0.18, 0.76],
        k=2,
    )

    assert len(results) == 2
    assert results[0].index == 1
    assert results[1].index == 3


def test_equal_scores_are_handled():
    ranker = ImportanceRanker()

    results = ranker.rank(
        [0.5, 0.5, 0.2]
    )

    assert len(results) == 3
    assert results[0].score == 0.5
    assert results[1].score == 0.5


def test_empty_scores_are_rejected():
    ranker = ImportanceRanker()

    with pytest.raises(ValueError):
        ranker.rank([])


def test_invalid_score_is_rejected():
    ranker = ImportanceRanker()

    with pytest.raises(ValueError):
        ranker.rank([0.5, 1.5])


def test_invalid_k_is_rejected():
    ranker = ImportanceRanker()

    with pytest.raises(ValueError):
        ranker.top_k([0.5, 0.2], 0)