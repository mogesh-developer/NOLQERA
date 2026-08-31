import pytest

from nolqera.intelligence.retrieval_quality.evaluation import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


def test_precision_at_k():

    retrieved = [1, 2, 3, 4]
    relevant = {1, 3}

    assert precision_at_k(
        retrieved,
        relevant,
        4,
    ) == 0.5


def test_precision_at_k_limits_to_k():

    retrieved = [1, 2, 3, 4]
    relevant = {1, 3}

    assert precision_at_k(
        retrieved,
        relevant,
        2,
    ) == 0.5


def test_recall_at_k():

    retrieved = [1, 2, 3, 4]
    relevant = {1, 3, 5}

    assert recall_at_k(
        retrieved,
        relevant,
        4,
    ) == pytest.approx(2 / 3)


def test_recall_at_k_finds_all():

    retrieved = [1, 2, 3]
    relevant = {1, 2}

    assert recall_at_k(
        retrieved,
        relevant,
        3,
    ) == 1.0


def test_hit_rate_when_relevant_result_exists():

    retrieved = [4, 7, 2]
    relevant = {2}

    assert hit_rate_at_k(
        retrieved,
        relevant,
        3,
    ) == 1.0


def test_hit_rate_when_no_relevant_result_exists():

    retrieved = [4, 7, 8]
    relevant = {2}

    assert hit_rate_at_k(
        retrieved,
        relevant,
        3,
    ) == 0.0


def test_reciprocal_rank_first_result():

    retrieved = [1, 2, 3]
    relevant = {1}

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == 1.0


def test_reciprocal_rank_second_result():

    retrieved = [1, 2, 3]
    relevant = {2}

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == 0.5


def test_reciprocal_rank_third_result():

    retrieved = [1, 2, 3]
    relevant = {3}

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_missing():

    retrieved = [1, 2, 3]
    relevant = {9}

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == 0.0


def test_mean_reciprocal_rank():

    rankings = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    relevant_sets = [
        {1},
        {5},
    ]

    assert mean_reciprocal_rank(
        rankings,
        relevant_sets,
    ) == pytest.approx(0.75)


def test_empty_precision_returns_zero():

    assert precision_at_k(
        [],
        {1},
        5,
    ) == 0.0


def test_empty_relevant_precision_returns_zero():

    assert precision_at_k(
        [1, 2],
        set(),
        2,
    ) == 0.0


def test_empty_relevant_recall_returns_zero():

    assert recall_at_k(
        [1, 2],
        set(),
        2,
    ) == 0.0


def test_empty_results_hit_rate_returns_zero():

    assert hit_rate_at_k(
        [],
        {1},
        2,
    ) == 0.0


def test_empty_results_reciprocal_rank_returns_zero():

    assert reciprocal_rank(
        [],
        {1},
    ) == 0.0


def test_mrr_rejects_mismatched_lengths():

    with pytest.raises(ValueError):
        mean_reciprocal_rank(
            [[1, 2]],
            [{1}, {2}],
        )


def test_mrr_empty_rankings_returns_zero():

    assert mean_reciprocal_rank(
        [],
        [],
    ) == 0.0


def test_k_must_be_integer():

    with pytest.raises(TypeError):
        precision_at_k(
            [1],
            {1},
            1.5,
        )


def test_k_must_be_positive():

    with pytest.raises(ValueError):
        precision_at_k(
            [1],
            {1},
            0,
        )