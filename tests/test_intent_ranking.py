import pytest

from nolqera.intelligence.intent.ranking import (
    IntentRanker,
)

from nolqera.intelligence.intent.scorer import (
    IntentScore,
)


def test_rank_orders_highest_first():

    ranker = IntentRanker()

    scores = [
        IntentScore(
            intent="request",
            score=0.60,
            evidence_count=1,
        ),
        IntentScore(
            intent="question",
            score=0.90,
            evidence_count=2,
        ),
        IntentScore(
            intent="explanation",
            score=0.40,
            evidence_count=1,
        ),
    ]

    ranked = ranker.rank(scores)

    assert [
        item.intent
        for item in ranked
    ] == [
        "question",
        "request",
        "explanation",
    ]


def test_top_k_returns_requested_items():

    ranker = IntentRanker()

    scores = [
        IntentScore(
            intent="question",
            score=0.90,
            evidence_count=1,
        ),
        IntentScore(
            intent="request",
            score=0.70,
            evidence_count=1,
        ),
        IntentScore(
            intent="statement",
            score=0.30,
            evidence_count=1,
        ),
    ]

    result = ranker.top_k(
        scores,
        2,
    )

    assert len(result) == 2

    assert [
        item.intent
        for item in result
    ] == [
        "question",
        "request",
    ]


def test_equal_scores_are_handled():

    ranker = IntentRanker()

    scores = [
        IntentScore(
            intent="question",
            score=0.8,
            evidence_count=1,
        ),
        IntentScore(
            intent="request",
            score=0.8,
            evidence_count=1,
        ),
    ]

    ranked = ranker.rank(scores)

    assert len(ranked) == 2
    assert ranked[0].score == ranked[1].score


def test_empty_scores_are_rejected():

    ranker = IntentRanker()

    with pytest.raises(ValueError):
        ranker.rank([])


def test_invalid_score_is_rejected():

    ranker = IntentRanker()

    with pytest.raises(TypeError):
        ranker.rank(["invalid"])


def test_invalid_k_is_rejected():

    ranker = IntentRanker()

    scores = [
        IntentScore(
            intent="question",
            score=0.8,
            evidence_count=1,
        )
    ]

    with pytest.raises(ValueError):
        ranker.top_k(scores, 0)