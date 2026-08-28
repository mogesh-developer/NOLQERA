import pytest

from nolqera.intelligence.keyphrase.ranking import (
    KeyphraseRanker,
)


def test_rank_orders_keyphrases_highest_first():
    ranker = KeyphraseRanker()

    results = ranker.rank({
        "fastapi": 0.91,
        "mongodb": 0.87,
        "rest api": 0.82,
        "application": 0.22,
    })

    assert [item.phrase for item in results] == [
        "fastapi",
        "mongodb",
        "rest api",
        "application",
    ]

    assert [item.rank for item in results] == [
        1, 2, 3, 4
    ]


def test_top_k_returns_requested_keyphrases():
    ranker = KeyphraseRanker()

    results = ranker.top_k(
        {
            "fastapi": 0.91,
            "mongodb": 0.87,
            "rest api": 0.82,
        },
        k=2,
    )

    assert len(results) == 2
    assert results[0].phrase == "fastapi"
    assert results[1].phrase == "mongodb"


def test_equal_scores_are_handled():
    ranker = KeyphraseRanker()

    results = ranker.rank({
        "fastapi": 0.8,
        "mongodb": 0.8,
        "rest": 0.4,
    })

    assert len(results) == 3
    assert results[0].score == 0.8
    assert results[1].score == 0.8


def test_empty_scores_are_rejected():
    ranker = KeyphraseRanker()

    with pytest.raises(ValueError):
        ranker.rank({})


def test_invalid_score_is_rejected():
    ranker = KeyphraseRanker()

    with pytest.raises(ValueError):
        ranker.rank({
            "fastapi": 1.5,
        })


def test_invalid_k_is_rejected():
    ranker = KeyphraseRanker()

    with pytest.raises(ValueError):
        ranker.top_k(
            {"fastapi": 0.8},
            0,
        )

def test_shorter_overlapping_phrase_is_removed():
    ranker = KeyphraseRanker()

    ranked = ranker.rank({
        "persistent data storage": 0.90,
        "persistent data": 0.70,
        "fastapi": 0.80,
    })

    filtered = ranker.remove_overlapping(ranked)

    phrases = [
        item.phrase
        for item in filtered
    ]

    assert phrases == [
        "persistent data storage",
        "fastapi",
    ]