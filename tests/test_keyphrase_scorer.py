import pytest

from nolqera.intelligence.keyphrase.scorer import (
    KeyphraseScorer,
)


def test_high_tfidf_produces_high_score():
    scorer = KeyphraseScorer()

    score = scorer.score(
        tfidf_score=1.0,
        frequency_score=0.0,
        length_score=0.0,
    )

    assert score == pytest.approx(0.6)


def test_all_strong_signals_produce_maximum_score():
    scorer = KeyphraseScorer()

    score = scorer.score(
        tfidf_score=1.0,
        frequency_score=1.0,
        length_score=1.0,
    )

    assert score == pytest.approx(1.0)


def test_score_is_between_zero_and_one():
    scorer = KeyphraseScorer()

    score = scorer.score(
        tfidf_score=0.8,
        frequency_score=0.6,
        length_score=0.7,
    )

    assert 0.0 <= score <= 1.0


def test_weights_are_normalized():
    scorer = KeyphraseScorer(
        tfidf_weight=3.0,
        frequency_weight=1.0,
        length_weight=1.0,
    )

    score = scorer.score(
        tfidf_score=1.0,
        frequency_score=0.0,
        length_score=0.0,
    )

    assert score == pytest.approx(0.6)


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        KeyphraseScorer(
            tfidf_weight=-1.0,
        )


def test_zero_weights_are_rejected():
    with pytest.raises(ValueError):
        KeyphraseScorer(
            tfidf_weight=0.0,
            frequency_weight=0.0,
            length_weight=0.0,
        )


def test_invalid_signal_is_rejected():
    scorer = KeyphraseScorer()

    with pytest.raises(ValueError):
        scorer.score(
            tfidf_score=1.2,
            frequency_score=0.5,
            length_score=0.5,
        )