import pytest

from nolqera.intelligence.importance.scorer import (
    ImportanceScorer,
)


def test_importance_score_combines_signals():
    scorer = ImportanceScorer()

    score = scorer.score(
        tfidf_score=1.0,
        position_score=0.0,
        density_score=0.0,
    )

    assert score == pytest.approx(0.6)


def test_importance_score_is_between_zero_and_one():
    scorer = ImportanceScorer()

    score = scorer.score(
        tfidf_score=0.8,
        position_score=0.6,
        density_score=0.7,
    )

    assert 0.0 <= score <= 1.0


def test_weights_are_normalized():
    scorer = ImportanceScorer(
        tfidf_weight=3.0,
        position_weight=1.0,
        density_weight=1.0,
    )

    score = scorer.score(
        tfidf_score=1.0,
        position_score=0.0,
        density_score=0.0,
    )

    assert score == pytest.approx(0.6)


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        ImportanceScorer(
            tfidf_weight=-1.0,
        )


def test_zero_weights_are_rejected():
    with pytest.raises(ValueError):
        ImportanceScorer(
            tfidf_weight=0.0,
            position_weight=0.0,
            density_weight=0.0,
        )


def test_invalid_signal_is_rejected():
    scorer = ImportanceScorer()

    with pytest.raises(ValueError):
        scorer.score(
            tfidf_score=1.5,
            position_score=0.5,
            density_score=0.5,
        )