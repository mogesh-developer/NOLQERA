import pytest

from nolqera.intelligence.relevance.scorer import (
    RelevanceScorer,
)


def test_high_similarity_is_relevant():
    scorer = RelevanceScorer()

    result = scorer.score(0.90)

    assert result.score == 0.90
    assert result.label == "relevant"


def test_medium_similarity_is_weak():
    scorer = RelevanceScorer()

    result = scorer.score(0.35)

    assert result.score == 0.35
    assert result.label == "weak"


def test_low_similarity_is_irrelevant():
    scorer = RelevanceScorer()

    result = scorer.score(0.10)

    assert result.score == 0.10
    assert result.label == "irrelevant"


def test_thresholds_are_configurable():
    scorer = RelevanceScorer(
        relevant_threshold=0.70,
        weak_threshold=0.30,
    )

    assert scorer.score(0.70).label == "relevant"
    assert scorer.score(0.50).label == "weak"
    assert scorer.score(0.20).label == "irrelevant"


def test_invalid_similarity_is_rejected():
    scorer = RelevanceScorer()

    with pytest.raises(ValueError):
        scorer.score(1.1)

    with pytest.raises(ValueError):
        scorer.score(-0.1)


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        RelevanceScorer(
            relevant_threshold=0.20,
            weak_threshold=0.50,
        )