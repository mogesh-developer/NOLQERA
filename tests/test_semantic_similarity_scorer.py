import pytest

from nolqera.intelligence.semantic_similarity.scorer import (
    SemanticSimilarityScorer,
)


def test_high_similarity():

    scorer = SemanticSimilarityScorer()

    result = scorer.score(0.9)

    assert result.label == "high"
    assert result.score == 0.9


def test_medium_similarity():

    scorer = SemanticSimilarityScorer()

    result = scorer.score(0.6)

    assert result.label == "medium"


def test_low_similarity():

    scorer = SemanticSimilarityScorer()

    result = scorer.score(0.2)

    assert result.label == "low"


def test_thresholds_are_configurable():

    scorer = SemanticSimilarityScorer(
        high_threshold=0.8,
        medium_threshold=0.5,
    )

    assert scorer.score(0.8).label == "high"
    assert scorer.score(0.5).label == "medium"


def test_invalid_similarity_is_rejected():

    scorer = SemanticSimilarityScorer()

    with pytest.raises(ValueError):
        scorer.score(1.5)


def test_invalid_thresholds_are_rejected():

    with pytest.raises(ValueError):

        SemanticSimilarityScorer(
            high_threshold=0.4,
            medium_threshold=0.6,
        )
