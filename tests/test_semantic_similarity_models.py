import pytest

from nolqera.intelligence.semantic_similarity.models import (
    SemanticSimilarityResult,
)


def test_result_stores_similarity():

    result = SemanticSimilarityResult(
        text_a="car",
        text_b="vehicle",
        score=0.9,
    )

    assert result.text_a == "car"
    assert result.text_b == "vehicle"
    assert result.score == 0.9


def test_result_to_dict():

    result = SemanticSimilarityResult(
        text_a="car",
        text_b="vehicle",
        score=0.9,
    )

    assert result.to_dict() == {
        "text_a": "car",
        "text_b": "vehicle",
        "score": 0.9,
    }


def test_invalid_score_is_rejected():

    with pytest.raises(ValueError):

        SemanticSimilarityResult(
            text_a="a",
            text_b="b",
            score=1.5,
        )


def test_empty_text_is_rejected():

    with pytest.raises(ValueError):

        SemanticSimilarityResult(
            text_a="",
            text_b="b",
            score=0.5,
        )
