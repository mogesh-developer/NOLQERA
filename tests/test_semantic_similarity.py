import pytest

from nolqera.intelligence.semantic_similarity.similarity import (
    cosine_similarity,
)


def test_identical_vectors_have_maximum_similarity():

    assert cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ) == pytest.approx(1.0)


def test_orthogonal_vectors_have_zero_similarity():

    assert cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    ) == pytest.approx(0.0)


def test_similar_vectors_have_high_similarity():

    score = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 2.8],
    )

    assert score > 0.99


def test_zero_vector_returns_zero():

    assert cosine_similarity(
        [0.0, 0.0],
        [1.0, 2.0],
    ) == 0.0


def test_different_dimensions_are_rejected():

    with pytest.raises(ValueError):
        cosine_similarity(
            [1.0, 2.0],
            [1.0],
        )


def test_empty_vectors_are_rejected():

    with pytest.raises(ValueError):
        cosine_similarity([], [])


def test_non_numeric_values_are_rejected():

    with pytest.raises(TypeError):
        cosine_similarity(
            [1.0, "bad"],
            [1.0, 2.0],
        )
