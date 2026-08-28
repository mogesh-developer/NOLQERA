import pytest

from nolqera.intelligence.relevance.similarity import cosine_similarity


def test_identical_vectors_have_maximum_similarity():
    vector = [1.0, 2.0, 3.0]

    result = cosine_similarity(vector, vector)

    assert result == pytest.approx(1.0)


def test_orthogonal_vectors_have_zero_similarity():
    vector_a = [1.0, 0.0]
    vector_b = [0.0, 1.0]

    result = cosine_similarity(vector_a, vector_b)

    assert result == pytest.approx(0.0)


def test_similar_vectors_have_high_similarity():
    vector_a = [1.0, 2.0, 3.0]
    vector_b = [1.1, 2.1, 3.1]

    result = cosine_similarity(vector_a, vector_b)

    assert result > 0.99


def test_zero_vector_returns_zero():
    vector_a = [0.0, 0.0]
    vector_b = [1.0, 2.0]

    result = cosine_similarity(vector_a, vector_b)

    assert result == 0.0


def test_different_dimensions_are_rejected():
    with pytest.raises(ValueError):
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_empty_vectors_are_rejected():
    with pytest.raises(ValueError):
        cosine_similarity([], [])


def test_non_numeric_values_are_rejected():
    with pytest.raises(TypeError):
        cosine_similarity(
            [1.0, "invalid"],
            [1.0, 2.0],
        )