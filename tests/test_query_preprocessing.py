import pytest

from nolqera.intelligence.retrieval_quality.query_preprocessing import (
    QueryPreprocessor,
)


@pytest.fixture
def preprocessor():
    return QueryPreprocessor()


def test_preprocessor_accepts_query(preprocessor):

    result = preprocessor.preprocess(
        "python backend"
    )

    assert result == "python backend"


def test_preprocessor_removes_leading_whitespace(
    preprocessor,
):

    result = preprocessor.preprocess(
        "   python backend"
    )

    assert result == "python backend"


def test_preprocessor_removes_trailing_whitespace(
    preprocessor,
):

    result = preprocessor.preprocess(
        "python backend   "
    )

    assert result == "python backend"


def test_preprocessor_collapses_multiple_spaces(
    preprocessor,
):

    result = preprocessor.preprocess(
        "python    backend     api"
    )

    assert result == "python backend api"


def test_preprocessor_normalizes_mixed_whitespace(
    preprocessor,
):

    result = preprocessor.preprocess(
        "  python\t\tbackend\napi  "
    )

    assert result == "python backend api"


def test_preprocessor_preserves_query_content(
    preprocessor,
):

    query = "python backend api"

    result = preprocessor.preprocess(query)

    assert result == query


def test_preprocessor_rejects_non_string(
    preprocessor,
):

    with pytest.raises(TypeError):
        preprocessor.preprocess(123)


def test_preprocessor_rejects_empty_string(
    preprocessor,
):

    with pytest.raises(ValueError):
        preprocessor.preprocess("")


def test_preprocessor_rejects_whitespace_only_query(
    preprocessor,
):

    with pytest.raises(ValueError):
        preprocessor.preprocess("   \t\n  ")