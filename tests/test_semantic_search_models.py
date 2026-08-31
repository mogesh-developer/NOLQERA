import pytest

from nolqera.intelligence.semantic_search.models import SemanticSearchResult


def test_semantic_search_result_valid():
    result = SemanticSearchResult(
        text="hello world",
        score=0.85,
        index=42,
    )
    assert result.text == "hello world"
    assert result.score == 0.85
    assert result.index == 42
    assert result.to_dict() == {
        "text": "hello world",
        "score": 0.85,
        "index": 42,
    }


def test_text_must_be_string():
    with pytest.raises(TypeError, match="text must be a string"):
        SemanticSearchResult(text=123, score=0.5, index=0)


def test_text_cannot_be_empty():
    with pytest.raises(ValueError, match="text cannot be empty"):
        SemanticSearchResult(text="   ", score=0.5, index=0)


def test_score_must_be_numeric():
    with pytest.raises(TypeError, match="score must be numeric"):
        SemanticSearchResult(text="hello", score="0.5", index=0)


def test_score_range_validation():
    with pytest.raises(ValueError, match="score must be between 0 and 1"):
        SemanticSearchResult(text="hello", score=-0.1, index=0)

    with pytest.raises(ValueError, match="score must be between 0 and 1"):
        SemanticSearchResult(text="hello", score=1.1, index=0)


def test_index_must_be_integer():
    with pytest.raises(TypeError, match="index must be an integer"):
        SemanticSearchResult(text="hello", score=0.5, index="0")


def test_index_cannot_be_negative():
    with pytest.raises(ValueError, match="index cannot be negative"):
        SemanticSearchResult(text="hello", score=0.5, index=-1)
