import pytest

from nolqera.intelligence.pipeline.sentence_segmenter import (
    SentenceSegmenter,
)


def test_segmenter_returns_list():
    segmenter = SentenceSegmenter()

    result = segmenter.segment(
        "FastAPI is a Python framework. It is used for APIs."
    )

    assert isinstance(result, list)


def test_segmenter_returns_exact_sentences():
    segmenter = SentenceSegmenter()

    result = segmenter.segment(
        "FastAPI is a Python framework. It is used for APIs."
    )

    assert result == [
        "FastAPI is a Python framework.",
        "It is used for APIs.",
    ]


def test_segmenter_supports_question_mark():
    segmenter = SentenceSegmenter()

    result = segmenter.segment(
        "What is FastAPI? It is a Python framework."
    )

    assert result == [
        "What is FastAPI?",
        "It is a Python framework.",
    ]


def test_segmenter_supports_exclamation_mark():
    segmenter = SentenceSegmenter()

    result = segmenter.segment(
        "FastAPI is fast! It is useful."
    )

    assert result == [
        "FastAPI is fast!",
        "It is useful.",
    ]


def test_segmenter_preserves_sentence_order():
    segmenter = SentenceSegmenter()

    result = segmenter.segment(
        "First sentence. Second sentence. Third sentence."
    )

    assert result == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]


def test_segmenter_preserves_trailing_text():
    segmenter = SentenceSegmenter()

    result = segmenter.segment(
        "FastAPI is a framework. MongoDB stores data"
    )

    assert result == [
        "FastAPI is a framework.",
        "MongoDB stores data",
    ]


def test_segmenter_handles_single_sentence():
    segmenter = SentenceSegmenter()

    result = segmenter.segment(
        "FastAPI is a Python framework."
    )

    assert result == [
        "FastAPI is a Python framework.",
    ]


def test_segmenter_rejects_non_string():
    segmenter = SentenceSegmenter()

    with pytest.raises(TypeError):
        segmenter.segment(123)


def test_segmenter_rejects_empty_string():
    segmenter = SentenceSegmenter()

    with pytest.raises(ValueError):
        segmenter.segment("")


def test_segmenter_rejects_whitespace_only_text():
    segmenter = SentenceSegmenter()

    with pytest.raises(ValueError):
        segmenter.segment("   ")