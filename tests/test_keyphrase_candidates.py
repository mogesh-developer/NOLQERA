import pytest

from nolqera.intelligence.keyphrase.candidates import (
    KeyphraseCandidateExtractor,
)


def test_extracts_unigrams_bigrams_and_trigrams():
    extractor = KeyphraseCandidateExtractor()

    tokens = [
        "fastapi",
        "rest",
        "api",
    ]

    candidates = extractor.extract(tokens)

    assert "fastapi" in candidates
    assert "rest" in candidates
    assert "api" in candidates

    assert "fastapi rest" in candidates
    assert "rest api" in candidates

    assert "fastapi rest api" in candidates


def test_candidates_are_unique():
    extractor = KeyphraseCandidateExtractor(
        min_n=1,
        max_n=2,
    )

    tokens = [
        "python",
        "api",
        "python",
    ]

    candidates = extractor.extract(tokens)

    assert len(candidates) == len(set(candidates))


def test_custom_ngram_range():
    extractor = KeyphraseCandidateExtractor(
        min_n=2,
        max_n=2,
    )

    tokens = [
        "fastapi",
        "rest",
        "api",
    ]

    candidates = extractor.extract(tokens)

    assert candidates == [
        "fastapi rest",
        "rest api",
    ]


def test_empty_tokens_are_rejected():
    extractor = KeyphraseCandidateExtractor()

    with pytest.raises(ValueError):
        extractor.extract([])


def test_invalid_tokens_are_rejected():
    extractor = KeyphraseCandidateExtractor()

    with pytest.raises(TypeError):
        extractor.extract([
            "fastapi",
            123,
        ])


def test_invalid_ngram_range_is_rejected():
    with pytest.raises(ValueError):
        KeyphraseCandidateExtractor(
            min_n=3,
            max_n=2,
        )