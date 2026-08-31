import pytest

from nolqera.intelligence.retrieval_quality.score_normalization import (
    ScoreNormalizer,
    NormalizedRetrievalResult,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


@pytest.fixture
def results():

    return [
        SemanticSearchResult(
            index=0,
            text="fastapi python backend",
            score=0.4,
        ),
        SemanticSearchResult(
            index=1,
            text="python machine learning",
            score=0.6,
        ),
        SemanticSearchResult(
            index=2,
            text="mongodb database",
            score=0.8,
        ),
    ]


def test_normalizer_returns_results(results):

    normalizer = ScoreNormalizer()

    normalized = normalizer.normalize(results)

    assert len(normalized) == 3


def test_normalizer_returns_correct_result_type(
    results,
):

    normalizer = ScoreNormalizer()

    normalized = normalizer.normalize(results)

    assert all(
        isinstance(
            result,
            NormalizedRetrievalResult,
        )
        for result in normalized
    )


def test_min_max_normalization(results):

    normalizer = ScoreNormalizer()

    normalized = normalizer.normalize(results)

    scores = [
        result.score
        for result in normalized
    ]

    assert scores == pytest.approx([
        0.0,
        0.5,
        1.0,
    ])


def test_normalized_scores_are_between_zero_and_one(
    results,
):

    normalizer = ScoreNormalizer()

    normalized = normalizer.normalize(results)

    for result in normalized:
        assert 0.0 <= result.score <= 1.0


def test_highest_score_becomes_one(results):

    normalizer = ScoreNormalizer()

    normalized = normalizer.normalize(results)

    assert normalized[-1].score == 1.0


def test_lowest_score_becomes_zero(results):

    normalizer = ScoreNormalizer()

    normalized = normalizer.normalize(results)

    assert normalized[0].score == 0.0


def test_original_score_is_preserved(results):

    normalizer = ScoreNormalizer()

    normalized = normalizer.normalize(results)

    original_scores = [
        result.original_score
        for result in normalized
    ]

    assert original_scores == [
        0.4,
        0.6,
        0.8,
    ]


def test_result_metadata_is_preserved(results):

    normalizer = ScoreNormalizer()

    normalized = normalizer.normalize(results)

    assert normalized[0].index == 0
    assert normalized[0].text == (
        "fastapi python backend"
    )


def test_identical_scores_are_handled():

    results = [
        SemanticSearchResult(
            index=0,
            text="document one",
            score=0.5,
        ),
        SemanticSearchResult(
            index=1,
            text="document two",
            score=0.5,
        ),
    ]

    normalizer = ScoreNormalizer()

    normalized = normalizer.normalize(results)

    assert [
        result.score
        for result in normalized
    ] == [
        1.0,
        1.0,
    ]


def test_empty_results_return_empty_list():

    normalizer = ScoreNormalizer()

    assert normalizer.normalize([]) == []


def test_rejects_non_list():

    normalizer = ScoreNormalizer()

    with pytest.raises(TypeError):
        normalizer.normalize(None)


def test_rejects_invalid_result_type():

    normalizer = ScoreNormalizer()

    with pytest.raises(TypeError):
        normalizer.normalize(
            ["invalid"]
        )