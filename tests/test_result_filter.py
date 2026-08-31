import pytest

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)
from nolqera.intelligence.retrieval_quality.result_filter import (
    ResultFilter,
)


@pytest.fixture
def filter():

    return ResultFilter()


@pytest.fixture
def results():

    return [
        SemanticSearchResult(
            text="FastAPI Python backend",
            score=0.95,
            index=0,
        ),
        SemanticSearchResult(
            text="Python machine learning",
            score=0.80,
            index=1,
        ),
        SemanticSearchResult(
            text="React frontend",
            score=0.45,
            index=2,
        ),
        SemanticSearchResult(
            text="MongoDB database",
            score=0.20,
            index=3,
        ),
    ]


def test_filter_returns_results(filter, results):

    filtered = filter.filter(
        results,
        min_score=0.50,
    )

    assert isinstance(filtered, list)


def test_filter_removes_results_below_threshold(
    filter,
    results,
):

    filtered = filter.filter(
        results,
        min_score=0.50,
    )

    assert len(filtered) == 2

    assert filtered[0].text == (
        "FastAPI Python backend"
    )

    assert filtered[1].text == (
        "Python machine learning"
    )


def test_filter_keeps_result_at_threshold(
    filter,
):

    results = [
        SemanticSearchResult(
            text="Exact threshold",
            score=0.50,
            index=0,
        ),
    ]

    filtered = filter.filter(
        results,
        min_score=0.50,
    )

    assert len(filtered) == 1


def test_filter_removes_result_just_below_threshold(
    filter,
):

    results = [
        SemanticSearchResult(
            text="Below threshold",
            score=0.49,
            index=0,
        ),
    ]

    filtered = filter.filter(
        results,
        min_score=0.50,
    )

    assert filtered == []


def test_filter_preserves_result_order(
    filter,
    results,
):

    filtered = filter.filter(
        results,
        min_score=0.20,
    )

    assert [result.index for result in filtered] == [
        0,
        1,
        2,
        3,
    ]


def test_filter_preserves_result_objects(
    filter,
    results,
):

    filtered = filter.filter(
        results,
        min_score=0.80,
    )

    assert filtered[0] is results[0]
    assert filtered[1] is results[1]


def test_filter_with_zero_threshold_keeps_all(
    filter,
    results,
):

    filtered = filter.filter(
        results,
        min_score=0.0,
    )

    assert len(filtered) == len(results)


def test_filter_with_one_threshold_keeps_only_perfect_match(
    filter,
):

    results = [
        SemanticSearchResult(
            text="Perfect",
            score=1.0,
            index=0,
        ),
        SemanticSearchResult(
            text="Almost perfect",
            score=0.99,
            index=1,
        ),
    ]

    filtered = filter.filter(
        results,
        min_score=1.0,
    )

    assert len(filtered) == 1
    assert filtered[0].text == "Perfect"


def test_filter_empty_results_returns_empty_list(
    filter,
):

    filtered = filter.filter(
        [],
        min_score=0.50,
    )

    assert filtered == []


def test_filter_rejects_non_list(filter):

    with pytest.raises(TypeError):

        filter.filter(
            "invalid",
            min_score=0.50,
        )


def test_filter_rejects_invalid_min_score(
    filter,
    results,
):

    with pytest.raises(TypeError):

        filter.filter(
            results,
            min_score="0.50",
        )


def test_filter_rejects_out_of_range_min_score(
    filter,
    results,
):

    with pytest.raises(ValueError):

        filter.filter(
            results,
            min_score=1.5,
        )