import pytest

from nolqera.intelligence.retrieval_quality.diversity import (
    diversify_results,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def make_result(
    text: str,
    score: float,
    index: int,
) -> SemanticSearchResult:
    return SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )


def test_diversity_reduces_similar_result_variants():

    results = [
        make_result(
            "FastAPI Python backend framework",
            0.95,
            0,
        ),
        make_result(
            "FastAPI Python backend framework APIs",
            0.93,
            1,
        ),
        make_result(
            "FastAPI Python backend framework APIs development",
            0.91,
            2,
        ),
        make_result(
            "Django web framework",
            0.86,
            3,
        ),
        make_result(
            "Python machine learning",
            0.82,
            4,
        ),
    ]

    diversified = diversify_results(
        results,
        similarity_threshold=0.5,
    )

    assert [
        result.text
        for result in diversified
    ] == [
        "FastAPI Python backend framework",
        "Django web framework",
        "Python machine learning",
    ]

    results = [
        make_result(
            "FastAPI Python backend framework",
            0.95,
            0,
        ),
        make_result(
            "Django Python web framework",
            0.85,
            1,
        ),
    ]

    diversified = diversify_results(results)

    assert isinstance(diversified, list)
    assert len(diversified) == 2


def test_diversity_removes_highly_similar_results():

    results = [
        make_result(
            "FastAPI Python backend framework",
            0.95,
            0,
        ),
        make_result(
            "FastAPI Python backend framework",
            0.90,
            1,
        ),
        make_result(
            "Django Python web framework",
            0.85,
            2,
        ),
    ]

    diversified = diversify_results(
        results,
        similarity_threshold=0.8,
    )

    assert len(diversified) == 2

    assert [
        result.text
        for result in diversified
    ] == [
        "FastAPI Python backend framework",
        "Django Python web framework",
    ]


def test_diversity_keeps_first_result():

    first = make_result(
        "FastAPI Python backend framework",
        0.95,
        0,
    )

    similar = make_result(
        "FastAPI Python backend framework",
        0.70,
        1,
    )

    diversified = diversify_results(
        [first, similar],
        similarity_threshold=0.8,
    )

    assert len(diversified) == 1
    assert diversified[0] is first


def test_diversity_preserves_result_order():

    results = [
        make_result(
            "Python backend framework",
            0.95,
            0,
        ),
        make_result(
            "Machine learning algorithms",
            0.90,
            1,
        ),
        make_result(
            "Python backend framework",
            0.85,
            2,
        ),
        make_result(
            "Database storage system",
            0.80,
            3,
        ),
    ]

    diversified = diversify_results(
        results,
        similarity_threshold=0.8,
    )

    assert [
        result.text
        for result in diversified
    ] == [
        "Python backend framework",
        "Machine learning algorithms",
        "Database storage system",
    ]


def test_diversity_keeps_different_results():

    results = [
        make_result(
            "FastAPI Python backend",
            0.95,
            0,
        ),
        make_result(
            "MongoDB database storage",
            0.90,
            1,
        ),
        make_result(
            "React frontend application",
            0.85,
            2,
        ),
    ]

    diversified = diversify_results(results)

    assert len(diversified) == 3


def test_diversity_preserves_result_objects():

    first = make_result(
        "FastAPI backend",
        0.95,
        0,
    )

    second = make_result(
        "MongoDB database",
        0.90,
        1,
    )

    diversified = diversify_results(
        [first, second]
    )

    assert diversified[0] is first
    assert diversified[1] is second


def test_zero_threshold_keeps_only_first_result():

    results = [
        make_result(
            "FastAPI backend",
            0.95,
            0,
        ),
        make_result(
            "MongoDB database",
            0.90,
            1,
        ),
    ]

    diversified = diversify_results(
        results,
        similarity_threshold=0.0,
    )

    assert len(diversified) == 1
    assert diversified[0].text == "FastAPI backend"


def test_one_threshold_allows_different_results():

    results = [
        make_result(
            "FastAPI backend",
            0.95,
            0,
        ),
        make_result(
            "MongoDB database",
            0.90,
            1,
        ),
    ]

    diversified = diversify_results(
        results,
        similarity_threshold=1.0,
    )

    assert len(diversified) == 2


def test_empty_results_return_empty_list():

    assert diversify_results([]) == []


def test_diversity_rejects_non_list():

    with pytest.raises(TypeError):
        diversify_results(None)


def test_diversity_rejects_invalid_threshold():

    with pytest.raises(TypeError):
        diversify_results(
            [],
            similarity_threshold="0.8",
        )


def test_diversity_rejects_threshold_below_zero():

    with pytest.raises(ValueError):
        diversify_results(
            [],
            similarity_threshold=-0.1,
        )


def test_diversity_rejects_threshold_above_one():

    with pytest.raises(ValueError):
        diversify_results(
            [],
            similarity_threshold=1.1,
        )


def test_diversity_rejects_invalid_result_type():

    results = [
        make_result(
            "FastAPI backend",
            0.95,
            0,
        ),
        "invalid",
    ]

    with pytest.raises(TypeError):
        diversify_results(results)