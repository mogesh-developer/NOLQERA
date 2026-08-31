import pytest

from nolqera.intelligence.retrieval_quality.deduplication import (
    deduplicate_results,
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


def test_deduplication_returns_results():

    results = [
        make_result(
            "FastAPI is a backend framework",
            0.95,
            0,
        ),
        make_result(
            "Python is used for machine learning",
            0.80,
            1,
        ),
    ]

    deduplicated = deduplicate_results(results)

    assert isinstance(deduplicated, list)
    assert len(deduplicated) == 2


def test_deduplication_removes_exact_duplicates():

    results = [
        make_result(
            "FastAPI is a backend framework",
            0.95,
            0,
        ),
        make_result(
            "FastAPI is a backend framework",
            0.90,
            1,
        ),
        make_result(
            "Python is used for machine learning",
            0.80,
            2,
        ),
    ]

    deduplicated = deduplicate_results(results)

    assert len(deduplicated) == 2

    assert [
        result.text
        for result in deduplicated
    ] == [
        "FastAPI is a backend framework",
        "Python is used for machine learning",
    ]


def test_deduplication_keeps_first_occurrence():

    first = make_result(
        "FastAPI backend framework",
        0.95,
        0,
    )

    duplicate = make_result(
        "FastAPI backend framework",
        0.70,
        1,
    )

    deduplicated = deduplicate_results(
        [first, duplicate]
    )

    assert len(deduplicated) == 1
    assert deduplicated[0] is first


def test_deduplication_preserves_order():

    results = [
        make_result("Python backend", 0.95, 0),
        make_result("Machine learning", 0.90, 1),
        make_result("Python backend", 0.85, 2),
        make_result("FastAPI framework", 0.80, 3),
    ]

    deduplicated = deduplicate_results(results)

    assert [
        result.text
        for result in deduplicated
    ] == [
        "Python backend",
        "Machine learning",
        "FastAPI framework",
    ]


def test_deduplication_preserves_result_objects():

    first = make_result(
        "Python backend",
        0.95,
        0,
    )

    second = make_result(
        "FastAPI framework",
        0.90,
        1,
    )

    results = [first, second]

    deduplicated = deduplicate_results(results)

    assert deduplicated[0] is first
    assert deduplicated[1] is second


def test_deduplication_keeps_case_different_text():

    results = [
        make_result(
            "FastAPI backend",
            0.95,
            0,
        ),
        make_result(
            "fastapi backend",
            0.90,
            1,
        ),
    ]

    deduplicated = deduplicate_results(results)

    assert len(deduplicated) == 2


def test_deduplication_empty_results():

    results = []

    deduplicated = deduplicate_results(results)

    assert deduplicated == []


def test_deduplication_all_duplicates():

    results = [
        make_result("Python backend", 0.95, 0),
        make_result("Python backend", 0.90, 1),
        make_result("Python backend", 0.85, 2),
    ]

    deduplicated = deduplicate_results(results)

    assert len(deduplicated) == 1
    assert deduplicated[0].text == "Python backend"


def test_deduplication_rejects_non_list():

    with pytest.raises(TypeError):
        deduplicate_results(None)


def test_deduplication_rejects_invalid_result_type():

    results = [
        make_result(
            "Python backend",
            0.95,
            0,
        ),
        "invalid result",
    ]

    with pytest.raises(TypeError):
        deduplicate_results(results)