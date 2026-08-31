import pytest

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)

from nolqera.intelligence.context_optimization.near_duplicate import (
    is_near_duplicate,
    remove_near_duplicates,
)


def _result(
    text: str,
    score: float,
    index: int,
) -> SemanticSearchResult:

    return SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )


def test_near_duplicate_detects_similar_text():

    first = (
        "FastAPI is a Python backend framework."
    )

    second = (
        "FastAPI is a backend framework "
        "written in Python."
    )

    assert is_near_duplicate(
        first,
        second,
        similarity_threshold=0.6,
    )


def test_near_duplicate_rejects_different_text():

    first = (
        "FastAPI is a Python backend framework."
    )

    second = (
        "MongoDB is a NoSQL database."
    )

    assert not is_near_duplicate(
        first,
        second,
        similarity_threshold=0.6,
    )


def test_near_duplicate_ignores_case():

    first = "FastAPI Python Backend"

    second = "fastapi python backend"

    assert is_near_duplicate(
        first,
        second,
    )


def test_near_duplicate_ignores_punctuation():

    first = (
        "FastAPI is a Python framework."
    )

    second = (
        "FastAPI is a Python framework!"
    )

    assert is_near_duplicate(
        first,
        second,
    )


def test_identical_text_is_near_duplicate():

    text = (
        "FastAPI is a Python framework."
    )

    assert is_near_duplicate(
        text,
        text,
    )


def test_threshold_controls_detection():

    first = (
        "FastAPI is a Python backend framework."
    )

    second = (
        "FastAPI provides Python APIs."
    )

    assert not is_near_duplicate(
        first,
        second,
        similarity_threshold=0.9,
    )


def test_remove_near_duplicates():

    results = [
        _result(
            "FastAPI is a Python backend framework.",
            0.95,
            0,
        ),
        _result(
            "FastAPI is a backend framework written in Python.",
            0.90,
            1,
        ),
        _result(
            "MongoDB is a NoSQL database.",
            0.80,
            2,
        ),
    ]

    filtered = remove_near_duplicates(
        results,
        similarity_threshold=0.6,
    )

    assert len(filtered) == 2

    assert filtered[0].text == (
        "FastAPI is a Python backend framework."
    )

    assert filtered[1].text == (
        "MongoDB is a NoSQL database."
    )


def test_remove_near_duplicates_keeps_first_occurrence():

    first = _result(
        "Python is used for machine learning.",
        0.95,
        0,
    )

    second = _result(
        "Python is used in machine learning.",
        0.80,
        1,
    )

    results = [
        first,
        second,
    ]

    filtered = remove_near_duplicates(
        results,
        similarity_threshold=0.7,
    )

    assert filtered == [first]


def test_remove_near_duplicates_preserves_order():

    results = [
        _result(
            "FastAPI Python backend.",
            0.9,
            0,
        ),
        _result(
            "MongoDB database storage.",
            0.8,
            1,
        ),
        _result(
            "React frontend JavaScript.",
            0.7,
            2,
        ),
    ]

    filtered = remove_near_duplicates(
        results,
        similarity_threshold=0.9,
    )

    assert [result.index for result in filtered] == [
        0,
        1,
        2,
    ]


def test_remove_near_duplicates_preserves_objects():

    result = _result(
        "FastAPI Python backend.",
        0.9,
        0,
    )

    filtered = remove_near_duplicates(
        [result],
    )

    assert filtered[0] is result


def test_empty_results_return_empty_list():

    assert remove_near_duplicates([]) == []


def test_rejects_non_list():

    with pytest.raises(TypeError):
        remove_near_duplicates("invalid")


def test_rejects_invalid_result_type():

    with pytest.raises(TypeError):
        remove_near_duplicates(
            ["invalid"]
        )


def test_rejects_invalid_threshold():

    with pytest.raises(ValueError):
        remove_near_duplicates(
            [],
            similarity_threshold=1.5,
        )


def test_rejects_negative_threshold():

    with pytest.raises(ValueError):
        remove_near_duplicates(
            [],
            similarity_threshold=-0.1,
        )


def test_is_near_duplicate_rejects_invalid_first():

    with pytest.raises(TypeError):
        is_near_duplicate(
            123,
            "text",
        )


def test_is_near_duplicate_rejects_empty_text():

    with pytest.raises(ValueError):
        is_near_duplicate(
            "",
            "text",
        )