import pytest

from nolqera.intelligence.context_optimization.importance_separation import (
    ImportanceSeparator,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def make_result(
    text: str,
    score: float,
    index: int,
):
    return SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )


def test_separator_accepts_threshold():

    separator = ImportanceSeparator(
        importance_threshold=0.5
    )

    assert separator.importance_threshold == 0.5


def test_separator_rejects_non_numeric_threshold():

    with pytest.raises(TypeError):
        ImportanceSeparator("0.5")


def test_separator_rejects_negative_threshold():

    with pytest.raises(ValueError):
        ImportanceSeparator(-0.1)


def test_separator_rejects_threshold_above_one():

    with pytest.raises(ValueError):
        ImportanceSeparator(1.1)


def test_separate_returns_two_lists():

    separator = ImportanceSeparator()

    results = [
        make_result(
            "FastAPI authentication",
            0.9,
            0,
        ),
        make_result(
            "Python framework",
            0.3,
            1,
        ),
    ]

    important, unnecessary = separator.separate(results)

    assert isinstance(important, list)
    assert isinstance(unnecessary, list)


def test_high_score_result_is_important():

    separator = ImportanceSeparator(
        importance_threshold=0.5
    )

    result = make_result(
        "FastAPI authentication",
        0.8,
        0,
    )

    important, unnecessary = separator.separate(
        [result]
    )

    assert important == [result]
    assert unnecessary == []


def test_low_score_result_is_unnecessary():

    separator = ImportanceSeparator(
        importance_threshold=0.5
    )

    result = make_result(
        "FastAPI creator information",
        0.2,
        0,
    )

    important, unnecessary = separator.separate(
        [result]
    )

    assert important == []
    assert unnecessary == [result]


def test_result_at_threshold_is_important():

    separator = ImportanceSeparator(
        importance_threshold=0.5
    )

    result = make_result(
        "OAuth2 authentication",
        0.5,
        0,
    )

    important, unnecessary = separator.separate(
        [result]
    )

    assert important == [result]
    assert unnecessary == []


def test_results_are_correctly_separated():

    separator = ImportanceSeparator(
        importance_threshold=0.5
    )

    results = [
        make_result("JWT authentication", 0.9, 0),
        make_result("Python framework", 0.4, 1),
        make_result("OAuth2 bearer token", 0.8, 2),
        make_result("Creator information", 0.2, 3),
    ]

    important, unnecessary = separator.separate(
        results
    )

    assert [result.text for result in important] == [
        "JWT authentication",
        "OAuth2 bearer token",
    ]

    assert [result.text for result in unnecessary] == [
        "Python framework",
        "Creator information",
    ]


def test_separation_preserves_order():

    separator = ImportanceSeparator(
        importance_threshold=0.5
    )

    results = [
        make_result("First", 0.9, 0),
        make_result("Second", 0.2, 1),
        make_result("Third", 0.8, 2),
        make_result("Fourth", 0.1, 3),
    ]

    important, unnecessary = separator.separate(
        results
    )

    assert [result.index for result in important] == [
        0,
        2,
    ]

    assert [result.index for result in unnecessary] == [
        1,
        3,
    ]


def test_separation_preserves_result_objects():

    separator = ImportanceSeparator()

    important_result = make_result(
        "Important information",
        0.9,
        0,
    )

    unnecessary_result = make_result(
        "Unnecessary information",
        0.1,
        1,
    )

    important, unnecessary = separator.separate(
        [
            important_result,
            unnecessary_result,
        ]
    )

    assert important[0] is important_result
    assert unnecessary[0] is unnecessary_result


def test_empty_results_return_two_empty_lists():

    separator = ImportanceSeparator()

    important, unnecessary = separator.separate([])

    assert important == []
    assert unnecessary == []


def test_rejects_non_list_results():

    separator = ImportanceSeparator()

    with pytest.raises(TypeError):
        separator.separate(None)


def test_rejects_invalid_result_type():

    separator = ImportanceSeparator()

    with pytest.raises(TypeError):
        separator.separate(
            ["FastAPI authentication"]
        )


def test_threshold_controls_separation():

    result = make_result(
        "FastAPI authentication",
        0.6,
        0,
    )

    separator = ImportanceSeparator(
        importance_threshold=0.5
    )

    important, unnecessary = separator.separate(
        [result]
    )

    assert important == [result]
    assert unnecessary == []

    separator = ImportanceSeparator(
        importance_threshold=0.7
    )

    important, unnecessary = separator.separate(
        [result]
    )

    assert important == []
    assert unnecessary == [result]