import pytest

from nolqera.intelligence.retrieval_quality.reranking import (
    rerank_results,
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


def test_reranker_returns_results():

    results = [
        make_result(
            "Python backend framework",
            0.8,
            0,
        ),
        make_result(
            "Machine learning algorithms",
            0.7,
            1,
        ),
    ]

    reranked = rerank_results(
        "Python backend",
        results,
    )

    assert isinstance(reranked, list)
    assert len(reranked) == 2


def test_reranker_prioritizes_keyword_relevance():

    results = [
        make_result(
            "Database storage system",
            0.95,
            0,
        ),
        make_result(
            "Python backend framework",
            0.80,
            1,
        ),
    ]

    reranked = rerank_results(
        "Python backend",
        results,
        relevance_weight=0.5,
        keyword_weight=0.5,
    )

    assert reranked[0].text == (
        "Python backend framework"
    )


def test_reranker_preserves_result_objects():

    first = make_result(
        "Python backend",
        0.9,
        0,
    )

    second = make_result(
        "Database storage",
        0.8,
        1,
    )

    reranked = rerank_results(
        "Python",
        [first, second],
    )

    assert first in reranked
    assert second in reranked


def test_reranker_orders_by_final_score():

    results = [
        make_result(
            "Python backend",
            0.60,
            0,
        ),
        make_result(
            "Database storage",
            0.90,
            1,
        ),
    ]

    reranked = rerank_results(
        "Python backend",
        results,
        relevance_weight=0.5,
        keyword_weight=0.5,
    )

    assert reranked[0].text == (
        "Python backend"
    )


def test_reranker_preserves_all_results():

    results = [
        make_result("Python backend", 0.9, 0),
        make_result("Database storage", 0.8, 1),
        make_result("React frontend", 0.7, 2),
    ]

    reranked = rerank_results(
        "Python",
        results,
    )

    assert len(reranked) == len(results)

    assert {
        result.index
        for result in reranked
    } == {0, 1, 2}


def test_reranker_empty_results():

    assert rerank_results(
        "Python",
        [],
    ) == []


def test_reranker_rejects_non_string_query():

    with pytest.raises(TypeError):
        rerank_results(
            None,
            [],
        )


def test_reranker_rejects_empty_query():

    with pytest.raises(ValueError):
        rerank_results(
            "   ",
            [],
        )


def test_reranker_rejects_non_list_results():

    with pytest.raises(TypeError):
        rerank_results(
            "Python",
            None,
        )


def test_reranker_rejects_invalid_result_type():

    with pytest.raises(TypeError):
        rerank_results(
            "Python",
            ["invalid"],
        )


def test_reranker_rejects_invalid_relevance_weight():

    with pytest.raises(ValueError):
        rerank_results(
            "Python",
            [],
            relevance_weight=1.2,
            keyword_weight=-0.2,
        )


def test_reranker_rejects_invalid_keyword_weight():

    with pytest.raises(ValueError):
        rerank_results(
            "Python",
            [],
            relevance_weight=0.5,
            keyword_weight=1.2,
        )


def test_reranker_requires_weights_to_sum_to_one():

    with pytest.raises(ValueError):
        rerank_results(
            "Python",
            [],
            relevance_weight=0.6,
            keyword_weight=0.6,
        )


def test_reranker_accepts_zero_keyword_weight():

    results = [
        make_result(
            "Database storage",
            0.9,
            0,
        ),
        make_result(
            "Python backend",
            0.8,
            1,
        ),
    ]

    reranked = rerank_results(
        "Python",
        results,
        relevance_weight=1.0,
        keyword_weight=0.0,
    )

    assert reranked[0].index == 0


def test_reranker_accepts_zero_relevance_weight():

    results = [
        make_result(
            "Database storage",
            0.9,
            0,
        ),
        make_result(
            "Python backend",
            0.1,
            1,
        ),
    ]

    reranked = rerank_results(
        "Python",
        results,
        relevance_weight=0.0,
        keyword_weight=1.0,
    )

    assert reranked[0].index == 1


def test_reranker_is_deterministic_for_equal_scores():

    first = make_result(
        "Python backend",
        0.8,
        0,
    )

    second = make_result(
        "Python backend",
        0.8,
        1,
    )

    reranked = rerank_results(
        "Python",
        [first, second],
    )

    assert reranked[0] is first
    assert reranked[1] is second