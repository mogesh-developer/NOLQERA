
import pytest

from nolqera.intelligence.context_optimization.context_prioritization import (
    ContextPrioritizer,
    PrioritizedContext,
)
from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def make_ranked_context(
    text: str,
    relevance: float,
    importance: float,
    ranking: float,
    index: int,
) -> RankedContext:

    result = SemanticSearchResult(
        text=text,
        score=relevance,
        index=index,
    )

    return RankedContext(
        result=result,
        relevance_score=relevance,
        importance_score=importance,
        ranking_score=ranking,
    )


def test_default_prioritization_is_descending():

    prioritizer = ContextPrioritizer()

    assert prioritizer.descending is True


def test_invalid_descending_value_is_rejected():

    with pytest.raises(TypeError):
        ContextPrioritizer(
            descending="true"
        )


def test_prioritizes_by_existing_ranking_score():

    prioritizer = ContextPrioritizer()

    contexts = [
        make_ranked_context(
            "Low priority.",
            0.70,
            0.60,
            0.40,
            0,
        ),
        make_ranked_context(
            "Highest priority.",
            0.90,
            0.90,
            0.95,
            1,
        ),
        make_ranked_context(
            "Medium priority.",
            0.85,
            0.80,
            0.70,
            2,
        ),
    ]

    result = prioritizer.prioritize(contexts)

    assert [
        item.context.result.text
        for item in result
    ] == [
        "Highest priority.",
        "Medium priority.",
        "Low priority.",
    ]


def test_priority_numbers_are_exact():

    prioritizer = ContextPrioritizer()

    contexts = [
        make_ranked_context(
            "A",
            0.80,
            0.80,
            0.80,
            0,
        ),
        make_ranked_context(
            "B",
            0.90,
            0.90,
            0.90,
            1,
        ),
        make_ranked_context(
            "C",
            0.70,
            0.70,
            0.70,
            2,
        ),
    ]

    result = prioritizer.prioritize(contexts)

    assert [
        item.priority
        for item in result
    ] == [
        0,
        1,
        2,
    ]

    assert [
        item.context.result.text
        for item in result
    ] == [
        "B",
        "A",
        "C",
    ]


def test_importance_breaks_ranking_ties():

    prioritizer = ContextPrioritizer()

    contexts = [
        make_ranked_context(
            "Lower importance.",
            0.90,
            0.50,
            0.80,
            0,
        ),
        make_ranked_context(
            "Higher importance.",
            0.80,
            0.90,
            0.80,
            1,
        ),
    ]

    result = prioritizer.prioritize(contexts)

    assert [
        item.context.result.text
        for item in result
    ] == [
        "Higher importance.",
        "Lower importance.",
    ]


def test_relevance_breaks_complete_tie():

    prioritizer = ContextPrioritizer()

    contexts = [
        make_ranked_context(
            "Lower relevance.",
            0.60,
            0.80,
            0.80,
            0,
        ),
        make_ranked_context(
            "Higher relevance.",
            0.90,
            0.80,
            0.80,
            1,
        ),
    ]

    result = prioritizer.prioritize(contexts)

    assert [
        item.context.result.text
        for item in result
    ] == [
        "Higher relevance.",
        "Lower relevance.",
    ]


def test_index_provides_final_deterministic_tie_break():

    prioritizer = ContextPrioritizer()

    contexts = [
        make_ranked_context(
            "Index 2.",
            0.80,
            0.80,
            0.80,
            2,
        ),
        make_ranked_context(
            "Index 0.",
            0.80,
            0.80,
            0.80,
            0,
        ),
        make_ranked_context(
            "Index 1.",
            0.80,
            0.80,
            0.80,
            1,
        ),
    ]

    result = prioritizer.prioritize(contexts)

    assert [
        item.context.result.text
        for item in result
    ] == [
        "Index 0.",
        "Index 1.",
        "Index 2.",
    ]


def test_ascending_mode():

    prioritizer = ContextPrioritizer(
        descending=False
    )

    contexts = [
        make_ranked_context(
            "High.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "Low.",
            0.70,
            0.70,
            0.30,
            1,
        ),
    ]

    result = prioritizer.prioritize(contexts)

    assert [
        item.context.result.text
        for item in result
    ] == [
        "Low.",
        "High.",
    ]


def test_select_top_returns_exact_contexts():

    prioritizer = ContextPrioritizer()

    contexts = [
        make_ranked_context(
            "Third.",
            0.70,
            0.70,
            0.60,
            0,
        ),
        make_ranked_context(
            "First.",
            0.95,
            0.95,
            0.95,
            1,
        ),
        make_ranked_context(
            "Second.",
            0.85,
            0.85,
            0.80,
            2,
        ),
    ]

    result = prioritizer.select_top(
        contexts,
        2,
    )

    assert [
        item.result.text
        for item in result
    ] == [
        "First.",
        "Second.",
    ]


def test_select_top_zero_returns_empty():

    prioritizer = ContextPrioritizer()

    contexts = [
        make_ranked_context(
            "A",
            0.90,
            0.90,
            0.90,
            0,
        )
    ]

    result = prioritizer.select_top(
        contexts,
        0,
    )

    assert result == []


def test_select_top_larger_than_context_returns_all():

    prioritizer = ContextPrioritizer()

    contexts = [
        make_ranked_context(
            "A",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "B",
            0.80,
            0.80,
            0.80,
            1,
        ),
    ]

    result = prioritizer.select_top(
        contexts,
        10,
    )

    assert [
        item.result.text
        for item in result
    ] == [
        "A",
        "B",
    ]


def test_negative_limit_is_rejected():

    prioritizer = ContextPrioritizer()

    with pytest.raises(ValueError):
        prioritizer.select_top(
            [],
            -1,
        )


def test_non_integer_limit_is_rejected():

    prioritizer = ContextPrioritizer()

    with pytest.raises(TypeError):
        prioritizer.select_top(
            [],
            2.5,
        )


def test_empty_context_returns_empty():

    prioritizer = ContextPrioritizer()

    result = prioritizer.prioritize([])

    assert result == []


def test_invalid_context_type_is_rejected():

    prioritizer = ContextPrioritizer()

    with pytest.raises(TypeError):
        prioritizer.prioritize(
            None
        )


def test_invalid_context_item_is_rejected():

    prioritizer = ContextPrioritizer()

    with pytest.raises(TypeError):
        prioritizer.prioritize(
            ["invalid"]
        )


def test_result_type_is_exact():

    prioritizer = ContextPrioritizer()

    context = make_ranked_context(
        "Important context.",
        0.90,
        0.90,
        0.90,
        0,
    )

    result = prioritizer.prioritize(
        [context]
    )

    assert isinstance(
        result[0],
        PrioritizedContext,
    )


def test_original_context_objects_are_not_modified():

    prioritizer = ContextPrioritizer()

    context = make_ranked_context(
        "Original.",
        0.90,
        0.80,
        0.85,
        0,
    )

    original_text = context.result.text
    original_index = context.result.index
    original_ranking = context.ranking_score

    prioritizer.prioritize(
        [context]
    )

    assert context.result.text == original_text
    assert context.result.index == original_index
    assert context.ranking_score == original_ranking
