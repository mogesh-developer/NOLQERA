
import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.sentence_selection import (
    SentenceSelection,
    SentenceSelector,
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


def test_selector_accepts_valid_max_sentences():

    selector = SentenceSelector(
        max_sentences=2
    )

    assert selector.max_sentences == 2


def test_rejects_non_integer_max_sentences():

    with pytest.raises(TypeError):
        SentenceSelector(
            max_sentences="2"
        )


def test_rejects_zero_max_sentences():

    with pytest.raises(ValueError):
        SentenceSelector(
            max_sentences=0
        )


def test_rejects_negative_max_sentences():

    with pytest.raises(ValueError):
        SentenceSelector(
            max_sentences=-1
        )


def test_selects_highest_ranking_sentences():

    selector = SentenceSelector(
        max_sentences=2
    )

    ranked_context = [
        make_ranked_context(
            "Sentence A.",
            0.80,
            0.70,
            0.60,
            0,
        ),
        make_ranked_context(
            "Sentence B.",
            0.90,
            0.90,
            0.95,
            1,
        ),
        make_ranked_context(
            "Sentence C.",
            0.85,
            0.85,
            0.85,
            2,
        ),
    ]

    result = selector.select(
        ranked_context
    )

    assert result.text == (
        "Sentence B. Sentence C."
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "Sentence B.",
        "Sentence C.",
    ]


def test_selection_uses_ranking_score():

    selector = SentenceSelector(
        max_sentences=1
    )

    ranked_context = [
        make_ranked_context(
            "High relevance.",
            0.99,
            0.30,
            0.50,
            0,
        ),
        make_ranked_context(
            "High importance.",
            0.80,
            0.95,
            0.90,
            1,
        ),
    ]

    result = selector.select(
        ranked_context
    )

    assert result.text == (
        "High importance."
    )


def test_selected_sentences_restore_original_order():

    selector = SentenceSelector(
        max_sentences=3
    )

    ranked_context = [
        make_ranked_context(
            "First sentence.",
            0.70,
            0.70,
            0.70,
            0,
        ),
        make_ranked_context(
            "Second sentence.",
            0.95,
            0.95,
            0.99,
            1,
        ),
        make_ranked_context(
            "Third sentence.",
            0.85,
            0.85,
            0.90,
            2,
        ),
    ]

    result = selector.select(
        ranked_context
    )

    assert result.text == (
        "First sentence. "
        "Second sentence. "
        "Third sentence."
    )


def test_max_sentences_is_respected_exactly():

    selector = SentenceSelector(
        max_sentences=2
    )

    ranked_context = [
        make_ranked_context(
            "First.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "Second.",
            0.85,
            0.85,
            0.85,
            1,
        ),
        make_ranked_context(
            "Third.",
            0.80,
            0.80,
            0.80,
            2,
        ),
        make_ranked_context(
            "Fourth.",
            0.75,
            0.75,
            0.75,
            3,
        ),
    ]

    result = selector.select(
        ranked_context
    )

    assert len(result.selected) == 2

    assert result.text == (
        "First. Second."
    )


def test_less_candidates_than_limit_keeps_all():

    selector = SentenceSelector(
        max_sentences=10
    )

    ranked_context = [
        make_ranked_context(
            "First.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "Second.",
            0.80,
            0.80,
            0.80,
            1,
        ),
    ]

    result = selector.select(
        ranked_context
    )

    assert len(result.selected) == 2

    assert result.text == (
        "First. Second."
    )


def test_empty_context_returns_empty_selection():

    selector = SentenceSelector()

    result = selector.select([])

    assert isinstance(
        result,
        SentenceSelection,
    )

    assert result.selected == []
    assert result.text == ""


def test_rejects_non_list_context():

    selector = SentenceSelector()

    with pytest.raises(TypeError):
        selector.select(None)


def test_rejects_invalid_context_items():

    selector = SentenceSelector()

    with pytest.raises(TypeError):
        selector.select(
            ["Sentence"]
        )


def test_original_ranked_context_objects_are_preserved():

    selector = SentenceSelector(
        max_sentences=2
    )

    first = make_ranked_context(
        "First.",
        0.90,
        0.90,
        0.90,
        0,
    )

    second = make_ranked_context(
        "Second.",
        0.80,
        0.80,
        0.80,
        1,
    )

    result = selector.select(
        [first, second]
    )

    assert result.selected[0] is first
    assert result.selected[1] is second


def test_tie_is_resolved_by_original_index():

    selector = SentenceSelector(
        max_sentences=2
    )

    ranked_context = [
        make_ranked_context(
            "First.",
            0.80,
            0.80,
            0.90,
            0,
        ),
        make_ranked_context(
            "Second.",
            0.80,
            0.80,
            0.90,
            1,
        ),
        make_ranked_context(
            "Third.",
            0.80,
            0.80,
            0.80,
            2,
        ),
    ]

    result = selector.select(
        ranked_context
    )

    assert result.text == (
        "First. Second."
    )


def test_no_new_text_is_generated():

    selector = SentenceSelector(
        max_sentences=2
    )

    ranked_context = [
        make_ranked_context(
            "Python 3.11 is supported.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "FastAPI provides API tooling.",
            0.80,
            0.80,
            0.80,
            1,
        ),
    ]

    result = selector.select(
        ranked_context
    )

    assert result.text == (
        "Python 3.11 is supported. "
        "FastAPI provides API tooling."
    )

    assert "Python 3.11 is supported." in result.text
    assert "FastAPI provides API tooling." in result.text


def test_duplicate_indexes_are_selected_only_once():

    selector = SentenceSelector(
        max_sentences=3
    )

    first = make_ranked_context(
        "Original sentence.",
        0.90,
        0.90,
        0.95,
        0,
    )

    duplicate = make_ranked_context(
        "Duplicate representation.",
        0.85,
        0.85,
        0.90,
        0,
    )

    third = make_ranked_context(
        "Unique sentence.",
        0.80,
        0.80,
        0.80,
        1,
    )

    result = selector.select(
        [first, duplicate, third]
    )

    assert result.text == (
        "Original sentence. "
        "Unique sentence."
    )

    assert len(result.selected) == 2

