import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.extractive_summarization import (
    ExtractiveSummary,
    ExtractiveSummarizer,
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


def test_summarizer_accepts_max_sentences():

    summarizer = ExtractiveSummarizer(
        max_sentences=2
    )

    assert summarizer.max_sentences == 2


def test_rejects_non_integer_max_sentences():

    with pytest.raises(TypeError):
        ExtractiveSummarizer(
            max_sentences="2"
        )


def test_rejects_zero_max_sentences():

    with pytest.raises(ValueError):
        ExtractiveSummarizer(
            max_sentences=0
        )


def test_rejects_negative_max_sentences():

    with pytest.raises(ValueError):
        ExtractiveSummarizer(
            max_sentences=-1
        )


def test_returns_extractively_selected_sentences():

    summarizer = ExtractiveSummarizer(
        max_sentences=2
    )

    ranked_context = [
        make_ranked_context(
            "Python is a programming language.",
            0.80,
            0.70,
            0.77,
            0,
        ),
        make_ranked_context(
            "FastAPI is a Python web framework.",
            0.95,
            0.90,
            0.935,
            1,
        ),
        make_ranked_context(
            "JWT provides token based authentication.",
            0.90,
            0.85,
            0.885,
            2,
        ),
    ]

    summary = summarizer.summarize(
        ranked_context
    )

    assert summary.text == (
        "FastAPI is a Python web framework. "
        "JWT provides token based authentication."
    )


def test_highest_ranking_scores_are_selected():

    summarizer = ExtractiveSummarizer(
        max_sentences=2
    )

    ranked_context = [
        make_ranked_context(
            "Sentence A",
            0.90,
            0.50,
            0.70,
            0,
        ),
        make_ranked_context(
            "Sentence B",
            0.90,
            0.90,
            0.90,
            1,
        ),
        make_ranked_context(
            "Sentence C",
            0.80,
            0.80,
            0.80,
            2,
        ),
    ]

    summary = summarizer.summarize(
        ranked_context
    )

    assert [
        item.result.text
        for item in summary.selected
    ] == [
        "Sentence B",
        "Sentence C",
    ]


def test_selection_uses_ranking_score_not_relevance_only():

    summarizer = ExtractiveSummarizer(
        max_sentences=1
    )

    ranked_context = [
        make_ranked_context(
            "High relevance sentence",
            0.99,
            0.20,
            0.60,
            0,
        ),
        make_ranked_context(
            "High importance sentence",
            0.80,
            0.95,
            0.90,
            1,
        ),
    ]

    summary = summarizer.summarize(
        ranked_context
    )

    assert summary.text == (
        "High importance sentence"
    )


def test_selected_sentences_are_restored_to_original_order():

    summarizer = ExtractiveSummarizer(
        max_sentences=3
    )

    ranked_context = [
        make_ranked_context(
            "Sentence one.",
            0.80,
            0.80,
            0.70,
            0,
        ),
        make_ranked_context(
            "Sentence two.",
            0.90,
            0.90,
            0.95,
            1,
        ),
        make_ranked_context(
            "Sentence three.",
            0.85,
            0.85,
            0.85,
            2,
        ),
    ]

    summary = summarizer.summarize(
        ranked_context
    )

    assert summary.text == (
        "Sentence one. "
        "Sentence two. "
        "Sentence three."
    )


def test_only_top_n_sentences_are_selected():

    summarizer = ExtractiveSummarizer(
        max_sentences=2
    )

    ranked_context = [
        make_ranked_context(
            "Sentence one.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "Sentence two.",
            0.80,
            0.80,
            0.80,
            1,
        ),
        make_ranked_context(
            "Sentence three.",
            0.70,
            0.70,
            0.70,
            2,
        ),
    ]

    summary = summarizer.summarize(
        ranked_context
    )

    assert len(summary.selected) == 2

    assert summary.text == (
        "Sentence one. "
        "Sentence two."
    )


def test_max_sentences_larger_than_input_keeps_all():

    summarizer = ExtractiveSummarizer(
        max_sentences=10
    )

    ranked_context = [
        make_ranked_context(
            "First.",
            0.80,
            0.80,
            0.80,
            0,
        ),
        make_ranked_context(
            "Second.",
            0.70,
            0.70,
            0.70,
            1,
        ),
    ]

    summary = summarizer.summarize(
        ranked_context
    )

    assert summary.text == (
        "First. Second."
    )

    assert len(summary.selected) == 2


def test_empty_context_returns_empty_summary():

    summarizer = ExtractiveSummarizer()

    summary = summarizer.summarize([])

    assert isinstance(
        summary,
        ExtractiveSummary,
    )

    assert summary.selected == []
    assert summary.text == ""


def test_rejects_non_list_context():

    summarizer = ExtractiveSummarizer()

    with pytest.raises(TypeError):
        summarizer.summarize(None)


def test_rejects_invalid_context_item():

    summarizer = ExtractiveSummarizer()

    with pytest.raises(TypeError):
        summarizer.summarize(
            ["Sentence"]
        )


def test_preserves_original_ranked_context_objects():

    summarizer = ExtractiveSummarizer(
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

    summary = summarizer.summarize(
        [first, second]
    )

    assert summary.selected[0] is first
    assert summary.selected[1] is second


def test_summary_contains_only_original_text():

    summarizer = ExtractiveSummarizer(
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

    summary = summarizer.summarize(
        ranked_context
    )

    assert summary.text == (
        "Python 3.11 is supported. "
        "FastAPI provides API tooling."
    )

    assert "Python 3.11 is supported." in summary.text
    assert "FastAPI provides API tooling." in summary.text