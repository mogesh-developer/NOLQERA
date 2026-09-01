import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    ContextRanker,
    RankedContext,
)
from nolqera.intelligence.pipeline.context_compressor import (
    ContextCompressor,
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


def test_compressor_returns_string():
    compressor = ContextCompressor()

    ranked = [
        make_ranked_context(
            "FastAPI is a Python framework.",
            0.9,
            0.8,
            0.87,
            0,
        )
    ]

    result = compressor.compress(
        ranked,
        max_sentences=1,
    )

    assert isinstance(result, str)


def test_compressor_returns_exact_single_sentence():
    compressor = ContextCompressor()

    ranked = [
        make_ranked_context(
            "FastAPI is a Python framework.",
            0.9,
            0.8,
            0.87,
            0,
        )
    ]

    result = compressor.compress(
        ranked,
        max_sentences=1,
    )

    assert result == (
        "FastAPI is a Python framework."
    )


def test_compressor_selects_highest_ranked_sentences():
    compressor = ContextCompressor()

    ranked = [
        make_ranked_context(
            "FastAPI is a Python framework.",
            0.95,
            0.90,
            0.935,
            0,
        ),
        make_ranked_context(
            "FastAPI supports automatic API documentation.",
            0.85,
            0.80,
            0.835,
            1,
        ),
        make_ranked_context(
            "React is a frontend library.",
            0.30,
            0.40,
            0.33,
            2,
        ),
    ]

    result = compressor.compress(
        ranked,
        max_sentences=2,
    )

    assert result == (
        "FastAPI is a Python framework. "
        "FastAPI supports automatic API documentation."
    )


def test_compressor_respects_max_sentences():
    compressor = ContextCompressor()

    ranked = [
        make_ranked_context(
            "Sentence one.",
            0.9,
            0.9,
            0.9,
            0,
        ),
        make_ranked_context(
            "Sentence two.",
            0.8,
            0.8,
            0.8,
            1,
        ),
        make_ranked_context(
            "Sentence three.",
            0.7,
            0.7,
            0.7,
            2,
        ),
    ]

    result = compressor.compress(
        ranked,
        max_sentences=2,
    )

    assert result == (
        "Sentence one. Sentence two."
    )


def test_compressor_preserves_rank_order():
    compressor = ContextCompressor()

    ranked = [
        make_ranked_context(
            "Highest ranked.",
            0.95,
            0.90,
            0.935,
            5,
        ),
        make_ranked_context(
            "Second ranked.",
            0.85,
            0.80,
            0.835,
            2,
        ),
        make_ranked_context(
            "Third ranked.",
            0.75,
            0.70,
            0.735,
            8,
        ),
    ]

    result = compressor.compress(
        ranked,
        max_sentences=3,
    )

    assert result == (
        "Highest ranked. "
        "Second ranked. "
        "Third ranked."
    )


def test_compressor_does_not_rewrite_text():
    compressor = ContextCompressor()

    original = (
        "FastAPI provides automatic documentation "
        "for APIs."
    )

    ranked = [
        make_ranked_context(
            original,
            0.9,
            0.9,
            0.9,
            0,
        )
    ]

    result = compressor.compress(
        ranked,
        max_sentences=1,
    )

    assert result == original


def test_compressor_empty_context_returns_empty_string():
    compressor = ContextCompressor()

    result = compressor.compress(
        [],
        max_sentences=3,
    )

    assert result == ""


def test_compressor_rejects_non_list():
    compressor = ContextCompressor()

    with pytest.raises(
        TypeError,
        match="ranked_context must be a list",
    ):
        compressor.compress(
            "invalid",
            max_sentences=2,
        )


def test_compressor_rejects_invalid_ranked_context():
    compressor = ContextCompressor()

    with pytest.raises(
        TypeError,
        match="ranked_context must contain RankedContext instances",
    ):
        compressor.compress(
            ["invalid"],
            max_sentences=2,
        )


def test_compressor_rejects_non_integer_max_sentences():
    compressor = ContextCompressor()

    with pytest.raises(
        TypeError,
        match="max_sentences must be an integer",
    ):
        compressor.compress(
            [],
            max_sentences=2.5,
        )


def test_compressor_rejects_zero_max_sentences():
    compressor = ContextCompressor()

    with pytest.raises(
        ValueError,
        match="max_sentences must be greater than zero",
    ):
        compressor.compress(
            [],
            max_sentences=0,
        )


def test_compressor_rejects_negative_max_sentences():
    compressor = ContextCompressor()

    with pytest.raises(
        ValueError,
        match="max_sentences must be greater than zero",
    ):
        compressor.compress(
            [],
            max_sentences=-1,
        )


def test_compressor_keeps_all_when_limit_exceeds_context():
    compressor = ContextCompressor()

    ranked = [
        make_ranked_context(
            "First sentence.",
            0.9,
            0.9,
            0.9,
            0,
        ),
        make_ranked_context(
            "Second sentence.",
            0.8,
            0.8,
            0.8,
            1,
        ),
    ]

    result = compressor.compress(
        ranked,
        max_sentences=10,
    )

    assert result == (
        "First sentence. Second sentence."
    )