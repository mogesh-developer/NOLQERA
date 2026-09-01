
import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.semantic_compression import (
    SemanticCompressionResult,
    SemanticCompressor,
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


def same_topic_redundancy(
    first: str,
    second: str,
) -> bool:
    """
    Deterministic test double representing the existing
    semantic redundancy intelligence.

    These pairs are intentionally explicit so the test verifies
    exact compression behavior without depending on an external
    embedding model.
    """

    redundant_pairs = {
        frozenset(
            {
                "FastAPI is a Python web framework.",
                "FastAPI is a web framework built with Python.",
            }
        ),
        frozenset(
            {
                "FastAPI supports API development.",
                "FastAPI can be used to build APIs.",
            }
        ),
    }

    return frozenset({first, second}) in redundant_pairs


def test_requires_redundancy_checker():

    compressor = SemanticCompressor()

    sentences = [
        make_ranked_context(
            "Sentence A.",
            0.90,
            0.90,
            0.90,
            0,
        )
    ]

    with pytest.raises(ValueError):
        compressor.compress(sentences)


def test_rejects_non_callable_checker():

    with pytest.raises(TypeError):
        SemanticCompressor(
            redundancy_checker="invalid"
        )


def test_empty_input_returns_empty_result():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    result = compressor.compress([])

    assert isinstance(
        result,
        SemanticCompressionResult,
    )

    assert result.selected == []
    assert result.removed == []
    assert result.text == ""


def test_semantically_redundant_sentence_is_removed():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    first = make_ranked_context(
        "FastAPI is a Python web framework.",
        0.90,
        0.90,
        0.95,
        0,
    )

    duplicate = make_ranked_context(
        "FastAPI is a web framework built with Python.",
        0.85,
        0.80,
        0.80,
        1,
    )

    result = compressor.compress(
        [first, duplicate]
    )

    assert result.text == (
        "FastAPI is a Python web framework."
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "FastAPI is a Python web framework.",
    ]

    assert [
        item.result.text
        for item in result.removed
    ] == [
        "FastAPI is a web framework built with Python.",
    ]


def test_higher_ranked_sentence_is_kept_as_representative():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    lower_ranked = make_ranked_context(
        "FastAPI is a Python web framework.",
        0.80,
        0.80,
        0.70,
        0,
    )

    higher_ranked = make_ranked_context(
        "FastAPI is a web framework built with Python.",
        0.95,
        0.95,
        0.95,
        1,
    )

    result = compressor.compress(
        [lower_ranked, higher_ranked]
    )

    assert result.text == (
        "FastAPI is a web framework built with Python."
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "FastAPI is a web framework built with Python.",
    ]

    assert [
        item.result.text
        for item in result.removed
    ] == [
        "FastAPI is a Python web framework.",
    ]


def test_unique_information_is_preserved():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    sentences = [
        make_ranked_context(
            "FastAPI is a Python web framework.",
            0.95,
            0.90,
            0.95,
            0,
        ),
        make_ranked_context(
            "MongoDB is a NoSQL database.",
            0.85,
            0.85,
            0.85,
            1,
        ),
    ]

    result = compressor.compress(
        sentences
    )

    assert result.text == (
        "FastAPI is a Python web framework. "
        "MongoDB is a NoSQL database."
    )

    assert len(result.selected) == 2
    assert result.removed == []


def test_multiple_redundant_groups_are_compressed():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    sentences = [
        make_ranked_context(
            "FastAPI is a Python web framework.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "FastAPI is a web framework built with Python.",
            0.80,
            0.80,
            0.80,
            1,
        ),
        make_ranked_context(
            "FastAPI supports API development.",
            0.90,
            0.90,
            0.90,
            2,
        ),
        make_ranked_context(
            "FastAPI can be used to build APIs.",
            0.75,
            0.75,
            0.75,
            3,
        ),
    ]

    result = compressor.compress(
        sentences
    )

    assert result.text == (
        "FastAPI is a Python web framework. "
        "FastAPI supports API development."
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "FastAPI is a Python web framework.",
        "FastAPI supports API development.",
    ]

    assert [
        item.result.text
        for item in result.removed
    ] == [
        "FastAPI is a web framework built with Python.",
        "FastAPI can be used to build APIs.",
    ]


def test_final_output_restores_original_order():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    first = make_ranked_context(
        "FastAPI is a Python web framework.",
        0.70,
        0.70,
        0.70,
        0,
    )

    second = make_ranked_context(
        "MongoDB is a NoSQL database.",
        0.80,
        0.80,
        0.80,
        1,
    )

    third = make_ranked_context(
        "FastAPI is a web framework built with Python.",
        0.95,
        0.95,
        0.95,
        2,
    )

    result = compressor.compress(
        [first, second, third]
    )

    # Third is the representative of the first sentence
    # because it has the higher ranking score.
    #
    # Final output must still follow original context order.
    assert result.text == (
        "MongoDB is a NoSQL database. "
        "FastAPI is a web framework built with Python."
    )

    assert [
        item.result.index
        for item in result.selected
    ] == [
        1,
        2,
    ]


def test_non_redundant_sentences_are_never_removed():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    sentences = [
        make_ranked_context(
            "Python 3.11 is supported.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "MongoDB stores JSON-like documents.",
            0.90,
            0.90,
            0.90,
            1,
        ),
        make_ranked_context(
            "FastAPI provides API tooling.",
            0.85,
            0.85,
            0.85,
            2,
        ),
    ]

    result = compressor.compress(
        sentences
    )

    assert result.text == (
        "Python 3.11 is supported. "
        "MongoDB stores JSON-like documents. "
        "FastAPI provides API tooling."
    )

    assert len(result.selected) == 3
    assert result.removed == []


def test_original_objects_are_preserved():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    first = make_ranked_context(
        "FastAPI is a Python web framework.",
        0.95,
        0.95,
        0.95,
        0,
    )

    second = make_ranked_context(
        "MongoDB is a NoSQL database.",
        0.80,
        0.80,
        0.80,
        1,
    )

    result = compressor.compress(
        [first, second]
    )

    assert result.selected[0] is first
    assert result.selected[1] is second


def test_no_new_text_is_generated():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    sentences = [
        make_ranked_context(
            "Python 3.11 is supported.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "MongoDB is a NoSQL database.",
            0.80,
            0.80,
            0.80,
            1,
        ),
    ]

    result = compressor.compress(
        sentences
    )

    assert result.text == (
        "Python 3.11 is supported. "
        "MongoDB is a NoSQL database."
    )


def test_semantic_compression_does_not_use_plain_text_equality():

    def semantic_checker(
        first: str,
        second: str,
    ) -> bool:

        return {
            first,
            second,
        } == {
            "FastAPI is a Python web framework.",
            "FastAPI is a web framework built with Python.",
        }

    compressor = SemanticCompressor(
        redundancy_checker=semantic_checker
    )

    first = make_ranked_context(
        "FastAPI is a Python web framework.",
        0.90,
        0.90,
        0.90,
        0,
    )

    second = make_ranked_context(
        "FastAPI is a web framework built with Python.",
        0.80,
        0.80,
        0.80,
        1,
    )

    result = compressor.compress(
        [first, second]
    )

    assert len(result.selected) == 1
    assert len(result.removed) == 1

    assert result.text == (
        "FastAPI is a Python web framework."
    )


def test_rejects_invalid_input_type():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    with pytest.raises(TypeError):
        compressor.compress(None)


def test_rejects_invalid_sentence_item():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    with pytest.raises(TypeError):
        compressor.compress(
            ["invalid"]
        )


def test_tuple_input_is_supported():

    compressor = SemanticCompressor(
        redundancy_checker=same_topic_redundancy
    )

    first = make_ranked_context(
        "Python is a programming language.",
        0.90,
        0.90,
        0.90,
        0,
    )

    result = compressor.compress(
        (first,)
    )

    assert result.text == (
        "Python is a programming language."
    )

    assert len(result.selected) == 1

