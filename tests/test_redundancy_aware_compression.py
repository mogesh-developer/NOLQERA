
import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.redundancy_aware_compression import (
    RedundancyAwareCompressionResult,
    RedundancyAwareCompressor,
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


def exact_duplicate(
    first: str,
    second: str,
) -> bool:

    return first == second


def near_duplicate(
    first: str,
    second: str,
) -> bool:

    pairs = {
        frozenset(
            {
                "FastAPI is a Python web framework.",
                "FastAPI is a Python framework for web APIs.",
            }
        )
    }

    return frozenset({first, second}) in pairs


def semantic_redundancy(
    first: str,
    second: str,
) -> bool:

    pairs = {
        frozenset(
            {
                "FastAPI supports API development.",
                "FastAPI can be used to build APIs.",
            }
        )
    }

    return frozenset({first, second}) in pairs


def redundant_information(
    first: str,
    second: str,
) -> bool:

    pairs = {
        frozenset(
            {
                "The API uses JWT authentication.",
                "JWT is used for authenticating API requests.",
            }
        )
    }

    return frozenset({first, second}) in pairs


def test_requires_at_least_one_checker():

    compressor = RedundancyAwareCompressor()

    sentence = make_ranked_context(
        "Python is a programming language.",
        0.90,
        0.90,
        0.90,
        0,
    )

    with pytest.raises(ValueError):
        compressor.compress([sentence])


def test_rejects_non_callable_checker():

    with pytest.raises(TypeError):
        RedundancyAwareCompressor(
            exact_duplicate_checker="invalid"
        )


def test_empty_input_returns_empty_result():

    compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate
    )

    result = compressor.compress([])

    assert isinstance(
        result,
        RedundancyAwareCompressionResult,
    )

    assert result.selected == []
    assert result.removed == []
    assert result.text == ""


def test_exact_duplicate_is_removed():

    compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate
    )

    first = make_ranked_context(
        "Python is a programming language.",
        0.95,
        0.95,
        0.95,
        0,
    )

    duplicate = make_ranked_context(
        "Python is a programming language.",
        0.80,
        0.80,
        0.80,
        1,
    )

    result = compressor.compress(
        [first, duplicate]
    )

    assert result.text == (
        "Python is a programming language."
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "Python is a programming language.",
    ]

    assert [
        item.result.text
        for item in result.removed
    ] == [
        "Python is a programming language.",
    ]


def test_near_duplicate_is_removed():

    compressor = RedundancyAwareCompressor(
        near_duplicate_checker=near_duplicate
    )

    first = make_ranked_context(
        "FastAPI is a Python web framework.",
        0.95,
        0.95,
        0.95,
        0,
    )

    duplicate = make_ranked_context(
        "FastAPI is a Python framework for web APIs.",
        0.80,
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

    assert len(result.selected) == 1
    assert len(result.removed) == 1


def test_semantic_redundancy_is_removed():

    compressor = RedundancyAwareCompressor(
        semantic_redundancy_checker=semantic_redundancy
    )

    first = make_ranked_context(
        "FastAPI supports API development.",
        0.95,
        0.95,
        0.95,
        0,
    )

    duplicate = make_ranked_context(
        "FastAPI can be used to build APIs.",
        0.80,
        0.80,
        0.80,
        1,
    )

    result = compressor.compress(
        [first, duplicate]
    )

    assert result.text == (
        "FastAPI supports API development."
    )

    assert len(result.selected) == 1
    assert len(result.removed) == 1


def test_redundant_information_is_removed():

    compressor = RedundancyAwareCompressor(
        redundant_information_checker=redundant_information
    )

    first = make_ranked_context(
        "The API uses JWT authentication.",
        0.95,
        0.95,
        0.95,
        0,
    )

    duplicate = make_ranked_context(
        "JWT is used for authenticating API requests.",
        0.80,
        0.80,
        0.80,
        1,
    )

    result = compressor.compress(
        [first, duplicate]
    )

    assert result.text == (
        "The API uses JWT authentication."
    )

    assert len(result.selected) == 1
    assert len(result.removed) == 1


def test_multiple_redundancy_strategies_are_composed():

    compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate,
        near_duplicate_checker=near_duplicate,
        semantic_redundancy_checker=semantic_redundancy,
        redundant_information_checker=redundant_information,
    )

    sentences = [
        make_ranked_context(
            "Python is a programming language.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "Python is a programming language.",
            0.80,
            0.80,
            0.80,
            1,
        ),
        make_ranked_context(
            "FastAPI is a Python web framework.",
            0.90,
            0.90,
            0.90,
            2,
        ),
        make_ranked_context(
            "FastAPI is a Python framework for web APIs.",
            0.75,
            0.75,
            0.75,
            3,
        ),
        make_ranked_context(
            "FastAPI supports API development.",
            0.85,
            0.85,
            0.85,
            4,
        ),
        make_ranked_context(
            "FastAPI can be used to build APIs.",
            0.70,
            0.70,
            0.70,
            5,
        ),
    ]

    result = compressor.compress(
        sentences
    )

    assert result.text == (
        "Python is a programming language. "
        "FastAPI is a Python web framework. "
        "FastAPI supports API development."
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "Python is a programming language.",
        "FastAPI is a Python web framework.",
        "FastAPI supports API development.",
    ]

    assert [
        item.result.text
        for item in result.removed
    ] == [
        "Python is a programming language.",
        "FastAPI is a Python framework for web APIs.",
        "FastAPI can be used to build APIs.",
    ]


def test_unique_information_is_preserved():

    compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate,
        near_duplicate_checker=near_duplicate,
        semantic_redundancy_checker=semantic_redundancy,
        redundant_information_checker=redundant_information,
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


def test_higher_ranked_sentence_becomes_representative():

    compressor = RedundancyAwareCompressor(
        semantic_redundancy_checker=semantic_redundancy
    )

    lower = make_ranked_context(
        "FastAPI supports API development.",
        0.80,
        0.80,
        0.70,
        0,
    )

    higher = make_ranked_context(
        "FastAPI can be used to build APIs.",
        0.95,
        0.95,
        0.95,
        1,
    )

    result = compressor.compress(
        [lower, higher]
    )

    assert result.text == (
        "FastAPI can be used to build APIs."
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "FastAPI can be used to build APIs.",
    ]


def test_final_output_preserves_original_order():

    compressor = RedundancyAwareCompressor(
        semantic_redundancy_checker=semantic_redundancy
    )

    first = make_ranked_context(
        "FastAPI supports API development.",
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
        "FastAPI can be used to build APIs.",
        0.95,
        0.95,
        0.95,
        2,
    )

    result = compressor.compress(
        [first, second, third]
    )

    assert result.text == (
        "MongoDB is a NoSQL database. "
        "FastAPI can be used to build APIs."
    )

    assert [
        item.result.index
        for item in result.selected
    ] == [
        1,
        2,
    ]


def test_no_new_text_is_generated():

    compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate
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
            "FastAPI provides API tooling.",
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
        "FastAPI provides API tooling."
    )


def test_original_objects_are_preserved():

    compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate
    )

    first = make_ranked_context(
        "Python is a programming language.",
        0.90,
        0.90,
        0.90,
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


def test_rejects_non_list_or_tuple():

    compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate
    )

    with pytest.raises(TypeError):
        compressor.compress(None)


def test_rejects_invalid_items():

    compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate
    )

    with pytest.raises(TypeError):
        compressor.compress(
            ["invalid"]
        )


def test_tuple_input_is_supported():

    compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate
    )

    sentence = make_ranked_context(
        "Python is a programming language.",
        0.90,
        0.90,
        0.90,
        0,
    )

    result = compressor.compress(
        (sentence,)
    )

    assert result.text == (
        "Python is a programming language."
    )

    assert len(result.selected) == 1
