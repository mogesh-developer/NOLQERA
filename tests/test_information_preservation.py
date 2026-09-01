
import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.information_preservation import (
    InformationPreservationResult,
    InformationPreserver,
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


def test_default_threshold():

    preserver = InformationPreserver()

    assert preserver.importance_threshold == 0.70


def test_custom_threshold():

    preserver = InformationPreserver(
        importance_threshold=0.85
    )

    assert preserver.importance_threshold == 0.85


def test_rejects_non_numeric_threshold():

    with pytest.raises(TypeError):
        InformationPreserver(
            importance_threshold="0.7"
        )


def test_rejects_threshold_above_one():

    with pytest.raises(ValueError):
        InformationPreserver(
            importance_threshold=1.1
        )


def test_rejects_negative_threshold():

    with pytest.raises(ValueError):
        InformationPreserver(
            importance_threshold=-0.1
        )


def test_identifies_important_sentences_exactly():

    preserver = InformationPreserver(
        importance_threshold=0.70
    )

    sentences = [
        make_ranked_context(
            "Low importance.",
            0.80,
            0.40,
            0.70,
            0,
        ),
        make_ranked_context(
            "Important sentence.",
            0.90,
            0.80,
            0.90,
            1,
        ),
        make_ranked_context(
            "Very important sentence.",
            0.95,
            0.95,
            0.95,
            2,
        ),
    ]

    result = preserver.identify_important(
        sentences
    )

    assert [
        item.result.text
        for item in result
    ] == [
        "Important sentence.",
        "Very important sentence.",
    ]


def test_threshold_is_inclusive():

    preserver = InformationPreserver(
        importance_threshold=0.70
    )

    sentence = make_ranked_context(
        "Boundary sentence.",
        0.80,
        0.70,
        0.80,
        0,
    )

    result = preserver.identify_important(
        [sentence]
    )

    assert [
        item.result.text
        for item in result
    ] == [
        "Boundary sentence.",
    ]


def test_no_important_sentences_returns_empty():

    preserver = InformationPreserver(
        importance_threshold=0.90
    )

    sentences = [
        make_ranked_context(
            "Low value A.",
            0.80,
            0.40,
            0.80,
            0,
        ),
        make_ranked_context(
            "Low value B.",
            0.70,
            0.60,
            0.70,
            1,
        ),
    ]

    result = preserver.identify_important(
        sentences
    )

    assert result == []


def test_all_important_information_is_preserved():

    preserver = InformationPreserver(
        importance_threshold=0.70
    )

    original = [
        make_ranked_context(
            "Python 3.11 is supported.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "MongoDB stores documents.",
            0.85,
            0.80,
            0.85,
            1,
        ),
    ]

    compressed = [
        original[0],
        original[1],
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is True

    assert [
        item.result.text
        for item in result.preserved
    ] == [
        "Python 3.11 is supported.",
        "MongoDB stores documents.",
    ]

    assert result.missing == []


def test_missing_important_sentence_is_detected():

    preserver = InformationPreserver(
        importance_threshold=0.70
    )

    original = [
        make_ranked_context(
            "Python 3.11 is supported.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "MongoDB stores documents.",
            0.85,
            0.80,
            0.85,
            1,
        ),
    ]

    compressed = [
        original[1],
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is False

    assert [
        item.result.text
        for item in result.preserved
    ] == [
        "MongoDB stores documents.",
    ]

    assert [
        item.result.text
        for item in result.missing
    ] == [
        "Python 3.11 is supported.",
    ]


def test_low_importance_removal_does_not_fail_preservation():

    preserver = InformationPreserver(
        importance_threshold=0.70
    )

    original = [
        make_ranked_context(
            "Important information.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "Low value repetition.",
            0.50,
            0.30,
            0.40,
            1,
        ),
    ]

    compressed = [
        original[0],
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is True

    assert [
        item.result.text
        for item in result.preserved
    ] == [
        "Important information.",
    ]

    assert result.missing == []


def test_exact_index_matching_is_used():

    preserver = InformationPreserver(
        importance_threshold=0.70
    )

    original = [
        make_ranked_context(
            "Original important fact.",
            0.95,
            0.95,
            0.95,
            5,
        )
    ]

    different_index = make_ranked_context(
        "Different sentence with same importance.",
        0.95,
        0.95,
        0.95,
        6,
    )

    result = preserver.validate(
        original,
        [different_index],
    )

    assert result.is_preserved is False

    assert [
        item.result.text
        for item in result.missing
    ] == [
        "Original important fact.",
    ]


def test_original_order_is_preserved():

    preserver = InformationPreserver(
        importance_threshold=0.70
    )

    sentences = [
        make_ranked_context(
            "Third important.",
            0.90,
            0.90,
            0.90,
            2,
        ),
        make_ranked_context(
            "First important.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "Second important.",
            0.90,
            0.90,
            0.90,
            1,
        ),
    ]

    result = preserver.identify_important(
        sentences
    )

    assert [
        item.result.index
        for item in result
    ] == [
        0,
        1,
        2,
    ]

    assert [
        item.result.text
        for item in result
    ] == [
        "First important.",
        "Second important.",
        "Third important.",
    ]


def test_empty_original_context_is_preserved():

    preserver = InformationPreserver()

    result = preserver.validate(
        [],
        [],
    )

    assert result.is_preserved is True
    assert result.preserved == []
    assert result.missing == []


def test_empty_compressed_context_fails_when_important_data_exists():

    preserver = InformationPreserver()

    original = [
        make_ranked_context(
            "Critical information.",
            0.95,
            0.95,
            0.95,
            0,
        )
    ]

    result = preserver.validate(
        original,
        [],
    )

    assert result.is_preserved is False

    assert [
        item.result.text
        for item in result.missing
    ] == [
        "Critical information.",
    ]


def test_rejects_invalid_original_type():

    preserver = InformationPreserver()

    with pytest.raises(TypeError):
        preserver.validate(
            None,
            [],
        )


def test_rejects_invalid_compressed_type():

    preserver = InformationPreserver()

    with pytest.raises(TypeError):
        preserver.validate(
            [],
            None,
        )


def test_rejects_invalid_original_item():

    preserver = InformationPreserver()

    with pytest.raises(TypeError):
        preserver.validate(
            ["invalid"],
            [],
        )


def test_rejects_invalid_compressed_item():

    preserver = InformationPreserver()

    with pytest.raises(TypeError):
        preserver.validate(
            [],
            ["invalid"],
        )


def test_result_type_is_exact():

    preserver = InformationPreserver()

    sentence = make_ranked_context(
        "Important.",
        0.90,
        0.90,
        0.90,
        0,
    )

    result = preserver.validate(
        [sentence],
        [sentence],
    )

    assert isinstance(
        result,
        InformationPreservationResult,
    )
