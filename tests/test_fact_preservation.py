
import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.fact_preservation import (
    FactPreservationResult,
    FactPreserver,
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


def test_default_percentage_configuration():

    preserver = FactPreserver()

    assert preserver.preserve_percentages is True


def test_percentage_configuration_can_be_disabled():

    preserver = FactPreserver(
        preserve_percentages=False
    )

    assert preserver.preserve_percentages is False


def test_rejects_invalid_percentage_configuration():

    with pytest.raises(TypeError):
        FactPreserver(
            preserve_percentages="yes"
        )


def test_identifies_numbers_exactly():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "Python 3.11 was tested in 2026.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "Processing took 48.46 seconds.",
            0.85,
            0.85,
            0.85,
            1,
        ),
    ]

    result = preserver.identify_required_facts(
        original
    )

    assert result == [
        "3.11",
        "48.46",
        "2026",
    ]


def test_percentage_is_preserved_as_exact_fact():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "The model achieved 95% accuracy.",
            0.95,
            0.95,
            0.95,
            0,
        )
    ]

    result = preserver.identify_required_facts(
        original
    )

    assert result == [
        "95%",
    ]


def test_multiple_numbers_in_one_sentence_are_extracted():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "The system processed 10 documents "
            "in 48.46 seconds with 95% accuracy.",
            0.95,
            0.95,
            0.95,
            0,
        )
    ]

    result = preserver.identify_required_facts(
        original
    )

    assert result == [
        "10",
        "48.46",
        "95%",
    ]


def test_all_facts_are_preserved():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "Python 3.11 was tested in 2026.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "Processing took 48.46 seconds.",
            0.90,
            0.90,
            0.90,
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

    assert isinstance(
        result,
        FactPreservationResult,
    )

    assert result.required_facts == [
        "3.11",
        "48.46",
        "2026",
    ]

    assert result.preserved_facts == [
        "3.11",
        "48.46",
        "2026",
    ]

    assert result.missing_facts == []

    assert result.is_preserved is True


def test_missing_number_is_detected():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "Python 3.11 was tested in 2026.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "Processing took 48.46 seconds.",
            0.90,
            0.90,
            0.90,
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

    assert result.required_facts == [
        "3.11",
        "48.46",
        "2026",
    ]

    assert result.preserved_facts == [
        "3.11",
        "2026",
    ]

    assert result.missing_facts == [
        "48.46",
    ]

    assert result.is_preserved is False


def test_missing_percentage_is_detected():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "The model achieved 95% accuracy.",
            0.95,
            0.95,
            0.95,
            0,
        )
    ]

    compressed = [
        make_ranked_context(
            "The model achieved high accuracy.",
            0.80,
            0.80,
            0.80,
            0,
        )
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.required_facts == [
        "95%",
    ]

    assert result.preserved_facts == []

    assert result.missing_facts == [
        "95%",
    ]

    assert result.is_preserved is False


def test_unique_facts_from_multiple_sentences_are_preserved():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "Version 3.11 is supported.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "The benchmark ran for 48.46 seconds.",
            0.90,
            0.90,
            0.90,
            1,
        ),
        make_ranked_context(
            "The test used 10 documents.",
            0.85,
            0.85,
            0.85,
            2,
        ),
    ]

    compressed = [
        original[0],
        original[1],
        original[2],
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.required_facts == [
        "3.11",
        "10",
        "48.46",
    ]

    assert result.preserved_facts == [
        "3.11",
        "10",
        "48.46",
    ]

    assert result.missing_facts == []

    assert result.is_preserved is True


def test_zero_is_preserved():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "The error rate was reduced to 0%.",
            0.90,
            0.90,
            0.90,
            0,
        )
    ]

    result = preserver.identify_required_facts(
        original
    )

    assert result == [
        "0%",
    ]


def test_decimal_values_are_preserved_exactly():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "Latency improved from 48.46 to 12.40 seconds.",
            0.90,
            0.90,
            0.90,
            0,
        )
    ]

    result = preserver.identify_required_facts(
        original
    )

    assert result == [
        "12.40",
        "48.46",
    ]


def test_percentage_disabled_removes_percent_sign():

    preserver = FactPreserver(
        preserve_percentages=False
    )

    original = [
        make_ranked_context(
            "Accuracy reached 95%.",
            0.90,
            0.90,
            0.90,
            0,
        )
    ]

    result = preserver.identify_required_facts(
        original
    )

    assert result == [
        "95",
    ]


def test_empty_context_has_no_facts():

    preserver = FactPreserver()

    result = preserver.identify_required_facts(
        []
    )

    assert result == []


def test_empty_context_is_preserved():

    preserver = FactPreserver()

    result = preserver.validate(
        [],
        [],
    )

    assert result.required_facts == []
    assert result.preserved_facts == []
    assert result.missing_facts == []
    assert result.is_preserved is True


def test_empty_compressed_context_fails_when_facts_exist():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "Python 3.11 was released.",
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

    assert result.required_facts == [
        "3.11",
    ]

    assert result.preserved_facts == []

    assert result.missing_facts == [
        "3.11",
    ]

    assert result.is_preserved is False


def test_rejects_invalid_original_type():

    preserver = FactPreserver()

    with pytest.raises(TypeError):
        preserver.validate(
            None,
            [],
        )


def test_rejects_invalid_compressed_type():

    preserver = FactPreserver()

    with pytest.raises(TypeError):
        preserver.validate(
            [],
            None,
        )


def test_rejects_invalid_original_item():

    preserver = FactPreserver()

    with pytest.raises(TypeError):
        preserver.validate(
            ["invalid"],
            [],
        )


def test_rejects_invalid_compressed_item():

    preserver = FactPreserver()

    with pytest.raises(TypeError):
        preserver.validate(
            [],
            ["invalid"],
        )


def test_text_without_numbers_contains_no_facts():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "FastAPI is a Python framework.",
            0.90,
            0.90,
            0.90,
            0,
        )
    ]

    result = preserver.identify_required_facts(
        original
    )

    assert result == []

