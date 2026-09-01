
import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.entity_preservation import (
    EntityPreservationResult,
    EntityPreserver,
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


def test_entity_extractor_is_required():

    with pytest.raises(TypeError):
        EntityPreserver(None)


def exact_entity_extractor(text: str):

    known_entities = {
        "FastAPI",
        "Python",
        "MongoDB",
        "RaavOne",
        "NOLQERA",
    }

    return [
        entity
        for entity in known_entities
        if entity in text
    ]


def case_insensitive_entity_extractor(text: str):

    entities = {
        "FastAPI",
        "Python",
        "MongoDB",
    }

    text_lower = text.casefold()

    return [
        entity
        for entity in entities
        if entity.casefold() in text_lower
    ]


def test_identifies_entities_exactly():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    original = [
        make_ranked_context(
            "FastAPI is built with Python.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "MongoDB stores application data.",
            0.85,
            0.85,
            0.85,
            1,
        ),
    ]

    result = preserver.identify_required_entities(
        original
    )

    assert result == [
        "FastAPI",
        "MongoDB",
        "Python",
    ]


def test_entities_are_sorted_deterministically():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    original = [
        make_ranked_context(
            "RaavOne uses NOLQERA.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "Python and FastAPI are used.",
            0.90,
            0.90,
            0.90,
            1,
        ),
    ]

    result = preserver.identify_required_entities(
        original
    )

    assert result == [
        "FastAPI",
        "NOLQERA",
        "Python",
        "RaavOne",
    ]


def test_all_entities_are_preserved():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    original = [
        make_ranked_context(
            "FastAPI is built with Python.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "MongoDB stores application data.",
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
        EntityPreservationResult,
    )

    assert result.required_entities == [
        "FastAPI",
        "MongoDB",
        "Python",
    ]

    assert result.preserved_entities == [
        "FastAPI",
        "MongoDB",
        "Python",
    ]

    assert result.missing_entities == []

    assert result.is_preserved is True


def test_missing_entity_is_detected():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    original = [
        make_ranked_context(
            "FastAPI is built with Python.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "MongoDB stores application data.",
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

    assert result.required_entities == [
        "FastAPI",
        "MongoDB",
        "Python",
    ]

    assert result.preserved_entities == [
        "FastAPI",
        "Python",
    ]

    assert result.missing_entities == [
        "MongoDB",
    ]

    assert result.is_preserved is False


def test_multiple_missing_entities_are_reported_exactly():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    original = [
        make_ranked_context(
            "RaavOne uses NOLQERA with Python.",
            0.95,
            0.95,
            0.95,
            0,
        ),
        make_ranked_context(
            "FastAPI connects to MongoDB.",
            0.90,
            0.90,
            0.90,
            1,
        ),
    ]

    compressed = [
        make_ranked_context(
            "Python remains in the compressed context.",
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

    assert result.required_entities == [
        "FastAPI",
        "MongoDB",
        "NOLQERA",
        "Python",
        "RaavOne",
    ]

    assert result.preserved_entities == [
        "Python",
    ]

    assert result.missing_entities == [
        "FastAPI",
        "MongoDB",
        "NOLQERA",
        "RaavOne",
    ]

    assert result.is_preserved is False


def test_entity_comparison_is_case_insensitive():

    preserver = EntityPreserver(
        case_insensitive_entity_extractor
    )

    original = [
        make_ranked_context(
            "FastAPI uses Python and MongoDB.",
            0.95,
            0.95,
            0.95,
            0,
        )
    ]

    compressed = [
        make_ranked_context(
            "fastapi uses python and mongodb.",
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

    assert result.required_entities == [
        "FastAPI",
        "MongoDB",
        "Python",
    ]

    assert result.preserved_entities == [
        "FastAPI",
        "MongoDB",
        "Python",
    ]

    assert result.missing_entities == []

    assert result.is_preserved is True


def test_duplicate_entity_mentions_are_collapsed():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    original = [
        make_ranked_context(
            "FastAPI uses Python.",
            0.90,
            0.90,
            0.90,
            0,
        ),
        make_ranked_context(
            "FastAPI is a Python framework.",
            0.80,
            0.80,
            0.80,
            1,
        ),
    ]

    result = preserver.identify_required_entities(
        original
    )

    assert result == [
        "FastAPI",
        "Python",
    ]


def test_empty_context_has_no_required_entities():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    result = preserver.identify_required_entities(
        []
    )

    assert result == []


def test_empty_context_is_preserved():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    result = preserver.validate(
        [],
        [],
    )

    assert result.required_entities == []
    assert result.preserved_entities == []
    assert result.missing_entities == []
    assert result.is_preserved is True


def test_empty_compressed_context_fails_when_entities_exist():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    original = [
        make_ranked_context(
            "FastAPI uses Python.",
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

    assert result.required_entities == [
        "FastAPI",
        "Python",
    ]

    assert result.preserved_entities == []

    assert result.missing_entities == [
        "FastAPI",
        "Python",
    ]

    assert result.is_preserved is False


def test_extractor_returning_none_is_supported():

    def extractor(_text):
        return None

    preserver = EntityPreserver(
        extractor
    )

    sentence = make_ranked_context(
        "No entities.",
        0.80,
        0.80,
        0.80,
        0,
    )

    result = preserver.identify_required_entities(
        [sentence]
    )

    assert result == []


def test_invalid_entity_value_is_rejected():

    def invalid_extractor(_text):
        return ["Python", 123]

    preserver = EntityPreserver(
        invalid_extractor
    )

    sentence = make_ranked_context(
        "Python is used.",
        0.90,
        0.90,
        0.90,
        0,
    )

    with pytest.raises(TypeError):
        preserver.identify_required_entities(
            [sentence]
        )


def test_blank_entity_values_are_ignored():

    def extractor(_text):
        return [
            "Python",
            "",
            "   ",
        ]

    preserver = EntityPreserver(
        extractor
    )

    sentence = make_ranked_context(
        "Python is used.",
        0.90,
        0.90,
        0.90,
        0,
    )

    result = preserver.identify_required_entities(
        [sentence]
    )

    assert result == [
        "Python",
    ]


def test_rejects_invalid_original_type():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    with pytest.raises(TypeError):
        preserver.validate(
            None,
            [],
        )


def test_rejects_invalid_compressed_type():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    with pytest.raises(TypeError):
        preserver.validate(
            [],
            None,
        )


def test_rejects_invalid_original_item():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    with pytest.raises(TypeError):
        preserver.validate(
            ["invalid"],
            [],
        )


def test_rejects_invalid_compressed_item():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    with pytest.raises(TypeError):
        preserver.validate(
            [],
            ["invalid"],
        )


def test_original_ranked_context_objects_are_not_modified():

    preserver = EntityPreserver(
        exact_entity_extractor
    )

    sentence = make_ranked_context(
        "FastAPI uses Python.",
        0.90,
        0.90,
        0.90,
        0,
    )

    original_text = sentence.result.text
    original_index = sentence.result.index

    preserver.validate(
        [sentence],
        [sentence],
    )

    assert sentence.result.text == original_text
    assert sentence.result.index == original_index
