from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.entity_preservation import (
    EntityPreserver,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def make_ranked_context(
    text: str,
    index: int,
    relevance: float = 0.90,
    importance: float = 0.90,
    ranking: float = 0.90,
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


def sample_entity_extractor(text: str):
    known_entities = {
        "NOLQERA",
        "Python",
        "FastAPI",
        "PyTorch",
        "TensorFlow",
        "BERT",
    }
    return [
        entity
        for entity in known_entities
        if entity.casefold() in text.casefold()
    ]


def test_detailed_entities_are_identified():

    preserver = EntityPreserver(sample_entity_extractor)

    original = [
        make_ranked_context(
            "NOLQERA uses Python and FastAPI "
            "for backend development.",
            0,
        ),
        make_ranked_context(
            "PyTorch and TensorFlow are used "
            "for machine learning experiments.",
            1,
        ),
        make_ranked_context(
            "BERT is evaluated alongside "
            "the NOLQERA retrieval pipeline.",
            2,
        ),
    ]

    entities = preserver.identify_required_entities(original)

    assert "NOLQERA" in entities
    assert "Python" in entities
    assert "FastAPI" in entities
    assert "PyTorch" in entities
    assert "TensorFlow" in entities
    assert "BERT" in entities


def test_all_detailed_entities_are_preserved():

    preserver = EntityPreserver(sample_entity_extractor)

    original = [
        make_ranked_context(
            "NOLQERA uses Python and FastAPI "
            "for backend development.",
            0,
        ),
        make_ranked_context(
            "PyTorch and TensorFlow are used "
            "for machine learning experiments.",
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
    assert result.missing_entities == []


def test_missing_detailed_entity_is_detected():

    preserver = EntityPreserver(sample_entity_extractor)

    original = [
        make_ranked_context(
            "NOLQERA uses Python and FastAPI "
            "for backend development.",
            0,
        ),
        make_ranked_context(
            "PyTorch and TensorFlow are used "
            "for machine learning experiments.",
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

    assert result.is_preserved is False

    assert "PyTorch" in result.missing_entities
    assert "TensorFlow" in result.missing_entities


def test_multiple_entities_in_single_fact_are_preserved():

    preserver = EntityPreserver(sample_entity_extractor)

    original = [
        make_ranked_context(
            "NOLQERA integrates Python, FastAPI, "
            "PyTorch, and TensorFlow.",
            0,
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

    assert set(result.preserved_entities) >= {
        "NOLQERA",
        "Python",
        "FastAPI",
        "PyTorch",
        "TensorFlow",
    }


def test_exact_missing_entities_are_reported():

    preserver = EntityPreserver(sample_entity_extractor)

    original = [
        make_ranked_context(
            "NOLQERA uses Python and FastAPI.",
            0,
        ),
        make_ranked_context(
            "PyTorch and TensorFlow power the "
            "machine learning layer.",
            1,
        ),
    ]

    compressed = [
        make_ranked_context(
            "NOLQERA uses Python.",
            0,
        ),
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is False

    assert "FastAPI" in result.missing_entities
    assert "PyTorch" in result.missing_entities
    assert "TensorFlow" in result.missing_entities