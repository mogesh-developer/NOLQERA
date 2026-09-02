from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.entity_preservation import (
    EntityPreserver,
)
from nolqera.intelligence.context_optimization.fact_preservation import (
    FactPreserver,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def make_ranked_context(text: str, index: int = 0) -> RankedContext:
    result = SemanticSearchResult(
        text=text,
        score=0.90,
        index=index,
    )
    return RankedContext(
        result=result,
        relevance_score=0.90,
        importance_score=0.90,
        ranking_score=0.90,
    )


def sample_entity_extractor(text: str):
    known_entities = {
        "NOLQERA",
        "Python",
        "FastAPI",
    }
    return [
        entity
        for entity in known_entities
        if entity.casefold() in text.casefold()
    ]


def test_missing_fact_is_identified():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "NOLQERA uses Python 3.11 "
            "with an embedding dimension of 384.",
            0,
        )
    ]

    compressed = [
        make_ranked_context(
            "NOLQERA uses Python.",
            0,
        )
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is False

    assert "3.11" in result.missing_facts
    assert "384" in result.missing_facts


def test_preserved_facts_produce_no_failure():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "NOLQERA uses Python 3.11 "
            "with an embedding dimension of 384.",
            0,
        )
    ]

    compressed = original

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is True
    assert result.missing_facts == []


def test_missing_entity_is_identified():

    preserver = EntityPreserver(sample_entity_extractor)

    original = [
        make_ranked_context(
            "NOLQERA uses Python and FastAPI "
            "for backend processing.",
            0,
        )
    ]

    compressed = [
        make_ranked_context(
            "NOLQERA uses Python.",
            0,
        )
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is False
    assert "FastAPI" in result.missing_entities


def test_preserved_entities_produce_no_failure():

    preserver = EntityPreserver(sample_entity_extractor)

    original = [
        make_ranked_context(
            "NOLQERA uses Python and FastAPI "
            "for backend processing.",
            0,
        )
    ]

    compressed = original

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is True
    assert result.missing_entities == []