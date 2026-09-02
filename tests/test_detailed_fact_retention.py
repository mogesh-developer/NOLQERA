from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.fact_preservation import (
    FactPreserver,
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


def test_detailed_technical_facts_are_identified():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "NOLQERA uses Python 3.11.9 "
            "for the evaluation environment.",
            0,
        ),
        make_ranked_context(
            "The embedding model produces vectors "
            "with dimension 384.",
            1,
        ),
        make_ranked_context(
            "The retrieval threshold is configured "
            "at 0.82 similarity.",
            2,
        ),
        make_ranked_context(
            "Context compression targets 40% "
            "token reduction.",
            3,
        ),
        make_ranked_context(
            "The benchmark contains 128 "
            "evaluation documents.",
            4,
        ),
        make_ranked_context(
            "The Phase 6 benchmark is scheduled "
            "for 2026.",
            5,
        ),
    ]

    facts = preserver.identify_required_facts(original)

    assert "3.11.9" in facts
    assert "384" in facts
    assert "0.82" in facts
    assert "40%" in facts
    assert "128" in facts
    assert "2026" in facts


def test_all_detailed_facts_are_preserved():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "NOLQERA uses Python 3.11.9 "
            "for the evaluation environment.",
            0,
        ),
        make_ranked_context(
            "The embedding model produces vectors "
            "with dimension 384.",
            1,
        ),
        make_ranked_context(
            "The retrieval threshold is configured "
            "at 0.82 similarity.",
            2,
        ),
        make_ranked_context(
            "Context compression targets 40% "
            "token reduction.",
            3,
        ),
        make_ranked_context(
            "The benchmark contains 128 "
            "evaluation documents.",
            4,
        ),
        make_ranked_context(
            "The Phase 6 benchmark is scheduled "
            "for 2026.",
            5,
        ),
    ]

    compressed = [
        original[0],
        original[1],
        original[2],
        original[3],
        original[4],
        original[5],
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is True
    assert result.missing_facts == []

    assert set(result.preserved_facts) == {
        "6",
        "3.11.9",
        "384",
        "0.82",
        "40%",
        "128",
        "2026",
    }


def test_missing_detailed_fact_is_detected():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "NOLQERA uses Python 3.11.9 "
            "for the evaluation environment.",
            0,
        ),
        make_ranked_context(
            "The retrieval threshold is configured "
            "at 0.82 similarity.",
            1,
        ),
        make_ranked_context(
            "Context compression targets 40% "
            "token reduction.",
            2,
        ),
    ]

    compressed = [
        original[0],
        original[2],
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is False

    assert result.missing_facts == [
        "0.82",
    ]


def test_detailed_decimal_and_version_facts_are_exact():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "Python version is 3.11.9.",
            0,
        ),
        make_ranked_context(
            "Similarity threshold is 0.82.",
            1,
        ),
        make_ranked_context(
            "Latency was measured at 48.46 ms.",
            2,
        ),
    ]

    facts = preserver.identify_required_facts(original)

    assert "3.11.9" in facts
    assert "0.82" in facts
    assert "48.46" in facts

    assert "3.11" not in facts
    assert "9" not in facts


def test_detailed_fact_loss_reports_exact_missing_values():

    preserver = FactPreserver()

    original = [
        make_ranked_context(
            "Python 3.11.9 was used.",
            0,
        ),
        make_ranked_context(
            "The system achieved 95% accuracy.",
            1,
        ),
        make_ranked_context(
            "The benchmark processed 128 documents.",
            2,
        ),
    ]

    compressed = [
        make_ranked_context(
            "Python 3.11.9 was used.",
            0,
        ),
        make_ranked_context(
            "The benchmark processed documents.",
            2,
        ),
    ]

    result = preserver.validate(
        original,
        compressed,
    )

    assert result.is_preserved is False

    assert result.missing_facts == [
        "95%",
        "128",
    ]
