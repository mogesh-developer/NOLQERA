from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.final_context_compressor import (
    FinalContextCompressor,
)
from nolqera.intelligence.context_optimization.redundancy_aware_compression import (
    RedundancyAwareCompressor,
)
from nolqera.intelligence.context_optimization.token_reduction import (
    TokenReductionStrategy,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def make_context(
    text,
    index,
    relevance,
    importance,
    ranking,
):
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


def build_compressor():
    return FinalContextCompressor(
        redundancy_compressor=RedundancyAwareCompressor(
            exact_duplicate_checker=lambda cand, ret: cand.strip().casefold() == ret.strip().casefold()
        ),
        token_reduction_strategy=TokenReductionStrategy(
            token_counter=lambda text: len(text.split()),
        ),
        entity_extractor=lambda text: [
            entity
            for entity in (
                "NOLQERA",
                "Python",
                "FastAPI",
                "PyTorch",
            )
            if entity in text
        ],
        max_sentences=10,
        importance_threshold=0.70,
        require_preservation=False,
    )


def test_high_ranked_important_fact_is_preserved():

    compressor = build_compressor()

    contexts = [
        make_context(
            "NOLQERA uses Python 3.11.",
            0,
            0.98,
            0.98,
            0.99,
        ),
        make_context(
            "This is general implementation information.",
            1,
            0.50,
            0.40,
            0.30,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=20,
    )

    assert "Python 3.11" in result.text


def test_important_fact_can_outweigh_lower_ranking():

    compressor = build_compressor()

    contexts = [
        make_context(
            "General implementation details are available.",
            0,
            0.95,
            0.40,
            0.95,
        ),
        make_context(
            "NOLQERA uses Python 3.11.",
            1,
            0.70,
            0.98,
            0.70,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=12,
    )

    assert "Python 3.11" in result.text


def test_redundant_high_ranked_content_does_not_break_preservation():

    compressor = build_compressor()

    contexts = [
        make_context(
            "NOLQERA uses Python 3.11.",
            0,
            0.98,
            0.98,
            0.99,
        ),
        make_context(
            "NOLQERA uses Python 3.11.",
            1,
            0.97,
            0.60,
            0.98,
        ),
        make_context(
            "NOLQERA uses Python 3.11.",
            2,
            0.96,
            0.60,
            0.97,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=15,
    )

    assert "Python 3.11" in result.text

    assert result.text.count(
        "Python 3.11"
    ) >= 1


def test_ranking_changes_do_not_change_core_fact():

    compressor = build_compressor()

    contexts_a = [
        make_context(
            "NOLQERA uses Python 3.11.",
            0,
            0.90,
            0.98,
            0.95,
        ),
        make_context(
            "FastAPI provides backend APIs.",
            1,
            0.90,
            0.80,
            0.70,
        ),
    ]

    contexts_b = [
        make_context(
            "NOLQERA uses Python 3.11.",
            0,
            0.90,
            0.98,
            0.60,
        ),
        make_context(
            "FastAPI provides backend APIs.",
            1,
            0.90,
            0.80,
            0.95,
        ),
    ]

    result_a = compressor.compress(
        contexts_a,
        token_budget=15,
    )

    result_b = compressor.compress(
        contexts_b,
        token_budget=15,
    )

    assert "Python 3.11" in result_a.text
    assert "Python 3.11" in result_b.text


def test_preservation_metadata_is_available_after_ranking():

    compressor = build_compressor()

    contexts = [
        make_context(
            "NOLQERA uses Python 3.11.",
            0,
            0.98,
            0.98,
            0.99,
        ),
        make_context(
            "FastAPI provides backend APIs.",
            1,
            0.90,
            0.85,
            0.80,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=25,
    )

    assert result.information_preservation is not None
    assert result.entity_preservation is not None
    assert result.fact_preservation is not None