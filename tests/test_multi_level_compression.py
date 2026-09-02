import pytest

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


def make_context(text, index, score=0.90, importance=0.90):
    result = SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )

    return RankedContext(
        result=result,
        relevance_score=score,
        importance_score=importance,
        ranking_score=score,
    )


def build_compressor(require_preservation=False):
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
                "TensorFlow",
            )
            if entity in text
        ],
        max_sentences=10,
        importance_threshold=0.70,
        require_preservation=require_preservation,
    )


@pytest.fixture
def contexts():
    return [
        make_context(
            "NOLQERA is a Python retrieval system.",
            0,
            0.99,
            0.98,
        ),
        make_context(
            "NOLQERA uses FastAPI for backend APIs.",
            1,
            0.95,
            0.90,
        ),
        make_context(
            "NOLQERA supports PyTorch embeddings.",
            2,
            0.90,
            0.85,
        ),
        make_context(
            "NOLQERA can also work with TensorFlow.",
            3,
            0.85,
            0.80,
        ),
        make_context(
            "Redundant implementation details can be removed.",
            4,
            0.50,
            0.40,
        ),
    ]


def get_result(
    compressor,
    contexts,
    token_budget,
):
    return compressor.compress(
        contexts,
        token_budget=token_budget,
    )


def test_level_0_minimal_compression_preserves_information(contexts):

    compressor = build_compressor()

    result = get_result(
        compressor,
        contexts,
        token_budget=100,
    )

    assert result.is_preserved is True

    assert "NOLQERA" in result.text
    assert "Python" in result.text


def test_level_1_mild_compression_preserves_core_information(contexts):

    compressor = build_compressor()

    result = get_result(
        compressor,
        contexts,
        token_budget=60,
    )

    assert "NOLQERA" in result.text
    assert "Python" in result.text

    assert len(result.information_preservation.preserved) > 0


def test_level_2_medium_compression_still_preserves_core_fact(contexts):

    compressor = build_compressor()

    result = get_result(
        compressor,
        contexts,
        token_budget=35,
    )

    assert "NOLQERA" in result.text
    assert "Python" in result.text

    assert result.fact_preservation.is_preserved is True


def test_level_3_aggressive_compression_detects_information_loss(contexts):

    compressor = build_compressor()

    result = get_result(
        compressor,
        contexts,
        token_budget=15,
    )

    assert result.is_preserved is False


def test_level_4_extreme_compression_does_not_claim_success(contexts):

    compressor = build_compressor()

    result = get_result(
        compressor,
        contexts,
        token_budget=5,
    )

    assert result.is_preserved is False


def test_preservation_degrades_monotonically(contexts):

    compressor = build_compressor()

    budgets = [
        100,
        60,
        35,
        15,
        5,
    ]

    preservation_scores = []

    for budget in budgets:

        result = get_result(
            compressor,
            contexts,
            token_budget=budget,
        )

        preservation_scores.append(
            len(result.information_preservation.preserved)
        )

    for earlier, later in zip(
        preservation_scores,
        preservation_scores[1:],
    ):
        assert later <= earlier