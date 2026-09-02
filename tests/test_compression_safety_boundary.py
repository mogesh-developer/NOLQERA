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


def make_context(
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


def build_compressor(
    max_sentences: int = 3,
    require_preservation: bool = True,
) -> FinalContextCompressor:

    return FinalContextCompressor(
        redundancy_compressor=RedundancyAwareCompressor(
            exact_duplicate_checker=lambda cand, ret: cand.strip().casefold() == ret.strip().casefold()
        ),
        token_reduction_strategy=TokenReductionStrategy(
            token_counter=lambda text: len(text.split()),
        ),
        entity_extractor=lambda text: [
            word
            for word in (
                "Python",
                "FastAPI",
                "NOLQERA",
            )
            if word in text
        ],
        max_sentences=max_sentences,
        importance_threshold=0.70,
        require_preservation=require_preservation,
    )


def test_safe_compression_preserves_important_information():

    compressor = build_compressor(
        max_sentences=3,
        require_preservation=True,
    )

    contexts = [
        make_context(
            "NOLQERA uses Python 3.11.",
            0.98,
            0.95,
            0.98,
            0,
        ),
        make_context(
            "NOLQERA uses FastAPI for backend APIs.",
            0.95,
            0.90,
            0.95,
            1,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=20,
    )

    assert result.is_preserved is True

    assert result.fact_preservation.is_preserved is True
    assert result.entity_preservation.is_preserved is True

    assert "Python 3.11" in result.text
    assert "FastAPI" in result.text


def test_aggressive_compression_is_rejected_when_fact_is_lost():

    compressor = build_compressor(
        max_sentences=1,
        require_preservation=True,
    )

    contexts = [
        make_context(
            "NOLQERA uses Python 3.11.",
            0.98,
            0.95,
            0.98,
            0,
        ),
        make_context(
            "FastAPI version 2 is stable.",
            0.80,
            0.60,
            0.70,
            1,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="facts",
    ):
        compressor.compress(
            contexts,
            token_budget=4,
        )


def test_aggressive_compression_is_rejected_when_entity_is_lost():

    compressor = build_compressor(
        max_sentences=1,
        require_preservation=True,
    )

    contexts = [
        make_context(
            "Python is fast.",
            0.98,
            0.95,
            0.98,
            0,
        ),
        make_context(
            "FastAPI provides backend APIs.",
            0.80,
            0.60,
            0.70,
            1,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="entities",
    ):
        compressor.compress(
            contexts,
            token_budget=10,
        )


def test_aggressive_compression_can_continue_when_gate_is_disabled():

    compressor = build_compressor(
        max_sentences=1,
        require_preservation=False,
    )

    contexts = [
        make_context(
            "Python is fast.",
            0.98,
            0.95,
            0.98,
            0,
        ),
        make_context(
            "FastAPI provides backend APIs.",
            0.80,
            0.90,
            0.85,
            1,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=10,
    )

    assert result.is_preserved is False

    assert result.selected == [
        contexts[0]
    ]

    assert "Python" in result.text
    assert "FastAPI" not in result.text


def test_zero_token_budget_does_not_silently_claim_preservation():

    compressor = build_compressor(
        max_sentences=3,
        require_preservation=True,
    )

    contexts = [
        make_context(
            "NOLQERA uses Python 3.11.",
            0.98,
            0.95,
            0.98,
            0,
        ),
    ]

    with pytest.raises(
        ValueError,
    ):
        compressor.compress(
            contexts,
            token_budget=0,
        )