from __future__ import annotations

import pytest

from nolqera.intelligence.context_optimization.adaptive_context_compressor import (
    AdaptiveContextCompressor,
)
from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.final_context_compressor import (
    FinalContextCompressionResult,
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


def word_counter(text: str) -> int:
    return len(text.split())


def entity_extractor(text: str) -> list[str]:
    known = ["Python", "FastAPI", "MongoDB"]
    return [e for e in known if e.casefold() in text.casefold()]


def exact_duplicate(candidate: str, retained: str) -> bool:
    return candidate.strip().casefold() == retained.strip().casefold()


def build_adaptive_compressor(
    max_sentences: int = 3,
    require_preservation: bool = True,
) -> AdaptiveContextCompressor:
    redundancy_compressor = RedundancyAwareCompressor(
        exact_duplicate_checker=exact_duplicate
    )
    token_strategy = TokenReductionStrategy(token_counter=word_counter)

    inner = FinalContextCompressor(
        redundancy_compressor=redundancy_compressor,
        token_reduction_strategy=token_strategy,
        entity_extractor=entity_extractor,
        max_sentences=max_sentences,
        importance_threshold=0.70,
        require_preservation=require_preservation,
    )

    return AdaptiveContextCompressor(compressor=inner)


def test_adaptive_compressor_type_checks():
    with pytest.raises(TypeError):
        AdaptiveContextCompressor("invalid")


def test_adaptive_compress_empty_contexts():
    adaptive = build_adaptive_compressor()
    result = adaptive.compress([], token_budget=10)
    assert isinstance(result, FinalContextCompressionResult)
    assert result.selected == []
    assert result.text == ""
    assert result.is_preserved is True


def test_adaptive_compress_auto_mode():
    adaptive = build_adaptive_compressor(max_sentences=2)
    contexts = [
        make_context("Python 3.11 is fast.", 0.95, 0.90, 0.95, 0),
        make_context("FastAPI is lightweight.", 0.90, 0.85, 0.90, 1),
        make_context("MongoDB stores data.", 0.85, 0.80, 0.85, 2),
    ]

    result = adaptive.compress(
        contexts,
        token_budget=100,
        max_sentences="auto",
    )

    assert isinstance(result, FinalContextCompressionResult)
    assert result.is_preserved is True
    assert len(result.selected) == 3


def test_adaptive_compress_restores_missing_important_sentences():
    adaptive = build_adaptive_compressor(max_sentences=1)

    contexts = [
        make_context("Python is easy.", 0.95, 0.80, 0.95, 0),
        make_context("FastAPI handles web requests.", 0.90, 0.85, 0.90, 1),
    ]

    result = adaptive.compress(
        contexts,
        token_budget=100,
        max_sentences=1,
    )

    assert result.is_preserved is True
    assert len(result.selected) == 2
