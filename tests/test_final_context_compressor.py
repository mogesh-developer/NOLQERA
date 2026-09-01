
import pytest

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


def entity_extractor(text: str):

    known_entities = [
        "Python",
        "FastAPI",
        "MongoDB",
        "NOLQERA",
    ]

    return [
        entity
        for entity in known_entities
        if entity.casefold() in text.casefold()
    ]


def exact_duplicate(
    candidate: str,
    retained: str,
) -> bool:
    return (
        candidate.strip().casefold()
        == retained.strip().casefold()
    )


def build_compressor(
    max_sentences=3,
    require_preservation=False,
):

    redundancy_compressor = (
        RedundancyAwareCompressor(
            exact_duplicate_checker=exact_duplicate
        )
    )

    token_strategy = TokenReductionStrategy(
        token_counter=word_counter
    )

    return FinalContextCompressor(
        redundancy_compressor=redundancy_compressor,
        token_reduction_strategy=token_strategy,
        entity_extractor=entity_extractor,
        max_sentences=max_sentences,
        importance_threshold=0.70,
        require_preservation=require_preservation,
    )


def test_final_result_type():

    compressor = build_compressor()

    contexts = [
        make_context(
            "Python is fast.",
            0.95,
            0.90,
            0.95,
            0,
        )
    ]

    result = compressor.compress(
        contexts,
        token_budget=10,
    )

    assert isinstance(
        result,
        FinalContextCompressionResult,
    )


def test_empty_context_returns_exact_empty_result():

    compressor = build_compressor()

    result = compressor.compress(
        [],
        token_budget=10,
    )

    assert result.selected == []
    assert result.text == ""
    assert result.original_count == 0
    assert result.final_count == 0
    assert result.original_tokens == 0
    assert result.compressed_tokens == 0
    assert result.token_reduction == 0
    assert result.reduction_percentage == 0.0
    assert result.removed_by_redundancy == []
    assert result.is_preserved is True


def test_exact_duplicate_is_removed():

    compressor = build_compressor(
        max_sentences=3,
        require_preservation=True,
    )

    contexts = [
        make_context(
            "Python 3.11 improves performance.",
            0.95,
            0.90,
            0.95,
            0,
        ),
        make_context(
            "Python 3.11 improves performance.",
            0.80,
            0.80,
            0.80,
            1,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=20,
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "Python 3.11 improves performance.",
    ]

    assert result.text == (
        "Python 3.11 improves performance."
    )

    assert [
        item.result.text
        for item in result.removed_by_redundancy
    ] == [
        "Python 3.11 improves performance.",
    ]

    assert result.final_count == 1


def test_final_context_preserves_entity_and_fact():

    compressor = build_compressor(
        max_sentences=3
    )

    contexts = [
        make_context(
            "Python 3.11 improves API performance.",
            0.98,
            0.95,
            0.98,
            0,
        ),
        make_context(
            "FastAPI provides asynchronous endpoints.",
            0.96,
            0.90,
            0.96,
            1,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=20,
    )

    assert result.text == (
        "Python 3.11 improves API performance. "
        "FastAPI provides asynchronous endpoints."
    )

    assert result.information_preservation.is_preserved is True

    assert result.entity_preservation.is_preserved is True

    assert result.fact_preservation.is_preserved is True

    assert result.fact_preservation.required_facts == [
        "3.11",
    ]

    assert result.fact_preservation.preserved_facts == [
        "3.11",
    ]

    assert result.fact_preservation.missing_facts == []

    assert result.entity_preservation.required_entities == [
        "FastAPI",
        "Python",
    ]

    assert result.entity_preservation.preserved_entities == [
        "FastAPI",
        "Python",
    ]

    assert result.entity_preservation.missing_entities == []

    assert result.is_preserved is True


def test_token_budget_reduces_context_exactly():

    compressor = build_compressor(
        max_sentences=3
    )

    contexts = [
        make_context(
            "Python 3.11 is fast.",
            0.98,
            0.95,
            0.98,
            0,
        ),
        make_context(
            "FastAPI supports async.",
            0.90,
            0.90,
            0.90,
            1,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=5,
    )

    assert result.selected == [
        contexts[0]
    ]

    assert result.text == (
        "Python 3.11 is fast."
    )

    assert result.original_tokens == 7
    assert result.compressed_tokens == 4
    assert result.token_reduction == 3
    assert result.reduction_percentage == pytest.approx(
        42.857142857142854
    )


def test_original_context_order_is_preserved():

    compressor = build_compressor(
        max_sentences=3
    )

    contexts = [
        make_context(
            "FastAPI supports async.",
            0.80,
            0.80,
            0.80,
            0,
        ),
        make_context(
            "Python 3.11 improves speed.",
            0.95,
            0.95,
            0.95,
            1,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=20,
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "FastAPI supports async.",
        "Python 3.11 improves speed.",
    ]

    assert result.text == (
        "FastAPI supports async. "
        "Python 3.11 improves speed."
    )


def test_high_priority_sentence_wins_under_token_budget():

    compressor = build_compressor(
        max_sentences=3
    )

    low = make_context(
        "FastAPI supports async.",
        0.70,
        0.70,
        0.70,
        0,
    )

    high = make_context(
        "Python 3.11 is fast.",
        0.98,
        0.95,
        0.98,
        1,
    )

    result = compressor.compress(
        [low, high],
        token_budget=4,
    )

    assert result.selected == [
        high
    ]

    assert result.text == (
        "Python 3.11 is fast."
    )


def test_missing_fact_fails_final_gate():

    compressor = build_compressor(
        max_sentences=1,
        require_preservation=True,
    )

    contexts = [
        make_context(
            "Python 3.11 is fast.",
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


def test_missing_entity_fails_final_gate():

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
            "FastAPI provides APIs.",
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


def test_missing_important_information_fails_final_gate():

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
            "FastAPI handles requests.",
            0.80,
            0.90,
            0.85,
            1,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="important information",
    ):
        compressor.compress(
            contexts,
            token_budget=10,
        )


def test_preservation_gate_can_be_disabled():

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
            "FastAPI provides APIs.",
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

    assert result.text == (
        "Python is fast."
    )


def test_max_sentences_is_respected():

    compressor = build_compressor(
        max_sentences=2
    )

    contexts = [
        make_context(
            "Python is fast.",
            0.95,
            0.90,
            0.95,
            0,
        ),
        make_context(
            "FastAPI is lightweight.",
            0.90,
            0.85,
            0.90,
            1,
        ),
        make_context(
            "MongoDB stores data.",
            0.85,
            0.80,
            0.85,
            2,
        ),
    ]

    result = compressor.compress(
        contexts,
        token_budget=20,
        )

    assert len(result.selected) == 2

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "Python is fast.",
        "FastAPI is lightweight.",
    ]


def test_invalid_token_budget_is_rejected():

    compressor = build_compressor()

    with pytest.raises(TypeError):
        compressor.compress(
            [],
            10.5,
        )


def test_negative_token_budget_is_rejected():

    compressor = build_compressor()

    with pytest.raises(ValueError):
        compressor.compress(
            [],
            -1,
        )


def test_invalid_context_type_is_rejected():

    compressor = build_compressor()

    with pytest.raises(TypeError):
        compressor.compress(
            None,
            10,
        )


def test_invalid_context_item_is_rejected():

    compressor = build_compressor()

    with pytest.raises(TypeError):
        compressor.compress(
            ["invalid"],
            10,
        )


def test_invalid_max_sentences_is_rejected():

    with pytest.raises(ValueError):

        FinalContextCompressor(
            redundancy_compressor=(
                RedundancyAwareCompressor(
                    exact_duplicate_checker=exact_duplicate
                )
            ),
            token_reduction_strategy=(
                TokenReductionStrategy(
                    word_counter
                )
            ),
            entity_extractor=entity_extractor,
            max_sentences=0,
        )


def test_preservation_flags_are_all_true_for_valid_context():

    compressor = build_compressor()

    contexts = [
        make_context(
            "Python 3.11 is fast.",
            0.95,
            0.90,
            0.95,
            0,
        )
    ]

    result = compressor.compress(
        contexts,
        token_budget=10,
    )

    assert result.information_preservation.is_preserved is True
    assert result.entity_preservation.is_preserved is True
    assert result.fact_preservation.is_preserved is True
    assert result.is_preserved is True

