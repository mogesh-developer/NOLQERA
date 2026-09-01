
import pytest

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.token_reduction import (
    TokenReductionResult,
    TokenReductionStrategy,
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


def word_counter(text: str) -> int:
    return len(text.split())


def character_counter(text: str) -> int:
    return len(text)


def test_token_counter_is_required():

    with pytest.raises(TypeError):
        TokenReductionStrategy(None)


def test_count_context_tokens_exactly():

    strategy = TokenReductionStrategy(
        word_counter
    )

    contexts = [
        make_ranked_context(
            "Python is fast",
            0.9,
            0.9,
            0.9,
            0,
        ),
        make_ranked_context(
            "MongoDB stores data",
            0.8,
            0.8,
            0.8,
            1,
        ),
    ]

    result = strategy.count_context_tokens(
        contexts
    )

    assert result == 6


def test_selects_sentences_within_budget():

    strategy = TokenReductionStrategy(
        word_counter
    )

    contexts = [
        make_ranked_context(
            "Python is fast",
            0.9,
            0.9,
            0.9,
            0,
        ),  # 3

        make_ranked_context(
            "MongoDB stores data",
            0.8,
            0.8,
            0.8,
            1,
        ),  # 3

        make_ranked_context(
            "FastAPI framework",
            0.7,
            0.7,
            0.7,
            2,
        ),  # 2
    ]

    result = strategy.select(
        contexts,
        budget=6,
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "Python is fast",
        "MongoDB stores data",
    ]

    assert result.original_tokens == 8
    assert result.compressed_tokens == 6
    assert result.token_reduction == 2
    assert result.reduction_percentage == 25.0
    assert result.budget == 6


def test_budget_prevents_overflow():

    strategy = TokenReductionStrategy(
        word_counter
    )

    contexts = [
        make_ranked_context(
            "one two three",
            0.9,
            0.9,
            0.9,
            0,
        ),
        make_ranked_context(
            "four five six",
            0.8,
            0.8,
            0.8,
            1,
        ),
    ]

    result = strategy.select(
        contexts,
        budget=4,
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "one two three",
    ]

    assert result.compressed_tokens == 3
    assert result.compressed_tokens <= 4


def test_oversized_sentence_is_skipped():

    strategy = TokenReductionStrategy(
        word_counter
    )

    contexts = [
        make_ranked_context(
            "one two three four five",
            0.9,
            0.9,
            0.9,
            0,
        ),
        make_ranked_context(
            "small sentence",
            0.8,
            0.8,
            0.8,
            1,
        ),
    ]

    result = strategy.select(
        contexts,
        budget=2,
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "small sentence",
    ]

    assert result.compressed_tokens == 2


def test_selection_is_greedy_in_existing_order():

    strategy = TokenReductionStrategy(
        word_counter
    )

    contexts = [
        make_ranked_context(
            "one two three",
            0.9,
            0.9,
            0.9,
            0,
        ),
        make_ranked_context(
            "four five",
            0.8,
            0.8,
            0.8,
            1,
        ),
        make_ranked_context(
            "six seven",
            0.7,
            0.7,
            0.7,
            2,
        ),
    ]

    result = strategy.select(
        contexts,
        budget=5,
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "one two three",
        "four five",
    ]


def test_later_sentence_can_fit_after_skipping_large_sentence():

    strategy = TokenReductionStrategy(
        word_counter
    )

    contexts = [
        make_ranked_context(
            "one two three four",
            0.9,
            0.9,
            0.9,
            0,
        ),
        make_ranked_context(
            "five six",
            0.8,
            0.8,
            0.8,
            1,
        ),
        make_ranked_context(
            "seven eight",
            0.7,
            0.7,
            0.7,
            2,
        ),
    ]

    result = strategy.select(
        contexts,
        budget=4,
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "one two three four",
    ]


def test_zero_budget_selects_nothing():

    strategy = TokenReductionStrategy(
        word_counter
    )

    contexts = [
        make_ranked_context(
            "one two",
            0.9,
            0.9,
            0.9,
            0,
        )
    ]

    result = strategy.select(
        contexts,
        budget=0,
    )

    assert result.selected == []
    assert result.original_tokens == 2
    assert result.compressed_tokens == 0
    assert result.token_reduction == 2
    assert result.reduction_percentage == 100.0


def test_empty_context():

    strategy = TokenReductionStrategy(
        word_counter
    )

    result = strategy.select(
        [],
        budget=10,
    )

    assert result.selected == []
    assert result.original_tokens == 0
    assert result.compressed_tokens == 0
    assert result.token_reduction == 0
    assert result.reduction_percentage == 0.0
    assert result.budget == 10


def test_full_budget_keeps_everything():

    strategy = TokenReductionStrategy(
        word_counter
    )

    contexts = [
        make_ranked_context(
            "one two",
            0.9,
            0.9,
            0.9,
            0,
        ),
        make_ranked_context(
            "three four",
            0.8,
            0.8,
            0.8,
            1,
        ),
    ]

    result = strategy.select(
        contexts,
        budget=4,
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "one two",
        "three four",
    ]

    assert result.original_tokens == 4
    assert result.compressed_tokens == 4
    assert result.token_reduction == 0
    assert result.reduction_percentage == 0.0


def test_character_tokenizer_can_be_injected():

    strategy = TokenReductionStrategy(
        character_counter
    )

    contexts = [
        make_ranked_context(
            "abcd",
            0.9,
            0.9,
            0.9,
            0,
        ),
        make_ranked_context(
            "efgh",
            0.8,
            0.8,
            0.8,
            1,
        ),
    ]

    result = strategy.select(
        contexts,
        budget=4,
    )

    assert [
        item.result.text
        for item in result.selected
    ] == [
        "abcd",
    ]

    assert result.original_tokens == 8
    assert result.compressed_tokens == 4
    assert result.token_reduction == 4
    assert result.reduction_percentage == 50.0


def test_token_counter_must_return_integer():

    def invalid_counter(_text):
        return 1.5

    strategy = TokenReductionStrategy(
        invalid_counter
    )

    contexts = [
        make_ranked_context(
            "test",
            0.9,
            0.9,
            0.9,
            0,
        )
    ]

    with pytest.raises(TypeError):
        strategy.select(
            contexts,
            10,
        )


def test_token_counter_cannot_return_negative():

    def invalid_counter(_text):
        return -1

    strategy = TokenReductionStrategy(
        invalid_counter
    )

    contexts = [
        make_ranked_context(
            "test",
            0.9,
            0.9,
            0.9,
            0,
        )
    ]

    with pytest.raises(ValueError):
        strategy.select(
            contexts,
            10,
        )


def test_invalid_budget_type_is_rejected():

    strategy = TokenReductionStrategy(
        word_counter
    )

    with pytest.raises(TypeError):
        strategy.select(
            [],
            10.5,
        )


def test_negative_budget_is_rejected():

    strategy = TokenReductionStrategy(
        word_counter
    )

    with pytest.raises(ValueError):
        strategy.select(
            [],
            -1,
        )


def test_invalid_context_type_is_rejected():

    strategy = TokenReductionStrategy(
        word_counter
    )

    with pytest.raises(TypeError):
        strategy.select(
            None,
            10,
        )


def test_invalid_context_item_is_rejected():

    strategy = TokenReductionStrategy(
        word_counter
    )

    with pytest.raises(TypeError):
        strategy.select(
            ["invalid"],
            10,
        )


def test_result_type_is_exact():

    strategy = TokenReductionStrategy(
        word_counter
    )

    context = make_ranked_context(
        "Python",
        0.9,
        0.9,
        0.9,
        0,
    )

    result = strategy.select(
        [context],
        10,
    )

    assert isinstance(
        result,
        TokenReductionResult,
    )


def test_original_context_is_not_modified():

    strategy = TokenReductionStrategy(
        word_counter
    )

    context = make_ranked_context(
        "Python is fast",
        0.9,
        0.8,
        0.85,
        0,
    )

    original_text = context.result.text
    original_score = context.ranking_score

    strategy.select(
        [context],
        2,
    )

    assert context.result.text == original_text
    assert context.ranking_score == original_score

