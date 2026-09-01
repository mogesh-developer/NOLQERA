
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)


@dataclass(frozen=True)
class TokenReductionResult:
    """
    Result of token-budget based context selection.
    """

    selected: List[RankedContext]
    original_tokens: int
    compressed_tokens: int
    token_reduction: int
    reduction_percentage: float
    budget: int


class TokenReductionStrategy:
    """
    Selects already-ranked context items under a strict token budget.

    This class does not perform:
        - relevance scoring
        - importance scoring
        - ranking
        - semantic compression
        - tokenization

    Those responsibilities remain with existing NOLQERA modules.

    `token_counter` is injected so the strategy can later use the
    project's real tokenizer without changing this component.
    """

    def __init__(
        self,
        token_counter: Callable[[str], int],
    ) -> None:

        if not callable(token_counter):
            raise TypeError(
                "token_counter must be callable"
            )

        self.token_counter = token_counter

    def _count_tokens(
        self,
        text: str,
    ) -> int:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        count = self.token_counter(text)

        if not isinstance(count, int):
            raise TypeError(
                "token_counter must return an integer"
            )

        if count < 0:
            raise ValueError(
                "token_counter must return a non-negative integer"
            )

        return count

    def count_context_tokens(
        self,
        contexts: Sequence[RankedContext],
    ) -> int:
        """
        Count total tokens in a context collection.
        """

        if not isinstance(contexts, (list, tuple)):
            raise TypeError(
                "contexts must be a list or tuple"
            )

        for context in contexts:
            if not isinstance(context, RankedContext):
                raise TypeError(
                    "contexts must contain RankedContext objects"
                )

        return sum(
            self._count_tokens(
                context.result.text
            )
            for context in contexts
        )

    def select(
        self,
        contexts: Sequence[RankedContext],
        budget: int,
    ) -> TokenReductionResult:
        """
        Select contexts greedily in the order supplied.

        The input is expected to already be prioritized by the
        existing ContextPrioritizer.

        A sentence is selected only when the resulting total
        remains within the token budget.

        If an individual sentence exceeds the budget, it is skipped
        rather than partially truncated.
        """

        if not isinstance(contexts, (list, tuple)):
            raise TypeError(
                "contexts must be a list or tuple"
            )

        if not isinstance(budget, int):
            raise TypeError(
                "budget must be an integer"
            )

        if budget < 0:
            raise ValueError(
                "budget must be non-negative"
            )

        for context in contexts:
            if not isinstance(context, RankedContext):
                raise TypeError(
                    "contexts must contain RankedContext objects"
                )

        original_tokens = self.count_context_tokens(
            contexts
        )

        selected: List[RankedContext] = []
        compressed_tokens = 0

        for context in contexts:

            sentence_tokens = self._count_tokens(
                context.result.text
            )

            if (
                compressed_tokens
                + sentence_tokens
                <= budget
            ):
                selected.append(context)
                compressed_tokens += sentence_tokens

        token_reduction = (
            original_tokens
            - compressed_tokens
        )

        if original_tokens == 0:
            reduction_percentage = 0.0
        else:
            reduction_percentage = (
                token_reduction
                / original_tokens
            ) * 100.0

        return TokenReductionResult(
            selected=selected,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            token_reduction=token_reduction,
            reduction_percentage=reduction_percentage,
            budget=budget,
        )
