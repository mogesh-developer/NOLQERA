
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

from nolqera.intelligence.context_optimization.context_prioritization import (
    ContextPrioritizer,
)
from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.entity_preservation import (
    EntityPreservationResult,
    EntityPreserver,
)
from nolqera.intelligence.context_optimization.fact_preservation import (
    FactPreservationResult,
    FactPreserver,
)
from nolqera.intelligence.context_optimization.information_preservation import (
    InformationPreservationResult,
    InformationPreserver,
)
from nolqera.intelligence.context_optimization.redundancy_aware_compression import (
    RedundancyAwareCompressor,
)
from nolqera.intelligence.context_optimization.sentence_selection import (
    SentenceSelector,
)
from nolqera.intelligence.context_optimization.token_reduction import (
    TokenReductionResult,
    TokenReductionStrategy,
)


@dataclass(frozen=True)
class FinalContextCompressionResult:
    """
    Complete result of the final Phase 4 compression pipeline.
    """

    selected: List[RankedContext]
    text: str

    original_count: int
    final_count: int

    original_tokens: int
    compressed_tokens: int
    token_reduction: int
    reduction_percentage: float

    removed_by_redundancy: List[RankedContext]

    information_preservation: InformationPreservationResult
    entity_preservation: EntityPreservationResult
    fact_preservation: FactPreservationResult

    is_preserved: bool


class FinalContextCompressor:
    """
    Final Phase 4 context-compression orchestrator.

    This class does not implement new intelligence.

    It composes existing NOLQERA components:

        ContextPrioritizer
        RedundancyAwareCompressor
        SentenceSelector
        TokenReductionStrategy
        InformationPreserver
        EntityPreserver
        FactPreserver

    The final output is accepted only when configured preservation
    checks succeed.
    """

    def __init__(
        self,
        redundancy_compressor: RedundancyAwareCompressor,
        token_reduction_strategy: TokenReductionStrategy,
        entity_extractor: Callable[
            [str],
            Iterable[str],
        ],
        max_sentences: int = 3,
        importance_threshold: float = 0.70,
        require_preservation: bool = True,
    ) -> None:

        if not isinstance(
            redundancy_compressor,
            RedundancyAwareCompressor,
        ):
            raise TypeError(
                "redundancy_compressor must be "
                "a RedundancyAwareCompressor"
            )

        if not isinstance(
            token_reduction_strategy,
            TokenReductionStrategy,
        ):
            raise TypeError(
                "token_reduction_strategy must be "
                "a TokenReductionStrategy"
            )

        if not callable(entity_extractor):
            raise TypeError(
                "entity_extractor must be callable"
            )

        if not isinstance(max_sentences, int):
            raise TypeError(
                "max_sentences must be an integer"
            )

        if max_sentences <= 0:
            raise ValueError(
                "max_sentences must be positive"
            )

        if not isinstance(
            importance_threshold,
            (int, float),
        ):
            raise TypeError(
                "importance_threshold must be numeric"
            )

        if not 0.0 <= importance_threshold <= 1.0:
            raise ValueError(
                "importance_threshold must be between 0 and 1"
            )

        if not isinstance(
            require_preservation,
            bool,
        ):
            raise TypeError(
                "require_preservation must be boolean"
            )

        self.redundancy_compressor = (
            redundancy_compressor
        )

        self.token_reduction_strategy = (
            token_reduction_strategy
        )

        self.entity_preserver = EntityPreserver(
            entity_extractor=entity_extractor
        )

        self.fact_preserver = FactPreserver()

        self.information_preserver = (
            InformationPreserver(
                importance_threshold=importance_threshold
            )
        )

        self.prioritizer = ContextPrioritizer()

        self.sentence_selector = SentenceSelector(
            max_sentences=max_sentences
        )

        self.require_preservation = (
            require_preservation
        )

    @staticmethod
    def _validate_context(
        contexts: Sequence[RankedContext],
    ) -> None:

        if not isinstance(
            contexts,
            (list, tuple),
        ):
            raise TypeError(
                "contexts must be a list or tuple"
            )

        for context in contexts:
            if not isinstance(
                context,
                RankedContext,
            ):
                raise TypeError(
                    "contexts must contain "
                    "RankedContext objects"
                )

    def compress(
        self,
        contexts: Sequence[RankedContext],
        token_budget: int,
    ) -> FinalContextCompressionResult:
        """
        Execute the complete Phase 4 compression pipeline.
        """

        self._validate_context(contexts)

        if not isinstance(token_budget, int):
            raise TypeError(
                "token_budget must be an integer"
            )

        if token_budget < 0:
            raise ValueError(
                "token_budget must be non-negative"
            )

        original = list(contexts)

        if not original:
            empty_info = (
                self.information_preserver.validate(
                    [],
                    [],
                )
            )

            empty_entities = (
                self.entity_preserver.validate(
                    [],
                    [],
                )
            )

            empty_facts = (
                self.fact_preserver.validate(
                    [],
                    [],
                )
            )

            return FinalContextCompressionResult(
                selected=[],
                text="",
                original_count=0,
                final_count=0,
                original_tokens=0,
                compressed_tokens=0,
                token_reduction=0,
                reduction_percentage=0.0,
                removed_by_redundancy=[],
                information_preservation=empty_info,
                entity_preservation=empty_entities,
                fact_preservation=empty_facts,
                is_preserved=True,
            )

        # ---------------------------------------------------------
        # 1. Context prioritization
        # ---------------------------------------------------------

        prioritized = self.prioritizer.prioritize(
            original
        )

        prioritized_context = [
            item.context
            for item in prioritized
        ]

        # ---------------------------------------------------------
        # 2. Existing redundancy-aware compression
        # ---------------------------------------------------------

        redundancy_result = (
            self.redundancy_compressor.compress(
                prioritized_context
            )
        )

        redundancy_selected = (
            redundancy_result.selected
        )

        # ---------------------------------------------------------
        # 3. Existing sentence selection
        # ---------------------------------------------------------

        sentence_selection = (
            self.sentence_selector.select(
                redundancy_selected
            )
        )

        selected_candidates = (
            sentence_selection.selected
        )

        # ---------------------------------------------------------
        # 4. Re-prioritize before token budgeting.
        #
        # TokenReductionStrategy consumes the supplied order.
        # Therefore we explicitly provide priority order here.
        # ---------------------------------------------------------

        reprioritized = (
            self.prioritizer.prioritize(
                selected_candidates
            )
        )

        priority_order = [
            item.context
            for item in reprioritized
        ]

        # ---------------------------------------------------------
        # 5. Existing token-budget reduction
        # ---------------------------------------------------------

        token_result: TokenReductionResult = (
            self.token_reduction_strategy.select(
                priority_order,
                token_budget,
            )
        )

        final_selected = list(
            token_result.selected
        )

        # Final readable context follows source order.
        final_selected.sort(
            key=lambda item: item.result.index
        )

        final_text = " ".join(
            item.result.text
            for item in final_selected
        )

        # ---------------------------------------------------------
        # 6. Existing preservation validators
        # ---------------------------------------------------------

        information_result = (
            self.information_preserver.validate(
                redundancy_selected,
                final_selected,
            )
        )

        entity_result = (
            self.entity_preserver.validate(
                redundancy_selected,
                final_selected,
            )
        )

        fact_result = (
            self.fact_preserver.validate(
                redundancy_selected,
                final_selected,
            )
        )

        is_preserved = (
            information_result.is_preserved
            and entity_result.is_preserved
            and fact_result.is_preserved
        )

        # ---------------------------------------------------------
        # 7. Final gate
        # ---------------------------------------------------------

        if (
            self.require_preservation
            and not is_preserved
        ):
            missing_parts = []

            if not information_result.is_preserved:
                missing_parts.append(
                    "important information"
                )

            if not entity_result.is_preserved:
                missing_parts.append(
                    "entities"
                )

            if not fact_result.is_preserved:
                missing_parts.append(
                    "facts"
                )

            raise ValueError(
                "final context failed preservation: "
                + ", ".join(missing_parts)
            )

        return FinalContextCompressionResult(
            selected=final_selected,
            text=final_text,
            original_count=len(original),
            final_count=len(final_selected),
            original_tokens=(
                token_result.original_tokens
            ),
            compressed_tokens=(
                token_result.compressed_tokens
            ),
            token_reduction=(
                token_result.token_reduction
            ),
            reduction_percentage=(
                token_result.reduction_percentage
            ),
            removed_by_redundancy=(
                redundancy_result.removed
            ),
            information_preservation=(
                information_result
            ),
            entity_preservation=(
                entity_result
            ),
            fact_preservation=(
                fact_result
            ),
            is_preserved=is_preserved,
        )

