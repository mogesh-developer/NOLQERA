from __future__ import annotations

from typing import Sequence

from nolqera.intelligence.context_optimization.context_ranking import (
    RankedContext,
)
from nolqera.intelligence.context_optimization.final_context_compressor import (
    FinalContextCompressionResult,
    FinalContextCompressor,
)


class AdaptiveContextCompressor:
    """
    Adaptive wrapper over FinalContextCompressor.

    Dynamically determines the maximum safe context compression budget
    without requiring hardcoded sentence limits. If preservation validation
    fails (e.g. required entities, facts, or important sentences were omitted),
    it automatically restores the missing required sentences from the candidate
    pool to produce a guaranteed safe compressed context.
    """

    def __init__(
        self,
        compressor: FinalContextCompressor,
    ) -> None:

        if not isinstance(
            compressor,
            FinalContextCompressor,
        ):
            raise TypeError(
                "compressor must be a FinalContextCompressor"
            )

        self.compressor = compressor

    def compress(
        self,
        contexts: Sequence[RankedContext],
        token_budget: int,
        max_sentences: int | str = "auto",
    ) -> FinalContextCompressionResult:
        """
        Execute adaptive context compression.

        If `max_sentences == "auto"`, the budget is dynamically set to the
        total number of available candidate contexts.

        If initial compression leaves out required information, entities,
        or facts, missing required sentences are automatically restored into the
        selected context set and re-validated.
        """

        if not isinstance(contexts, (list, tuple)):
            raise TypeError(
                "contexts must be a list or tuple"
            )

        if not isinstance(token_budget, int):
            raise TypeError(
                "token_budget must be an integer"
            )

        if token_budget < 0:
            raise ValueError(
                "token_budget must be non-negative"
            )

        if not contexts:
            return self.compressor.compress(
                contexts=[],
                token_budget=token_budget,
            )

        if max_sentences == "auto":
            target_max = len(contexts)
        elif isinstance(max_sentences, int):
            if max_sentences <= 0:
                raise ValueError(
                    "max_sentences must be positive or 'auto'"
                )
            target_max = max_sentences
        else:
            raise TypeError(
                "max_sentences must be an integer or 'auto'"
            )

        original_max_sentences = (
            self.compressor.sentence_selector.max_sentences
        )
        original_require_preservation = (
            self.compressor.require_preservation
        )

        try:
            self.compressor.sentence_selector.max_sentences = (
                target_max
            )
            self.compressor.require_preservation = False

            result = self.compressor.compress(
                contexts=contexts,
                token_budget=token_budget,
            )
        finally:
            self.compressor.sentence_selector.max_sentences = (
                original_max_sentences
            )
            self.compressor.require_preservation = (
                original_require_preservation
            )

        if result.is_preserved:
            return result

        # Automatically restore missing required sentences
        selected_set = {
            item.result.index: item
            for item in result.selected
        }

        # Restore missing important information sentences
        if not result.information_preservation.is_preserved:
            for item in result.information_preservation.missing:
                selected_set[item.result.index] = item

        # Restore missing entity sentences
        if not result.entity_preservation.is_preserved:
            missing_entities = set(
                result.entity_preservation.missing_entities
            )
            for item in contexts:
                if item.result.index not in selected_set:
                    extracted = set(
                        self.compressor.entity_preserver.entity_extractor(
                            item.result.text
                        )
                    )
                    if any(
                        entity.casefold()
                        in {
                            e.casefold()
                            for e in missing_entities
                        }
                        for entity in extracted
                    ):
                        selected_set[item.result.index] = item

        # Restore missing fact sentences
        if not result.fact_preservation.is_preserved:
            missing_facts = set(
                result.fact_preservation.missing_facts
            )
            for item in contexts:
                if item.result.index not in selected_set:
                    extracted = set(
                        self.compressor.fact_preserver._extract_facts(
                            item.result.text
                        )
                    )
                    if missing_facts.intersection(extracted):
                        selected_set[item.result.index] = item

        restored_selected = sorted(
            list(selected_set.values()),
            key=lambda item: item.result.index,
        )

        restored_text = " ".join(
            item.result.text
            for item in restored_selected
        )

        info_val = (
            self.compressor.information_preserver.validate(
                contexts,
                restored_selected,
            )
        )

        entity_val = (
            self.compressor.entity_preserver.validate(
                contexts,
                restored_selected,
            )
        )

        fact_val = (
            self.compressor.fact_preserver.validate(
                contexts,
                restored_selected,
            )
        )

        is_preserved = (
            info_val.is_preserved
            and entity_val.is_preserved
            and fact_val.is_preserved
        )

        original_tokens = result.original_tokens
        compressed_tokens = sum(
            self.compressor.token_reduction_strategy._count_tokens(
                item.result.text
            )
            for item in restored_selected
        )

        token_reduction = original_tokens - compressed_tokens
        reduction_percentage = (
            (token_reduction / original_tokens) * 100.0
            if original_tokens > 0
            else 0.0
        )

        return FinalContextCompressionResult(
            selected=restored_selected,
            text=restored_text,
            original_count=result.original_count,
            final_count=len(restored_selected),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            token_reduction=token_reduction,
            reduction_percentage=reduction_percentage,
            removed_by_redundancy=result.removed_by_redundancy,
            information_preservation=info_val,
            entity_preservation=entity_val,
            fact_preservation=fact_val,
            is_preserved=is_preserved,
        )
