from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """
    Central configuration for the NOLQERA core pipeline.

    The configuration controls pipeline-level behavior.
    Individual intelligence algorithms remain responsible
    for their own internal implementation.
    """

    keyword_top_k: int = 5
    max_sentences: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.keyword_top_k, int):
            raise TypeError(
                "keyword_top_k must be an integer"
            )

        if self.keyword_top_k <= 0:
            raise ValueError(
                "keyword_top_k must be greater than zero"
            )

        if not isinstance(self.max_sentences, int):
            raise TypeError(
                "max_sentences must be an integer"
            )

        if self.max_sentences <= 0:
            raise ValueError(
                "max_sentences must be greater than zero"
            )