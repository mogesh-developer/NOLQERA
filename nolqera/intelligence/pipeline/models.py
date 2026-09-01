from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineMetadata:
    """
    Metadata describing one NOLQERA pipeline execution.
    """

    input_count: int = 0
    sentence_count: int = 0
    filtered_count: int = 0
    ranked_count: int = 0

    def __post_init__(self) -> None:
        values = {
            "input_count": self.input_count,
            "sentence_count": self.sentence_count,
            "filtered_count": self.filtered_count,
            "ranked_count": self.ranked_count,
        }

        for name, value in values.items():
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer"
                )

            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )


@dataclass(frozen=True)
class PipelineResult:
    """
    Unified data contract for the NOLQERA pipeline.

    This model contains outputs from the different
    intelligence stages without implementing any
    intelligence itself.
    """

    input_text: str
    normalized_text: str

    sentences: list[str] = field(
        default_factory=list
    )

    relevance: list[dict[str, Any]] = field(
        default_factory=list
    )

    importance: list[dict[str, Any]] = field(
        default_factory=list
    )

    keywords: Any = None
    entities: Any = None
    intents: Any = None

    filtered_results: list[Any] = field(
        default_factory=list
    )

    ranked_context: list[Any] = field(
        default_factory=list
    )

    compressed_context: str = ""

    metadata: PipelineMetadata = field(
        default_factory=PipelineMetadata
    )

    def __post_init__(self) -> None:
        if not isinstance(self.input_text, str):
            raise TypeError(
                "input_text must be a string"
            )

        if not isinstance(self.normalized_text, str):
            raise TypeError(
                "normalized_text must be a string"
            )

        if not isinstance(self.sentences, list):
            raise TypeError(
                "sentences must be a list"
            )

        if not isinstance(self.relevance, list):
            raise TypeError(
                "relevance must be a list"
            )

        if not isinstance(self.importance, list):
            raise TypeError(
                "importance must be a list"
            )

        if not isinstance(self.filtered_results, list):
            raise TypeError(
                "filtered_results must be a list"
            )

        if not isinstance(self.ranked_context, list):
            raise TypeError(
                "ranked_context must be a list"
            )

        if not isinstance(
            self.compressed_context,
            str,
        ):
            raise TypeError(
                "compressed_context must be a string"
            )

        if not isinstance(
            self.metadata,
            PipelineMetadata,
        ):
            raise TypeError(
                "metadata must be a PipelineMetadata"
            )

    @property
    def is_empty(self) -> bool:
        """
        Return True when no compressed context exists.
        """
        return not bool(
            self.compressed_context.strip()
        )