from __future__ import annotations

import json
from dataclasses import asdict, dataclass


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
    compression_strategy: str = "standard"

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

        if not isinstance(self.compression_strategy, str):
            raise TypeError(
                "compression_strategy must be a string"
            )

        if self.compression_strategy not in ("standard", "adaptive"):
            raise ValueError(
                "compression_strategy must be either 'standard' or 'adaptive'"
            )

    def to_dict(self) -> dict:
        """Serialize configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        """Create configuration from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        return cls(
            keyword_top_k=data.get("keyword_top_k", 5),
            max_sentences=data.get("max_sentences", 3),
            compression_strategy=data.get("compression_strategy", "standard"),
        )

    def to_json(self, indent: int | None = None) -> str:
        """Serialize configuration to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> PipelineConfig:
        """Create configuration from a JSON string."""
        if not isinstance(json_str, str):
            raise TypeError("json_str must be a string")
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON string: {exc}") from exc
        return cls.from_dict(data)