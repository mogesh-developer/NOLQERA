from __future__ import annotations

from nolqera.intelligence.entities import (
    EntityEngine,
    HuggingFaceEntityRecognizer,
)

from .config import PipelineConfig
from .models import PipelineResult
from .orchestrator import NOLQERAPipeline

class NOLQERAEngine:
    def __init__(self,pipeline: NOLQERAPipeline)->None:
        if not isinstance(pipeline, NOLQERAPipeline):
            raise TypeError("pipeline must be NOLQERAPipeline")
        self.pipeline = pipeline
    def process(self,query: str,raw_input: str)->PipelineResult:
        return run_pipeline(pipeline=self.pipeline, query=query,raw_input=raw_input)

def run_pipeline(
    pipeline: NOLQERAPipeline,
    query: str,
    raw_input: str,
) -> PipelineResult:
    """
    Execute a configured NOLQERA pipeline.

    This is a thin integration-level entry point.
    All intelligence remains inside the existing
    pipeline components.
    """

    if not isinstance(
        pipeline,
        NOLQERAPipeline,
    ):
        raise TypeError(
            "pipeline must be a NOLQERAPipeline"
        )

    if not isinstance(query, str):
        raise TypeError(
            "query must be a string"
        )

    if not isinstance(raw_input, str):
        raise TypeError(
            "raw_input must be a string"
        )

    return pipeline.process(
        query=query,
        raw_input=raw_input,
    )


def create_default_configured_pipeline(
    semantic_search_engine,
    importance_engine,
    keyphrase_engine,
    entity_engine,
    intent_engine,
    noise_remover,
    context_ranker,
    context_compressor,
    config: PipelineConfig | None = None,
) -> NOLQERAPipeline:
    """
    Construct a NOLQERA pipeline using one centralized
    PipelineConfig.
    """

    return NOLQERAPipeline(
        semantic_search_engine=semantic_search_engine,
        importance_engine=importance_engine,
        keyphrase_engine=keyphrase_engine,
        entity_engine=entity_engine,
        intent_engine=intent_engine,
        noise_remover=noise_remover,
        context_ranker=context_ranker,
        context_compressor=context_compressor,
        config=config,
    )

def create_engine(
    semantic_search_engine,
    importance_engine,
    keyphrase_engine,
    entity_engine,
    intent_engine,
    noise_remover,
    context_ranker,
    context_compressor,
    config: PipelineConfig | None = None,
    use_external_recognizer: bool = False,
    recognizer: HuggingFaceEntityRecognizer | None = None,
) -> NOLQERAEngine:
    """Construct the public NOLQERA integration engine."""

    if not isinstance(use_external_recognizer, bool):
        raise TypeError(
            "use_external_recognizer must be a boolean"
        )

    if use_external_recognizer:
        if recognizer is None:
            recognizer = HuggingFaceEntityRecognizer()

        if not isinstance(
            recognizer,
            HuggingFaceEntityRecognizer,
        ):
            raise TypeError(
                "recognizer must be a HuggingFaceEntityRecognizer"
            )

        entity_engine = EntityEngine(
            recognizer=recognizer,
            use_external_recognizer=True,
        )

    pipeline = create_default_configured_pipeline(
        semantic_search_engine=semantic_search_engine,
        importance_engine=importance_engine,
        keyphrase_engine=keyphrase_engine,
        entity_engine=entity_engine,
        intent_engine=intent_engine,
        noise_remover=noise_remover,
        context_ranker=context_ranker,
        context_compressor=context_compressor,
        config=config,
    )

    return NOLQERAEngine(pipeline)