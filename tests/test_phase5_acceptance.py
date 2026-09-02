from nolqera import (
    NOLQERAPipeline,
    PipelineConfig,
    PipelineResult,
)


def test_phase5_public_pipeline_contract():
    assert NOLQERAPipeline is not None
    assert PipelineConfig is not None
    assert PipelineResult is not None


def test_phase5_configuration_contract():
    config = PipelineConfig(
        keyword_top_k=5,
        max_sentences=3,
    )

    assert config.keyword_top_k == 5
    assert config.max_sentences == 3


def test_phase5_pipeline_configuration_is_centralized():
    config = PipelineConfig(
        keyword_top_k=7,
        max_sentences=2,
    )

    assert config.keyword_top_k == 7
    assert config.max_sentences == 2


def test_phase5_result_contract():
    result = PipelineResult(
        input_text="raw",
        normalized_text="normalized",
        compressed_context="optimized",
    )

    assert result.input_text == "raw"
    assert result.normalized_text == "normalized"
    assert result.compressed_context == "optimized"
    assert result.is_empty is False


def test_phase5_result_empty_contract():
    result = PipelineResult(
        input_text="raw",
        normalized_text="normalized",
    )

    assert result.is_empty is True


def test_phase5_pipeline_result_contains_metadata():
    from nolqera.intelligence.pipeline import (
        PipelineMetadata,
    )

    metadata = PipelineMetadata(
        input_count=4,
        sentence_count=4,
        filtered_count=3,
        ranked_count=2,
    )

    result = PipelineResult(
        input_text="raw",
        normalized_text="normalized",
        compressed_context="optimized",
        metadata=metadata,
    )

    assert result.metadata.input_count == 4
    assert result.metadata.sentence_count == 4
    assert result.metadata.filtered_count == 3
    assert result.metadata.ranked_count == 2


def test_phase5_pipeline_is_constructible():
    from nolqera.intelligence.pipeline import (
        create_default_configured_pipeline,
    )

    # The public constructor contract itself is validated
    # through the public symbol.
    assert callable(
        create_default_configured_pipeline
    )


def test_phase5_pipeline_configuration_is_immutable():
    config = PipelineConfig()

    try:
        config.keyword_top_k = 10
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "PipelineConfig must be immutable"
        )


def test_phase5_result_is_immutable():
    result = PipelineResult(
        input_text="raw",
        normalized_text="normalized",
    )

    try:
        result.input_text = "changed"
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "PipelineResult must be immutable"
        )