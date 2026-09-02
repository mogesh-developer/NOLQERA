import pytest

from nolqera.intelligence.pipeline.exceptions import (
    NOLQERAPipelineError,
    PipelineConfigurationError,
    PipelineExecutionError,
    PipelineStageError,
)


def test_base_pipeline_error():
    error = NOLQERAPipelineError(
        "pipeline error"
    )

    assert str(error) == "pipeline error"


def test_configuration_error_inherits_pipeline_error():
    error = PipelineConfigurationError(
        "invalid configuration"
    )

    assert isinstance(
        error,
        NOLQERAPipelineError,
    )

    assert str(error) == (
        "invalid configuration"
    )


def test_execution_error_inherits_pipeline_error():
    error = PipelineExecutionError(
        "execution failed"
    )

    assert isinstance(
        error,
        NOLQERAPipelineError,
    )

    assert str(error) == (
        "execution failed"
    )


def test_stage_error_inherits_execution_error():
    error = PipelineStageError(
        stage="relevance",
        message="analysis failed",
    )

    assert isinstance(
        error,
        PipelineExecutionError,
    )

    assert isinstance(
        error,
        NOLQERAPipelineError,
    )


def test_stage_error_contains_stage():
    error = PipelineStageError(
        stage="compression",
        message="token limit exceeded",
    )

    assert error.stage == "compression"


def test_stage_error_contains_message():
    error = PipelineStageError(
        stage="compression",
        message="token limit exceeded",
    )

    assert error.message == (
        "token limit exceeded"
    )


def test_stage_error_formats_message():
    error = PipelineStageError(
        stage="compression",
        message="token limit exceeded",
    )

    assert str(error) == (
        "Pipeline stage 'compression' failed: "
        "token limit exceeded"
    )


def test_stage_error_can_be_caught_as_pipeline_error():
    with pytest.raises(NOLQERAPipelineError):
        raise PipelineStageError(
            stage="ranking",
            message="ranking failed",
        )


def test_stage_error_can_be_caught_as_execution_error():
    with pytest.raises(PipelineExecutionError):
        raise PipelineStageError(
            stage="ranking",
            message="ranking failed",
        )


def test_exception_hierarchy():
    assert issubclass(
        PipelineConfigurationError,
        NOLQERAPipelineError,
    )

    assert issubclass(
        PipelineExecutionError,
        NOLQERAPipelineError,
    )

    assert issubclass(
        PipelineStageError,
        PipelineExecutionError,
    )