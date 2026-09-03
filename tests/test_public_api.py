import pytest

from nolqera.intelligence.pipeline import (
    NOLQERAEngine,
    PipelineResult,
    create_engine,
)


def test_create_engine_returns_public_engine(
    mock_components,
):
    engine = create_engine(**mock_components)

    assert isinstance(engine, NOLQERAEngine)


def test_engine_process_returns_pipeline_result(
    mock_components,
):
    engine = create_engine(**mock_components)

    result = engine.process(
        query="Python API",
        raw_input="Python API development with FastAPI.",
    )

    assert isinstance(result, PipelineResult)


def test_public_engine_process_requires_query(
    mock_components,
):
    engine = create_engine(**mock_components)

    with pytest.raises(TypeError):
        engine.process(
            query=None,
            raw_input="Python API development.",
        )


def test_public_engine_process_requires_raw_input(
    mock_components,
):
    engine = create_engine(**mock_components)

    with pytest.raises(TypeError):
        engine.process(
            query="Python API",
            raw_input=None,
        )   