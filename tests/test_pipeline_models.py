import pytest

from nolqera.intelligence.pipeline.models import (
    PipelineMetadata,
    PipelineResult,
)


def test_default_metadata():
    metadata = PipelineMetadata()

    assert metadata.input_count == 0
    assert metadata.sentence_count == 0
    assert metadata.filtered_count == 0
    assert metadata.ranked_count == 0


def test_custom_metadata():
    metadata = PipelineMetadata(
        input_count=10,
        sentence_count=8,
        filtered_count=5,
        ranked_count=3,
    )

    assert metadata.input_count == 10
    assert metadata.sentence_count == 8
    assert metadata.filtered_count == 5
    assert metadata.ranked_count == 3


def test_metadata_rejects_negative_values():
    with pytest.raises(ValueError):
        PipelineMetadata(
            input_count=-1
        )


def test_metadata_rejects_invalid_types():
    with pytest.raises(TypeError):
        PipelineMetadata(
            input_count="10"
        )


def test_pipeline_result_defaults():
    result = PipelineResult(
        input_text="raw input",
        normalized_text="raw input",
    )

    assert result.sentences == []
    assert result.relevance == []
    assert result.importance == []
    assert result.keywords is None
    assert result.entities is None
    assert result.intents is None
    assert result.filtered_results == []
    assert result.ranked_context == []
    assert result.compressed_context == ""


def test_pipeline_result_stores_all_stage_outputs():
    result = PipelineResult(
        input_text="Raw input",
        normalized_text="Normalized input",
        sentences=["Sentence one."],
        relevance=[
            {
                "text": "Sentence one.",
                "score": 0.9,
            }
        ],
        importance=[
            {
                "text": "Sentence one.",
                "score": 0.8,
            }
        ],
        keywords=["python", "fastapi"],
        entities=["Python", "FastAPI"],
        intents=["information"],
        filtered_results=["filtered"],
        ranked_context=["ranked"],
        compressed_context="Sentence one.",
        metadata=PipelineMetadata(
            input_count=1,
            sentence_count=1,
            filtered_count=1,
            ranked_count=1,
        ),
    )

    assert result.input_text == "Raw input"
    assert result.normalized_text == "Normalized input"
    assert result.sentences == ["Sentence one."]
    assert result.relevance[0]["score"] == 0.9
    assert result.importance[0]["score"] == 0.8
    assert result.keywords == ["python", "fastapi"]
    assert result.entities == ["Python", "FastAPI"]
    assert result.intents == ["information"]
    assert result.filtered_results == ["filtered"]
    assert result.ranked_context == ["ranked"]
    assert result.compressed_context == "Sentence one."


def test_pipeline_result_is_empty():
    result = PipelineResult(
        input_text="input",
        normalized_text="input",
    )

    assert result.is_empty is True


def test_pipeline_result_is_not_empty():
    result = PipelineResult(
        input_text="input",
        normalized_text="input",
        compressed_context="Useful context.",
    )

    assert result.is_empty is False


def test_pipeline_result_whitespace_context_is_empty():
    result = PipelineResult(
        input_text="input",
        normalized_text="input",
        compressed_context="   ",
    )

    assert result.is_empty is True


def test_pipeline_result_rejects_invalid_input_text():
    with pytest.raises(TypeError):
        PipelineResult(
            input_text=123,
            normalized_text="input",
        )


def test_pipeline_result_rejects_invalid_normalized_text():
    with pytest.raises(TypeError):
        PipelineResult(
            input_text="input",
            normalized_text=123,
        )


def test_pipeline_result_rejects_invalid_sentences():
    with pytest.raises(TypeError):
        PipelineResult(
            input_text="input",
            normalized_text="input",
            sentences="not a list",
        )


def test_pipeline_result_rejects_invalid_compressed_context():
    with pytest.raises(TypeError):
        PipelineResult(
            input_text="input",
            normalized_text="input",
            compressed_context=123,
        )


def test_pipeline_result_rejects_invalid_metadata():
    with pytest.raises(TypeError):
        PipelineResult(
            input_text="input",
            normalized_text="input",
            metadata="invalid",
        )


def test_metadata_is_immutable():
    metadata = PipelineMetadata(
        input_count=5
    )

    with pytest.raises(AttributeError):
        metadata.input_count = 10


def test_pipeline_result_is_immutable():
    result = PipelineResult(
        input_text="input",
        normalized_text="input",
    )

    with pytest.raises(AttributeError):
        result.input_text = "changed"