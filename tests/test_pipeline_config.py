import pytest

from nolqera.intelligence.pipeline.config import (
    PipelineConfig,
)


def test_default_configuration():
    config = PipelineConfig()

    assert config.keyword_top_k == 5
    assert config.max_sentences == 3


def test_custom_configuration():
    config = PipelineConfig(
        keyword_top_k=10,
        max_sentences=7,
    )

    assert config.keyword_top_k == 10
    assert config.max_sentences == 7


def test_configuration_is_immutable():
    config = PipelineConfig()

    with pytest.raises(
        AttributeError
    ):
        config.keyword_top_k = 10


def test_keyword_top_k_must_be_integer():
    with pytest.raises(TypeError):
        PipelineConfig(
            keyword_top_k="5"
        )


def test_keyword_top_k_must_be_positive():
    with pytest.raises(ValueError):
        PipelineConfig(
            keyword_top_k=0
        )


def test_negative_keyword_top_k_is_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(
            keyword_top_k=-1
        )


def test_max_sentences_must_be_integer():
    with pytest.raises(TypeError):
        PipelineConfig(
            max_sentences="3"
        )


def test_max_sentences_must_be_positive():
    with pytest.raises(ValueError):
        PipelineConfig(
            max_sentences=0
        )


def test_negative_max_sentences_is_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(
            max_sentences=-1
        )


def test_configuration_can_be_used_as_value_object():
    config_a = PipelineConfig(
        keyword_top_k=5,
        max_sentences=3,
    )

    config_b = PipelineConfig(
        keyword_top_k=5,
        max_sentences=3,
    )

    assert config_a == config_b