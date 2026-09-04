import pytest

from nolqera.intelligence.entities.providers import (
    HuggingFaceEntityRecognizer,
)


@pytest.fixture(scope="module")
def recognizer():
    return HuggingFaceEntityRecognizer()


def test_recognize_returns_entities(recognizer):
    text = (
        "The application uses FastAPI and MongoDB. "
        "Python is used for development."
    )

    entities = recognizer.recognize(text)

    assert isinstance(entities, list)
    assert entities


def test_technical_entities_are_detected(recognizer):
    text = (
        "The application uses FastAPI and MongoDB. "
        "Python is used for development."
    )

    entities = recognizer.recognize(text)
    values = [entity["text"] for entity in entities]

    assert "FastAPI" in values
    assert "MongoDB" in values
    assert "Python" in values


def test_model_entity_type_is_preserved(recognizer):
    text = (
        "The application uses FastAPI and MongoDB. "
        "Python is used for development."
    )

    entities = recognizer.recognize(text)

    entity_map = {
        entity["text"]: entity["entity_type"]
        for entity in entities
    }

    assert entity_map["FastAPI"] == "MISC"
    assert entity_map["MongoDB"] == "MISC"
    assert entity_map["Python"] == "MISC"


def test_entity_scores_are_valid(recognizer):
    text = "FastAPI and Python are used."

    entities = recognizer.recognize(text)

    assert entities

    for entity in entities:
        assert 0.0 <= entity["score"] <= 1.0


def test_entity_offsets_are_valid(recognizer):
    text = "FastAPI and Python are used."

    entities = recognizer.recognize(text)

    assert entities

    for entity in entities:
        assert 0 <= entity["start"] < entity["end"] <= len(text)
        assert text[entity["start"]:entity["end"]] == entity["text"]


def test_empty_text_raises_value_error(recognizer):
    with pytest.raises(ValueError, match="text cannot be empty"):
        recognizer.recognize("")


def test_whitespace_text_raises_value_error(recognizer):
    with pytest.raises(ValueError, match="text cannot be empty"):
        recognizer.recognize("   ")


def test_non_string_text_raises_type_error(recognizer):
    with pytest.raises(TypeError, match="text must be a string"):
        recognizer.recognize(None)


def test_invalid_model_name_raises_error():
    with pytest.raises(Exception):
        HuggingFaceEntityRecognizer(
            model_name="this-model-does-not-exist-123456"
        )