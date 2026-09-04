from nolqera.intelligence.entities.adapters import EntityAdapter
from nolqera.intelligence.entities.models import EntityResult
from nolqera.intelligence.entities.providers import (
    HuggingFaceEntityRecognizer,
)


def test_huggingface_output_can_be_adapted():
    recognizer = HuggingFaceEntityRecognizer()

    text = (
        "The application uses FastAPI and MongoDB. "
        "Python is used for development."
    )

    raw_entities = recognizer.recognize(text)
    results = EntityAdapter.to_results(raw_entities)

    assert results
    assert all(isinstance(result, EntityResult) for result in results)

    values = [result.text for result in results]

    assert "FastAPI" in values
    assert "MongoDB" in values
    assert "Python" in values


def test_huggingface_offsets_survive_adaptation():
    recognizer = HuggingFaceEntityRecognizer()

    text = "FastAPI and Python are used."

    raw_entities = recognizer.recognize(text)
    results = EntityAdapter.to_results(raw_entities)

    for result in results:
        assert text[result.start:result.end] == result.text


def test_huggingface_entity_type_survives_adaptation():
    recognizer = HuggingFaceEntityRecognizer()

    text = "FastAPI and Python are used."

    raw_entities = recognizer.recognize(text)
    results = EntityAdapter.to_results(raw_entities)

    result_map = {
        result.text: result.entity_type
        for result in results
    }

    assert result_map["FastAPI"] == "MISC"
    assert result_map["Python"] == "MISC"