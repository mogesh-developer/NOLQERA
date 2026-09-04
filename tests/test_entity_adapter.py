import pytest

from nolqera.intelligence.entities.adapters import EntityAdapter
from nolqera.intelligence.entities.models import EntityResult


def test_to_result_converts_external_entity():
    entity = {
        "text": "Python",
        "entity_type": "MISC",
        "score": 0.99,
        "start": 10,
        "end": 16,
    }

    result = EntityAdapter.to_result(entity)

    assert isinstance(result, EntityResult)
    assert result.text == "Python"
    assert result.entity_type == "MISC"
    assert result.score == 0.99
    assert result.start == 10
    assert result.end == 16


def test_to_result_converts_numeric_values():
    entity = {
        "text": "FastAPI",
        "entity_type": "MISC",
        "score": "0.95",
        "start": "20",
        "end": "27",
    }

    result = EntityAdapter.to_result(entity)

    assert isinstance(result, EntityResult)
    assert result.score == 0.95
    assert result.start == 20
    assert result.end == 27


def test_to_result_preserves_model_entity_type():
    entity = {
        "text": "MongoDB",
        "entity_type": "MISC",
        "score": 0.91,
        "start": 5,
        "end": 12,
    }

    result = EntityAdapter.to_result(entity)

    assert result.entity_type == "MISC"


def test_to_result_rejects_non_dictionary():
    with pytest.raises(TypeError, match="entity must be a dictionary"):
        EntityAdapter.to_result("Python")


def test_to_result_rejects_missing_fields():
    entity = {
        "text": "Python",
        "entity_type": "MISC",
        "score": 0.99,
    }

    with pytest.raises(
        ValueError,
        match="entity is missing required fields",
    ):
        EntityAdapter.to_result(entity)


def test_to_results_converts_multiple_entities():
    entities = [
        {
            "text": "FastAPI",
            "entity_type": "MISC",
            "score": 0.98,
            "start": 0,
            "end": 7,
        },
        {
            "text": "Python",
            "entity_type": "MISC",
            "score": 0.99,
            "start": 12,
            "end": 18,
        },
    ]

    results = EntityAdapter.to_results(entities)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(result, EntityResult) for result in results)

    assert results[0].text == "FastAPI"
    assert results[1].text == "Python"


def test_to_results_returns_empty_list_for_empty_input():
    results = EntityAdapter.to_results([])

    assert results == []


def test_to_results_rejects_non_list():
    with pytest.raises(TypeError, match="entities must be a list"):
        EntityAdapter.to_results("Python")