import pytest

from nolqera.intelligence.entities.models import (
    EntityResult,
)


def test_entity_result_stores_analysis():

    result = EntityResult(
        text="Chennai",
        entity_type="LOCATION",
        score=0.85,
        start=10,
        end=17,
    )

    assert result.text == "Chennai"
    assert result.entity_type == "LOCATION"
    assert result.score == 0.85
    assert result.start == 10
    assert result.end == 17


def test_entity_result_calculates_length():

    result = EntityResult(
        text="Chennai",
        entity_type="LOCATION",
        score=0.85,
        start=10,
        end=17,
    )

    assert result.length == 7


def test_entity_result_supports_metadata():

    result = EntityResult(
        text="FastAPI",
        entity_type="TECHNOLOGY",
        score=0.91,
        start=0,
        end=7,
        metadata={
            "source": "detector",
        },
    )

    assert result.metadata["source"] == "detector"


def test_entity_result_to_dict():

    result = EntityResult(
        text="MongoDB",
        entity_type="TECHNOLOGY",
        score=0.88,
        start=5,
        end=12,
    )

    data = result.to_dict()

    assert data["text"] == "MongoDB"
    assert data["entity_type"] == "TECHNOLOGY"
    assert data["score"] == 0.88
    assert data["start"] == 5
    assert data["end"] == 12


@pytest.mark.parametrize(
    "score",
    [-0.1, 1.1],
)
def test_invalid_score_is_rejected(score):

    with pytest.raises(ValueError):
        EntityResult(
            text="Test",
            entity_type="UNKNOWN",
            score=score,
            start=0,
            end=4,
        )


def test_empty_text_is_rejected():

    with pytest.raises(ValueError):
        EntityResult(
            text="",
            entity_type="PERSON",
            score=0.8,
            start=0,
            end=1,
        )


def test_invalid_span_is_rejected():

    with pytest.raises(ValueError):
        EntityResult(
            text="John",
            entity_type="PERSON",
            score=0.8,
            start=10,
            end=5,
        )