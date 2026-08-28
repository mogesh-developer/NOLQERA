from nolqera.intelligence.entities.candidates import (
    EntityCandidate,
)
from nolqera.intelligence.entities.detector import (
    DetectedEntity,
)
from nolqera.intelligence.entities.cleanup import (
    EntitySpanCleaner,
)


def make_entity(
    text: str,
    start: int,
    entity_type: str,
    score: float,
) -> DetectedEntity:

    candidate = EntityCandidate(
        text=text,
        start=start,
        end=start + len(text),
    )

    return DetectedEntity(
        candidate=candidate,
        entity_type=entity_type,
        score=score,
    )


def test_longer_span_wins_when_scores_are_equal():

    entities = [
        make_entity(
            "American College",
            20,
            "ORGANIZATION",
            0.85,
        ),
        make_entity(
            "American",
            20,
            "UNKNOWN",
            0.85,
        ),
        make_entity(
            "College",
            29,
            "ORGANIZATION",
            0.85,
        ),
    ]

    cleaner = EntitySpanCleaner()

    result = cleaner.clean(entities)

    assert len(result) == 1

    assert result[0].text == (
        "American College"
    )


def test_higher_confidence_span_wins():

    entities = [
        make_entity(
            "Dr John",
            0,
            "UNKNOWN",
            0.55,
        ),
        make_entity(
            "John",
            3,
            "PERSON",
            0.85,
        ),
    ]

    cleaner = EntitySpanCleaner()

    result = cleaner.clean(entities)

    assert len(result) == 1

    assert result[0].text == "John"

    assert result[0].entity_type == "PERSON"


def test_nested_span_is_removed():

    entities = [
        make_entity(
            "New York City",
            0,
            "LOCATION",
            0.90,
        ),
        make_entity(
            "New York",
            0,
            "LOCATION",
            0.80,
        ),
        make_entity(
            "York City",
            4,
            "LOCATION",
            0.70,
        ),
    ]

    cleaner = EntitySpanCleaner()

    result = cleaner.clean(entities)

    assert len(result) == 1

    assert result[0].text == (
        "New York City"
    )


def test_non_overlapping_entities_are_preserved():

    entities = [
        make_entity(
            "John",
            0,
            "PERSON",
            0.85,
        ),
        make_entity(
            "Chennai",
            20,
            "LOCATION",
            0.65,
        ),
        make_entity(
            "Google",
            40,
            "ORGANIZATION",
            0.75,
        ),
    ]

    cleaner = EntitySpanCleaner()

    result = cleaner.clean(entities)

    assert [
        entity.text
        for entity in result
    ] == [
        "John",
        "Chennai",
        "Google",
    ]


def test_earlier_span_wins_when_everything_is_equal():

    entities = [
        make_entity(
            "Alpha",
            10,
            "UNKNOWN",
            0.80,
        ),
        make_entity(
            "Beta",
            20,
            "UNKNOWN",
            0.80,
        ),
    ]

    cleaner = EntitySpanCleaner()

    result = cleaner.clean(entities)

    assert [
        entity.text
        for entity in result
    ] == [
        "Alpha",
        "Beta",
    ]


def test_empty_entities_return_empty_list():

    cleaner = EntitySpanCleaner()

    result = cleaner.clean([])

    assert result == []


def test_invalid_entities_are_rejected():

    cleaner = EntitySpanCleaner()

    try:
        cleaner.clean(
            ["invalid"]
        )
    except TypeError:
        return

    assert False

def test_longer_span_wins_when_scores_are_equal():

    entities = [
        make_entity(
            "American College",
            20,
            "ORGANIZATION",
            0.85,
        ),
        make_entity(
            "American",
            20,
            "LOCATION",
            0.65,
        ),
        make_entity(
            "College",
            29,
            "ORGANIZATION",
            0.85,
        ),
    ]

    cleaner = EntitySpanCleaner()

    print("\n--- Input Entities ---")

    for entity in entities:
        print(
            f"{entity.text:<20} "
            f"{entity.score:.4f}"
        )

    result = cleaner.clean(entities)

    print("\n--- Cleaned Entities ---")

    for entity in result:
        print(
            f"{entity.text:<20} "
            f"{entity.score:.4f}"
        )

    assert len(result) == 1

    assert result[0].text == "American College"

    assert result[0].score == 0.85