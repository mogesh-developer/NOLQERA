from nolqera.intelligence.entities.candidates import (
    EntityCandidateExtractor,
)
from nolqera.intelligence.entities.detector import (
    EntityDetector,
)


def test_detector_uses_person_context():
    text = "Dr John joined the team."

    extractor = EntityCandidateExtractor()
    candidates = extractor.extract(text)

    detector = EntityDetector()
    entities = detector.detect(
        candidates,
        text,
    )

    john = next(
        entity
        for entity in entities
        if entity.text == "John"
    )

    assert john.entity_type == "PERSON"
    assert john.score > 0.0


def test_detector_uses_location_context():
    text = "I travelled to Chennai yesterday."

    extractor = EntityCandidateExtractor()
    candidates = extractor.extract(text)

    detector = EntityDetector()
    entities = detector.detect(
        candidates,
        text,
    )

    chennai = next(
        entity
        for entity in entities
        if entity.text == "Chennai"
    )

    assert chennai.entity_type == "LOCATION"


def test_detector_detects_organization_style_phrase():
    text = "I studied at American College."

    extractor = EntityCandidateExtractor()
    candidates = extractor.extract(text)

    detector = EntityDetector()
    entities = detector.detect(
        candidates,
        text,
    )

    college = next(
        entity
        for entity in entities
        if entity.text == "American College"
    )

    assert college.entity_type == "ORGANIZATION"


def test_detector_rejects_invalid_candidates():
    detector = EntityDetector()

    try:
        detector.detect(
            ["invalid"],
            "Some text.",
        )
    except TypeError:
        return

    assert False