from nolqera.intelligence.entities.candidates import (
    EntityCandidateExtractor,
)
from nolqera.intelligence.entities.detector import (
    EntityDetector,
)
from nolqera.intelligence.entities.cleanup import (
    EntitySpanCleaner,
)


def test_real_detector_output_is_cleaned():

    text = (
        "Dr John travelled to Chennai "
        "and studied at American College."
    )

    # 1. Extract possible entity candidates
    extractor = EntityCandidateExtractor()

    candidates = extractor.extract(text)

    # 2. Detect entity type + confidence
    detector = EntityDetector()

    detected = detector.detect(
        candidates,
        text,
    )

    print("\n--- Detected Entities ---")

    for entity in detected:
        print(
            f"{entity.text:<25} "
            f"{entity.entity_type:<15} "
            f"{entity.score:.4f}"
        )

    # 3. Remove overlapping spans
    cleaner = EntitySpanCleaner()

    cleaned = cleaner.clean(detected)

    print("\n--- Cleaned Entities ---")

    for entity in cleaned:
        print(
            f"{entity.text:<25} "
            f"{entity.entity_type:<15} "
            f"{entity.score:.4f}"
        )

    # Every final entity must have valid boundaries.
    for entity in cleaned:
        assert entity.start < entity.end

    # No two final entities may overlap.
    for index, first in enumerate(cleaned):
        for second in cleaned[index + 1:]:
            assert not (
                first.start < second.end
                and second.start < first.end
            )

    # Cleanup must not create new entities.
    assert len(cleaned) <= len(detected)