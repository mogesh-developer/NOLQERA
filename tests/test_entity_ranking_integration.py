from nolqera.intelligence.entities.candidates import (
    EntityCandidateExtractor,
)
from nolqera.intelligence.entities.detector import (
    EntityDetector,
)
from nolqera.intelligence.entities.cleanup import (
    EntitySpanCleaner,
)
from nolqera.intelligence.entities.ranking import (
    EntityRanker,
)


def test_real_entities_are_cleaned_and_ranked():

    text = (
        "Dr John travelled to Chennai "
        "and studied at American College."
    )

    # Candidate extraction
    extractor = EntityCandidateExtractor()
    candidates = extractor.extract(text)

    # Entity detection
    detector = EntityDetector()
    detected = detector.detect(
        candidates,
        text,
    )

    # Remove overlapping spans
    cleaner = EntitySpanCleaner()
    cleaned = cleaner.clean(detected)

    # Rank final entities
    ranker = EntityRanker()
    ranked = ranker.rank(cleaned)

    print("\n--- Cleaned Entities ---")

    for entity in cleaned:
        print(
            f"{entity.text:<25} "
            f"{entity.entity_type:<15} "
            f"{entity.score:.4f}"
        )

    print("\n--- Ranked Entities ---")

    for index, entity in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"{index}. "
            f"{entity.text:<25} "
            f"{entity.entity_type:<15} "
            f"{entity.score:.4f}"
        )

    # Ranking must preserve all cleaned entities.
    assert len(ranked) == len(cleaned)

    # Highest score must come first.
    for first, second in zip(
        ranked,
        ranked[1:],
    ):
        assert first.score >= second.score

    # Ranking must not create overlaps.
    for index, first in enumerate(ranked):
        for second in ranked[index + 1:]:
            assert not (
                first.start < second.end
                and second.start < first.end
            )