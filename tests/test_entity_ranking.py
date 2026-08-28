from nolqera.intelligence.entities.candidates import (
    EntityCandidateExtractor,
)
from nolqera.intelligence.entities.detector import (
    EntityDetector,
)
from nolqera.intelligence.entities.ranking import (
    EntityRanker,
)


def test_rank_real_detected_entities():
    text = (
        "Dr John travelled to Chennai "
        "and studied at American College."
    )

    extractor = EntityCandidateExtractor()
    candidates = extractor.extract(text)

    detector = EntityDetector()
    entities = detector.detect(
        candidates,
        text,
    )

    ranker = EntityRanker()
    ranked = ranker.rank(entities)

    print("\n--- Real Entity Ranking ---")

    for rank, entity in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{entity.text} | "
            f"{entity.entity_type} | "
            f"{entity.score:.4f}"
        )

    assert ranked