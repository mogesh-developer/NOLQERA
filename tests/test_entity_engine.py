from nolqera.intelligence.entities.engine import (
    EntityEngine,
)
from nolqera.intelligence.entities.models import (
    EntityResult,
)


def test_entity_engine_runs_full_pipeline():

    text = (
        "Dr John travelled to Chennai "
        "and studied at American College."
    )

    engine = EntityEngine()

    results = engine.analyze(text)

    print("\n--- NOLQERA Entity Engine ---")

    for rank, entity in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{entity.text:<25} "
            f"{entity.entity_type:<15} "
            f"{entity.score:.4f}"
        )

    assert results

    assert all(
        isinstance(entity, EntityResult)
        for entity in results
    )

    # Results must already be ranked.
    for first, second in zip(
        results,
        results[1:],
    ):
        assert first.score >= second.score

    # Final results must not overlap.
    for index, first in enumerate(results):
        for second in results[index + 1:]:
            assert not (
                first.start < second.end
                and second.start < first.end
            )