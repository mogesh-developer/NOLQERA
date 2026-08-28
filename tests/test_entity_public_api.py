from nolqera.intelligence.entities import (
    EntityEngine,
    EntityResult,
)


def test_entity_public_api_exports():

    assert EntityEngine is not None
    assert EntityResult is not None


def test_entity_public_api_runs_engine():

    engine = EntityEngine()

    results = engine.analyze(
        "John travelled to Chennai."
    )

    assert isinstance(results, list)

    for result in results:
        assert isinstance(result, EntityResult)