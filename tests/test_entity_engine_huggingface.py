import pytest

from nolqera.intelligence.entities.engine import EntityEngine
from nolqera.intelligence.entities.models import EntityResult
from nolqera.intelligence.entities.providers import (
    HuggingFaceEntityRecognizer,
)


@pytest.fixture(scope="module")
def engine():
    recognizer = HuggingFaceEntityRecognizer()

    return EntityEngine(
        recognizer=recognizer,
    )


def test_analyze_with_recognizer_returns_entity_results(engine):
    text = (
        "The application uses FastAPI and MongoDB. "
        "Python is used for development."
    )

    results = engine.analyze_with_recognizer(text)

    assert isinstance(results, list)
    assert results
    assert all(
        isinstance(result, EntityResult)
        for result in results
    )


def test_analyze_with_recognizer_detects_entities(engine):
    text = (
        "The application uses FastAPI and MongoDB. "
        "Python is used for development."
    )

    results = engine.analyze_with_recognizer(text)

    values = [result.text for result in results]

    assert "FastAPI" in values
    assert "MongoDB" in values
    assert "Python" in values


def test_analyze_with_recognizer_preserves_entity_type(engine):
    text = "FastAPI and Python are used."

    results = engine.analyze_with_recognizer(text)

    result_map = {
        result.text: result.entity_type
        for result in results
    }

    assert result_map["FastAPI"] == "MISC"
    assert result_map["Python"] == "MISC"


def test_analyze_with_recognizer_preserves_offsets(engine):
    text = "FastAPI and Python are used."

    results = engine.analyze_with_recognizer(text)

    for result in results:
        assert text[result.start:result.end] == result.text


def test_analyze_with_recognizer_requires_recognizer():
    engine = EntityEngine()

    with pytest.raises(
        RuntimeError,
        match="entity recognizer is not configured",
    ):
        engine.analyze_with_recognizer("FastAPI is useful.")


def test_analyze_with_recognizer_rejects_empty_text(engine):
    with pytest.raises(
        ValueError,
        match="text cannot be empty",
    ):
        engine.analyze_with_recognizer("")


def test_analyze_with_recognizer_rejects_non_string(engine):
    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        engine.analyze_with_recognizer(None)


def test_analyze_routes_to_recognizer_when_use_external_recognizer_is_true(engine):
    external_engine = EntityEngine(
        recognizer=engine.recognizer,
        use_external_recognizer=True,
    )

    text = "The application uses FastAPI and MongoDB."
    results = external_engine.analyze(text)

    assert isinstance(results, list)
    values = [result.text for result in results]
    assert "FastAPI" in values
    assert "MongoDB" in values


def test_analyze_raises_runtime_error_when_use_external_recognizer_true_without_recognizer():
    engine = EntityEngine(use_external_recognizer=True)

    with pytest.raises(
        RuntimeError,
        match="external entity recognizer is enabled but no recognizer is configured",
    ):
        engine.analyze("FastAPI is useful.")


def test_entity_engine_validates_use_external_recognizer_type():
    with pytest.raises(
        TypeError,
        match="use_external_recognizer must be a boolean",
    ):
        EntityEngine(use_external_recognizer="invalid")

def test_analyze_uses_external_recognizer_when_enabled():
    recognizer = HuggingFaceEntityRecognizer()

    engine = EntityEngine(
        recognizer=recognizer,
        use_external_recognizer=True,
    )

    text = (
        "The application is built using FastAPI. "
        "The application uses MongoDB for data storage. "
        "Python is the main programming language."
    )

    results = engine.analyze(text)

    values = [result.text for result in results]

    assert "FastAPI" in values
    assert "MongoDB" in values
    assert "Python" in values


def test_analyze_preserves_external_entity_type():
    recognizer = HuggingFaceEntityRecognizer()

    engine = EntityEngine(
        recognizer=recognizer,
        use_external_recognizer=True,
    )
    text = (
    "The application is built using FastAPI. "
    "The application uses MongoDB for data storage. "
    "Python is the main programming language."
    )


    results = engine.analyze(text)

    fastapi = next(
        result for result in results
        if result.text == "FastAPI"
    )

    assert fastapi.entity_type == "MISC"


def test_analyze_uses_existing_pipeline_by_default():
    engine = EntityEngine()

    results = engine.analyze(
        "FastAPI is used for the application."
    )

    assert isinstance(results, list)


def test_external_recognizer_requires_recognizer():
    engine = EntityEngine(
        use_external_recognizer=True,
    )

    with pytest.raises(RuntimeError, match="no recognizer is configured"):
        engine.analyze("FastAPI is used.")


def test_use_external_recognizer_must_be_boolean():
    with pytest.raises(
        TypeError,
        match="use_external_recognizer must be a boolean",
    ):
        EntityEngine(
            use_external_recognizer="true",
        )