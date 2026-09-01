import pytest

from nolqera.intelligence.entities.engine import EntityEngine
from nolqera.intelligence.pipeline.entity_analyzer import EntityAnalyzer


def test_entity_analyzer_accepts_entity_engine():
    engine = EntityEngine()
    analyzer = EntityAnalyzer(engine)

    assert analyzer is not None


def test_entity_analyzer_rejects_invalid_engine():
    with pytest.raises(TypeError, match="entity_engine must be an EntityEngine"):
        EntityAnalyzer(object())


def test_entity_analyzer_rejects_non_string_text():
    analyzer = EntityAnalyzer(EntityEngine())

    with pytest.raises(TypeError, match="text must be a string"):
        analyzer.analyze(123)


def test_entity_analyzer_rejects_empty_text():
    analyzer = EntityAnalyzer(EntityEngine())

    with pytest.raises(ValueError, match="text cannot be empty"):
        analyzer.analyze("")


def test_entity_analyzer_rejects_whitespace_text():
    analyzer = EntityAnalyzer(EntityEngine())

    with pytest.raises(ValueError, match="text cannot be empty"):
        analyzer.analyze("   ")


def test_entity_analyzer_delegates_to_entity_engine():
    engine = EntityEngine()
    analyzer = EntityAnalyzer(engine)

    result = analyzer.analyze(
        "OpenAI released a new model."
    )

    expected = engine.analyze(
        "OpenAI released a new model."
    )

    assert result == expected
    
    class FakeEntityEngine(EntityEngine):
        def analyze(self, text):
            return [
                {
                    "text": "OpenAI",
                    "label": "ORGANIZATION",
                }
            ]


    analyzer = EntityAnalyzer(FakeEntityEngine())

    result = analyzer.analyze("OpenAI released a model.")

    assert result == [
        {
            "text": "OpenAI",
            "label": "ORGANIZATION",
        }
    ]