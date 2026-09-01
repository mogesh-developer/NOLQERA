import pytest

from nolqera.intelligence.pipeline.importance_analyzer import ImportanceAnalyzer
from nolqera.intelligence.importance.engine import ImportanceEngine


from nolqera.intelligence.importance.models import ImportanceResult


class FakeImportanceEngine(ImportanceEngine):
    def analyze(self, sentences):
        return [
            ImportanceResult(
                sentence=sentence,
                score=1.0 - (index * 0.1),
                rank=index + 1
            )
            for index, sentence in enumerate(sentences)
        ]


def create_analyzer():
    return ImportanceAnalyzer(FakeImportanceEngine())


def test_importance_analyzer_accepts_engine():
    analyzer = create_analyzer()
    assert isinstance(analyzer, ImportanceAnalyzer)


def test_importance_analyzer_returns_exact_results():
    analyzer = create_analyzer()

    sentences = [
        "Python is a programming language.",
        "FastAPI is a web framework.",
    ]

    result = analyzer.analyze(sentences)

    assert result == [
        {
            "index": 0,
            "text": "Python is a programming language.",
            "score": 1.0,
        },
        {
            "index": 1,
            "text": "FastAPI is a web framework.",
            "score": 0.9,
        },
    ]


def test_importance_analyzer_preserves_order():
    analyzer = create_analyzer()

    sentences = [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]

    result = analyzer.analyze(sentences)

    assert [item["index"] for item in result] == [0, 1, 2]


def test_importance_analyzer_delegates_to_engine():
    analyzer = create_analyzer()

    sentences = ["Important information."]

    result = analyzer.analyze(sentences)

    assert result[0]["score"] == 1.0


def test_importance_analyzer_empty_sentences():
    analyzer = create_analyzer()

    assert analyzer.analyze([]) == []


def test_importance_analyzer_rejects_invalid_engine():
    with pytest.raises(TypeError):
        ImportanceAnalyzer(object())


def test_importance_analyzer_rejects_non_list():
    analyzer = create_analyzer()

    with pytest.raises(TypeError):
        analyzer.analyze("Python")


def test_importance_analyzer_rejects_non_string_sentence():
    analyzer = create_analyzer()

    with pytest.raises(TypeError):
        analyzer.analyze(["Python", 123])


def test_importance_analyzer_rejects_empty_sentence():
    analyzer = create_analyzer()

    with pytest.raises(ValueError):
        analyzer.analyze(["Python", ""])