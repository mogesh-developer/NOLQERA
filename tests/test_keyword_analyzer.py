import pytest

from nolqera.intelligence.pipeline.keyword_analyzer import KeywordAnalyzer
from nolqera.intelligence.keyphrase.engine import KeyphraseEngine


class FakeKeyphraseEngine(KeyphraseEngine):
    def analyze(self, text, top_k=5):
        return {
            "keywords": ["python", "api"][:top_k],
            "keyphrases": ["python api"][:top_k],
        }


def create_analyzer():
    return KeywordAnalyzer(FakeKeyphraseEngine())


def test_keyword_analyzer_accepts_engine():
    analyzer = create_analyzer()
    assert isinstance(analyzer, KeywordAnalyzer)


def test_keyword_analyzer_returns_exact_result():
    analyzer = create_analyzer()

    result = analyzer.analyze(
        "Python API development",
        top_k=2,
    )

    assert result == {
        "keywords": ["python", "api"],
        "keyphrases": ["python api"],
    }


def test_keyword_analyzer_respects_top_k():
    analyzer = create_analyzer()

    result = analyzer.analyze(
        "Python API development",
        top_k=1,
    )

    assert result["keywords"] == ["python"]
    assert result["keyphrases"] == ["python api"]


def test_keyword_analyzer_rejects_invalid_engine():
    with pytest.raises(TypeError):
        KeywordAnalyzer(object())


def test_keyword_analyzer_rejects_non_string():
    analyzer = create_analyzer()

    with pytest.raises(TypeError):
        analyzer.analyze(123)


def test_keyword_analyzer_rejects_empty_text():
    analyzer = create_analyzer()

    with pytest.raises(ValueError):
        analyzer.analyze("   ")


def test_keyword_analyzer_rejects_invalid_top_k_type():
    analyzer = create_analyzer()

    with pytest.raises(TypeError):
        analyzer.analyze("Python", top_k="5")


def test_keyword_analyzer_rejects_invalid_top_k():
    analyzer = create_analyzer()

    with pytest.raises(ValueError):
        analyzer.analyze("Python", top_k=0)