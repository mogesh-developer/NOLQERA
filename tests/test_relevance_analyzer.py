import pytest

from nolqera.intelligence.pipeline.relevance_analyzer import RelevanceAnalyzer
from nolqera.intelligence.semantic_search.engine import SemanticSearchEngine


from nolqera.intelligence.semantic_similarity.embeddings.base import (
    EmbeddingProvider,
)


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, text):
        if isinstance(text, list):
            text_str = " ".join(text).lower()
        else:
            text_str = text.lower()
        return [1.0, 0.0] if "python" in text_str else [0.0, 1.0]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]



def create_analyzer():
    engine = SemanticSearchEngine(
        embedding_provider=FakeEmbeddingProvider()
    )
    return RelevanceAnalyzer(engine)


def test_relevance_analyzer_accepts_engine():
    analyzer = create_analyzer()
    assert isinstance(analyzer, RelevanceAnalyzer)


def test_relevance_analyzer_returns_exact_results():
    analyzer = create_analyzer()

    sentences = [
        "Python is a programming language.",
        "The weather is sunny.",
    ]

    result = analyzer.analyze(
        "Python programming",
        sentences,
    )

    assert result == [
        {
            "index": 0,
            "text": "Python is a programming language.",
            "score": result[0]["score"],
        },
        {
            "index": 1,
            "text": "The weather is sunny.",
            "score": result[1]["score"],
        },
    ]

    assert result[0]["index"] == 0
    assert result[1]["index"] == 1
    assert result[0]["score"] > result[1]["score"]


def test_relevance_analyzer_preserves_sentence_order():
    analyzer = create_analyzer()

    sentences = [
        "Python programming",
        "Weather information",
    ]

    result = analyzer.analyze(
        "Python",
        sentences,
    )

    assert [item["index"] for item in result] == [0, 1]


def test_relevance_analyzer_empty_sentences():
    analyzer = create_analyzer()

    assert analyzer.analyze("Python", []) == []


def test_relevance_analyzer_rejects_invalid_engine():
    with pytest.raises(TypeError):
        RelevanceAnalyzer(object())


def test_relevance_analyzer_rejects_invalid_query():
    analyzer = create_analyzer()

    with pytest.raises(TypeError):
        analyzer.analyze(123, ["Python"])


def test_relevance_analyzer_rejects_empty_query():
    analyzer = create_analyzer()

    with pytest.raises(ValueError):
        analyzer.analyze("   ", ["Python"])


def test_relevance_analyzer_rejects_invalid_sentences():
    analyzer = create_analyzer()

    with pytest.raises(TypeError):
        analyzer.analyze("Python", "Python")


def test_relevance_analyzer_rejects_empty_sentence():
    analyzer = create_analyzer()

    with pytest.raises(ValueError):
        analyzer.analyze("Python", [""])