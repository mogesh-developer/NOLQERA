import pytest

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)

from nolqera.intelligence.semantic_similarity.engine import (
    SemanticSimilarityEngine,
)

from nolqera.intelligence.semantic_similarity.embeddings.base import (
    EmbeddingProvider,
)
from nolqera.intelligence.context_optimization.semantic_redundancy import (
    detect_semantic_redundancy,
    remove_semantic_redundancy,
)


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Small deterministic provider for unit testing.

    Similar meanings are intentionally mapped to similar vectors.
    """

    def embed(self, tokens: list[str] | str) -> list[float]:

        mapping = {
            "fastapi is a python backend framework": [
                1.0,
                0.0,
            ],
            "fastapi is a modern python api framework": [
                0.98,
                0.02,
            ],
            "fastapi uses python type hints": [
                0.55,
                0.45,
            ],
            "mongodb is a nosql database": [
                0.0,
                1.0,
            ],
        }

        if isinstance(tokens, str):
            text = tokens
        else:
            text = " ".join(tokens)

        key = text.lower().strip()

        if key not in mapping:
            raise ValueError(
                f"Unknown test text: {text}"
            )

        return mapping[key]


@pytest.fixture
def engine():

    return SemanticSimilarityEngine(
        embedding_provider=FakeEmbeddingProvider()
    )


def _result(
    text: str,
    score: float,
    index: int,
):
    return SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )


def test_detects_semantic_redundancy(engine):

    first = (
        "FastAPI is a Python backend framework"
    )

    second = (
        "FastAPI is a modern Python API framework"
    )

    assert detect_semantic_redundancy(
        first,
        second,
        engine,
        similarity_threshold=0.9,
    )


def test_rejects_semantically_different_text(engine):

    first = (
        "FastAPI is a Python backend framework"
    )

    second = (
        "MongoDB is a NoSQL database"
    )

    assert not detect_semantic_redundancy(
        first,
        second,
        engine,
        similarity_threshold=0.9,
    )


def test_threshold_controls_detection(engine):

    first = (
        "FastAPI is a Python backend framework"
    )

    second = (
        "FastAPI uses Python type hints"
    )

    assert not detect_semantic_redundancy(
        first,
        second,
        engine,
        similarity_threshold=0.9,
    )


def test_identical_text_is_redundant(engine):

    first = (
        "FastAPI is a Python backend framework"
    )

    assert detect_semantic_redundancy(
        first,
        first,
        engine,
        similarity_threshold=0.99,
    )


def test_remove_semantic_redundancy(engine):

    results = [
        _result(
            "FastAPI is a Python backend framework",
            0.95,
            0,
        ),
        _result(
            "FastAPI is a modern Python API framework",
            0.90,
            1,
        ),
        _result(
            "MongoDB is a NoSQL database",
            0.80,
            2,
        ),
    ]

    filtered = remove_semantic_redundancy(
        results,
        engine,
        similarity_threshold=0.9,
    )

    assert len(filtered) == 2

    assert filtered[0].index == 0
    assert filtered[1].index == 2


def test_first_result_is_preserved(engine):

    first = _result(
        "FastAPI is a Python backend framework",
        0.95,
        0,
    )

    second = _result(
        "FastAPI is a modern Python API framework",
        0.85,
        1,
    )

    filtered = remove_semantic_redundancy(
        [first, second],
        engine,
        similarity_threshold=0.9,
    )

    assert filtered == [first]


def test_non_redundant_results_are_preserved(engine):

    first = _result(
        "FastAPI is a Python backend framework",
        0.95,
        0,
    )

    second = _result(
        "MongoDB is a NoSQL database",
        0.80,
        1,
    )

    filtered = remove_semantic_redundancy(
        [first, second],
        engine,
        similarity_threshold=0.9,
    )

    assert filtered == [
        first,
        second,
    ]


def test_order_is_preserved(engine):

    results = [
        _result(
            "FastAPI is a Python backend framework",
            0.95,
            0,
        ),
        _result(
            "MongoDB is a NoSQL database",
            0.85,
            1,
        ),
    ]

    filtered = remove_semantic_redundancy(
        results,
        engine,
        similarity_threshold=0.9,
    )

    assert [r.index for r in filtered] == [
        0,
        1,
    ]


def test_result_objects_are_preserved(engine):

    result = _result(
        "FastAPI is a Python backend framework",
        0.95,
        0,
    )

    filtered = remove_semantic_redundancy(
        [result],
        engine,
    )

    assert filtered[0] is result


def test_empty_results_return_empty_list(engine):

    assert remove_semantic_redundancy(
        [],
        engine,
    ) == []


def test_rejects_non_list(engine):

    with pytest.raises(TypeError):
        remove_semantic_redundancy(
            "invalid",
            engine,
        )


def test_rejects_invalid_result_type(engine):

    with pytest.raises(TypeError):
        remove_semantic_redundancy(
            ["invalid"],
            engine,
        )


def test_rejects_invalid_threshold(engine):

    with pytest.raises(ValueError):
        remove_semantic_redundancy(
            [],
            engine,
            similarity_threshold=1.5,
        )


def test_rejects_negative_threshold(engine):

    with pytest.raises(ValueError):
        remove_semantic_redundancy(
            [],
            engine,
            similarity_threshold=-0.1,
        )


def test_rejects_invalid_engine():

    with pytest.raises(TypeError):
        detect_semantic_redundancy(
            "first",
            "second",
            None,
        )


def test_rejects_empty_first(engine):

    with pytest.raises(ValueError):
        detect_semantic_redundancy(
            "",
            "text",
            engine,
        )


def test_rejects_empty_second(engine):

    with pytest.raises(ValueError):
        detect_semantic_redundancy(
            "text",
            "",
            engine,
        )