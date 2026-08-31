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
from nolqera.intelligence.context_optimization.redundant_information_collapse import (
    collapse_redundant_information,
)


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic embedding provider for testing.
    """

    def embed(self, tokens: list[str] | str) -> list[float]:

        mapping = {
            "fastapi is a python backend framework": [
                1.0,
                0.0,
                0.0,
            ],
            "fastapi is a modern python api framework": [
                0.98,
                0.02,
                0.0,
            ],
            "fastapi uses python type hints": [
                0.55,
                0.45,
                0.0,
            ],
            "mongodb is a nosql database": [
                0.0,
                1.0,
                0.0,
            ],
            "react is a frontend library": [
                0.0,
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


def make_result(
    text: str,
    score: float,
    index: int,
):
    return SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )


def test_collapse_returns_results(engine):

    results = [
        make_result(
            "FastAPI is a Python backend framework",
            0.95,
            0,
        )
    ]

    collapsed = collapse_redundant_information(
        results,
        engine,
    )

    assert isinstance(collapsed, list)


def test_collapse_removes_semantically_redundant_result(
    engine,
):

    results = [
        make_result(
            "FastAPI is a Python backend framework",
            0.95,
            0,
        ),
        make_result(
            "FastAPI is a modern Python API framework",
            0.90,
            1,
        ),
    ]

    collapsed = collapse_redundant_information(
        results,
        engine,
        similarity_threshold=0.9,
    )

    assert len(collapsed) == 1
    assert collapsed[0].index == 0


def test_collapse_preserves_first_highest_ranked_result(
    engine,
):

    first = make_result(
        "FastAPI is a Python backend framework",
        0.95,
        0,
    )

    second = make_result(
        "FastAPI is a modern Python API framework",
        0.90,
        1,
    )

    collapsed = collapse_redundant_information(
        [first, second],
        engine,
        similarity_threshold=0.9,
    )

    assert collapsed[0] is first


def test_collapse_keeps_non_redundant_information(
    engine,
):

    results = [
        make_result(
            "FastAPI is a Python backend framework",
            0.95,
            0,
        ),
        make_result(
            "MongoDB is a NoSQL database",
            0.85,
            1,
        ),
    ]

    collapsed = collapse_redundant_information(
        results,
        engine,
        similarity_threshold=0.9,
    )

    assert len(collapsed) == 2


def test_collapse_preserves_order(engine):

    results = [
        make_result(
            "FastAPI is a Python backend framework",
            0.95,
            0,
        ),
        make_result(
            "MongoDB is a NoSQL database",
            0.85,
            1,
        ),
        make_result(
            "React is a frontend library",
            0.80,
            2,
        ),
    ]

    collapsed = collapse_redundant_information(
        results,
        engine,
        similarity_threshold=0.9,
    )

    assert [result.index for result in collapsed] == [
        0,
        1,
        2,
    ]


def test_collapse_preserves_result_objects(engine):

    result = make_result(
        "FastAPI is a Python backend framework",
        0.95,
        0,
    )

    collapsed = collapse_redundant_information(
        [result],
        engine,
    )

    assert collapsed[0] is result


def test_collapse_empty_results(engine):

    collapsed = collapse_redundant_information(
        [],
        engine,
    )

    assert collapsed == []


def test_collapse_all_redundant_results(engine):

    results = [
        make_result(
            "FastAPI is a Python backend framework",
            0.95,
            0,
        ),
        make_result(
            "FastAPI is a modern Python API framework",
            0.90,
            1,
        ),
        make_result(
            "FastAPI is a modern Python API framework",
            0.85,
            2,
        ),
    ]

    collapsed = collapse_redundant_information(
        results,
        engine,
        similarity_threshold=0.9,
    )

    assert len(collapsed) == 1
    assert collapsed[0].index == 0


def test_threshold_controls_collapse(engine):

    results = [
        make_result(
            "FastAPI is a Python backend framework",
            0.95,
            0,
        ),
        make_result(
            "FastAPI uses Python type hints",
            0.80,
            1,
        ),
    ]

    collapsed = collapse_redundant_information(
        results,
        engine,
        similarity_threshold=0.9,
    )

    assert len(collapsed) == 2


def test_result_at_threshold_is_collapsed(engine):

    results = [
        make_result(
            "FastAPI is a Python backend framework",
            0.95,
            0,
        ),
        make_result(
            "FastAPI is a modern Python API framework",
            0.90,
            1,
        ),
    ]

    collapsed = collapse_redundant_information(
        results,
        engine,
        similarity_threshold=0.9,
    )

    assert len(collapsed) == 1


def test_rejects_non_list(engine):

    with pytest.raises(TypeError):

        collapse_redundant_information(
            "invalid",
            engine,
        )


def test_rejects_invalid_result_type(engine):

    with pytest.raises(TypeError):

        collapse_redundant_information(
            ["invalid"],
            engine,
        )


def test_rejects_invalid_engine():

    with pytest.raises(TypeError):

        collapse_redundant_information(
            [],
            None,
        )


def test_rejects_non_numeric_threshold(engine):

    with pytest.raises(TypeError):

        collapse_redundant_information(
            [],
            engine,
            similarity_threshold="0.9",
        )


def test_rejects_threshold_above_one(engine):

    with pytest.raises(ValueError):

        collapse_redundant_information(
            [],
            engine,
            similarity_threshold=1.1,
        )


def test_rejects_negative_threshold(engine):

    with pytest.raises(ValueError):

        collapse_redundant_information(
            [],
            engine,
            similarity_threshold=-0.1,
        )