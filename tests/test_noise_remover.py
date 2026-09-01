import pytest

from nolqera.intelligence.context_optimization.noise_detection import (
    NoiseDetector,
)
from nolqera.intelligence.pipeline.noise_remover import NoiseRemover
from nolqera.intelligence.semantic_search.models import SemanticSearchResult


def make_result(
    text: str,
    score: float = 0.9,
    index: int = 0,
) -> SemanticSearchResult:
    return SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )


def test_noise_remover_accepts_noise_detector():
    detector = NoiseDetector()

    remover = NoiseRemover(detector)

    assert isinstance(remover, NoiseRemover)


def test_noise_remover_rejects_invalid_detector():
    with pytest.raises(
        TypeError,
        match="noise_detector must be a NoiseDetector",
    ):
        NoiseRemover(object())


def test_noise_remover_rejects_non_list():
    remover = NoiseRemover(NoiseDetector())

    with pytest.raises(
        TypeError,
        match="results must be a list",
    ):
        remover.remove("not a list")


def test_noise_remover_rejects_invalid_result():
    remover = NoiseRemover(NoiseDetector())

    with pytest.raises(
        TypeError,
        match="all results must be SemanticSearchResult instances",
    ):
        remover.remove(["invalid"])


def test_noise_remover_removes_noise():
    remover = NoiseRemover(NoiseDetector())

    results = [
        make_result("FastAPI Python", 0.9, 0),
        make_result("...", 0.8, 1),
        make_result("machine learning", 0.7, 2),
    ]

    filtered = remover.remove(results)

    assert filtered == [
        results[0],
        results[2],
    ]


def test_noise_remover_preserves_order():
    remover = NoiseRemover(NoiseDetector())

    results = [
        make_result("Python backend", 0.9, 0),
        make_result("React frontend", 0.8, 1),
        make_result("!!!", 0.7, 2),
        make_result("MongoDB database", 0.6, 3),
    ]

    filtered = remover.remove(results)

    assert [result.text for result in filtered] == [
        "Python backend",
        "React frontend",
        "MongoDB database",
    ]


def test_noise_remover_preserves_result_objects():
    remover = NoiseRemover(NoiseDetector())

    first = make_result("Python backend", 0.9, 0)
    noise = make_result("!!!", 0.8, 1)

    filtered = remover.remove([first, noise])

    assert filtered[0] is first


def test_noise_remover_keeps_meaningful_result_exactly():
    remover = NoiseRemover(NoiseDetector())

    result = make_result(
        "FastAPI Python backend API",
        0.9321,
        4,
    )

    filtered = remover.remove([result])

    assert len(filtered) == 1
    assert filtered[0] is result
    assert filtered[0].text == "FastAPI Python backend API"
    assert filtered[0].score == pytest.approx(0.9321)
    assert filtered[0].index == 4


def test_noise_remover_empty_results():
    remover = NoiseRemover(NoiseDetector())

    assert remover.remove([]) == []


def test_noise_remover_respects_detector_threshold():
    remover = NoiseRemover(
        NoiseDetector(min_meaningful_tokens=3)
    )

    results = [
        make_result("Python backend API", 0.9, 0),
        make_result("Python backend", 0.8, 1),
        make_result("Python", 0.7, 2),
    ]

    filtered = remover.remove(results)

    assert filtered == [
        results[0],
    ]