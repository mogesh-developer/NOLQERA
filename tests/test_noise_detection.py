import pytest

from nolqera.intelligence.context_optimization.noise_detection import (
    NoiseDetector,
)
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)


def make_result(text: str, score: float = 0.8, index: int = 0):
    return SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )


def test_detector_accepts_configuration():

    detector = NoiseDetector(
        min_meaningful_tokens=2
    )

    assert detector.min_meaningful_tokens == 2


def test_detector_rejects_invalid_token_threshold():

    with pytest.raises(TypeError):
        NoiseDetector("2")


def test_detector_rejects_zero_token_threshold():

    with pytest.raises(ValueError):
        NoiseDetector(0)


def test_detector_rejects_negative_token_threshold():

    with pytest.raises(ValueError):
        NoiseDetector(-1)


def test_detects_empty_text_as_noise():

    detector = NoiseDetector()

    result = object.__new__(SemanticSearchResult)
    object.__setattr__(result, "text", "")
    object.__setattr__(result, "score", 0.8)
    object.__setattr__(result, "index", 0)

    assert detector.is_noise(result) is True


def test_detects_punctuation_only_as_noise():

    detector = NoiseDetector()

    result = make_result("!!! ??? ...")

    assert detector.is_noise(result) is True


def test_detects_single_token_as_noise():

    detector = NoiseDetector(
        min_meaningful_tokens=2
    )

    result = make_result("FastAPI")

    assert detector.is_noise(result) is True


def test_keeps_meaningful_text():

    detector = NoiseDetector(
        min_meaningful_tokens=2
    )

    result = make_result(
        "FastAPI authentication"
    )

    assert detector.is_noise(result) is False


def test_keeps_text_with_numbers():

    detector = NoiseDetector(
        min_meaningful_tokens=2
    )

    result = make_result(
        "Python 3.11 framework"
    )

    assert detector.is_noise(result) is False


def test_filter_removes_noise():

    detector = NoiseDetector()

    results = [
        make_result("FastAPI authentication", index=0),
        make_result("!!!", index=1),
        make_result("JWT security", index=2),
    ]

    filtered = detector.filter(results)

    assert [result.text for result in filtered] == [
        "FastAPI authentication",
        "JWT security",
    ]


def test_filter_preserves_order():

    detector = NoiseDetector()

    results = [
        make_result("Python backend API", index=0),
        make_result("???", index=1),
        make_result("JWT authentication security", index=2),
        make_result("...", index=3),
    ]

    filtered = detector.filter(results)

    assert [result.index for result in filtered] == [
        0,
        2,
    ]


def test_filter_preserves_result_objects():

    detector = NoiseDetector()

    first = make_result(
        "FastAPI authentication",
        index=0,
    )

    second = make_result(
        "JWT security",
        index=1,
    )

    results = [first, second]

    filtered = detector.filter(results)

    assert filtered[0] is first
    assert filtered[1] is second


def test_filter_empty_results():

    detector = NoiseDetector()

    assert detector.filter([]) == []


def test_filter_rejects_non_list():

    detector = NoiseDetector()

    with pytest.raises(TypeError):
        detector.filter(None)


def test_filter_rejects_invalid_result():

    detector = NoiseDetector()

    with pytest.raises(TypeError):
        detector.filter(
            ["FastAPI authentication"]
        )


def test_is_noise_rejects_invalid_result():

    detector = NoiseDetector()

    with pytest.raises(TypeError):
        detector.is_noise("FastAPI")


def test_custom_threshold_controls_detection():

    detector = NoiseDetector(
        min_meaningful_tokens=3
    )

    result = make_result(
        "FastAPI authentication"
    )

    assert detector.is_noise(result) is True

    result = make_result(
        "FastAPI JWT authentication"
    )

    assert detector.is_noise(result) is False