import pytest

from nolqera.intelligence.intent.candidates import (
    IntentCandidate,
)

from nolqera.intelligence.intent.classifier import (
    IntentClassifier,
    IntentClassification,
)


def test_question_signal_is_classified():

    classifier = IntentClassifier()

    candidates = [
        IntentCandidate(
            text="What is MongoDB?",
            signal="question_form",
            score=1.0,
        )
    ]

    results = classifier.classify(candidates)

    assert len(results) == 1

    assert isinstance(
        results[0],
        IntentClassification,
    )

    assert results[0].intent == "question"
    assert results[0].score == 1.0


def test_unknown_signal_is_not_rejected():

    classifier = IntentClassifier()

    candidates = [
        IntentCandidate(
            text="FastAPI MongoDB",
            signal="unknown_signal",
            score=0.5,
        )
    ]

    results = classifier.classify(candidates)

    assert len(results) == 1
    assert results[0].intent == "unknown"
    assert results[0].score == 0.5


def test_empty_candidates_return_empty_result():

    classifier = IntentClassifier()

    assert classifier.classify([]) == []


def test_invalid_candidates_are_rejected():

    classifier = IntentClassifier()

    with pytest.raises(TypeError):

        classifier.classify(
            ["invalid"]
        )


def test_invalid_input_is_rejected():

    classifier = IntentClassifier()

    with pytest.raises(TypeError):

        classifier.classify("invalid")