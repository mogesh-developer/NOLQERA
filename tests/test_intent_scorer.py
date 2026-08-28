import pytest

from nolqera.intelligence.intent.classifier import (
    IntentClassification,
)

from nolqera.intelligence.intent.scorer import (
    IntentScorer,
    IntentScore,
)


def test_single_intent_score():

    scorer = IntentScorer()

    classifications = [
        IntentClassification(
            intent="question",
            score=0.8,
        )
    ]

    results = scorer.score(
        classifications
    )

    assert len(results) == 1
    assert isinstance(
        results[0],
        IntentScore,
    )

    assert results[0].intent == "question"
    assert results[0].score == pytest.approx(0.8)
    assert results[0].evidence_count == 1


def test_multiple_signals_are_combined():

    scorer = IntentScorer()

    classifications = [
        IntentClassification(
            intent="question",
            score=0.7,
        ),
        IntentClassification(
            intent="question",
            score=0.6,
        ),
    ]

    results = scorer.score(
        classifications
    )

    assert len(results) == 1

    assert results[0].intent == "question"

    assert results[0].score == pytest.approx(
        0.88
    )

    assert results[0].evidence_count == 2


def test_different_intents_remain_separate():

    scorer = IntentScorer()

    classifications = [
        IntentClassification(
            intent="question",
            score=0.8,
        ),
        IntentClassification(
            intent="request",
            score=0.6,
        ),
    ]

    results = scorer.score(
        classifications
    )

    assert len(results) == 2

    intents = {
        result.intent
        for result in results
    }

    assert intents == {
        "question",
        "request",
    }


def test_empty_classifications_return_empty():

    scorer = IntentScorer()

    assert scorer.score([]) == []


def test_invalid_input_is_rejected():

    scorer = IntentScorer()

    with pytest.raises(TypeError):
        scorer.score("invalid")


def test_invalid_classification_is_rejected():

    scorer = IntentScorer()

    with pytest.raises(TypeError):
        scorer.score(
            ["invalid"]
        )