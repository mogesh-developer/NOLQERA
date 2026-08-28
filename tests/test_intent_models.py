import pytest

from nolqera.intelligence.intent.models import (
    IntentResult,
)


def test_intent_result_stores_analysis():

    result = IntentResult(
        intent="question",
        score=0.92,
        evidence_count=2,
    )

    assert result.intent == "question"
    assert result.score == 0.92
    assert result.evidence_count == 2


def test_intent_result_supports_metadata():

    result = IntentResult(
        intent="request",
        score=0.85,
        evidence_count=1,
        metadata={
            "source": "classifier",
        },
    )

    assert result.metadata["source"] == "classifier"


def test_intent_result_to_dict():

    result = IntentResult(
        intent="question",
        score=0.9,
        evidence_count=2,
    )

    data = result.to_dict()

    assert data["intent"] == "question"
    assert data["score"] == 0.9
    assert data["evidence_count"] == 2
    assert data["metadata"] is None


@pytest.mark.parametrize(
    "score",
    [-0.1, 1.1],
)
def test_invalid_score_is_rejected(score):

    with pytest.raises(ValueError):
        IntentResult(
            intent="question",
            score=score,
            evidence_count=1,
        )


def test_empty_intent_is_rejected():

    with pytest.raises(ValueError):

        IntentResult(
            intent="",
            score=0.8,
            evidence_count=1,
        )


def test_invalid_evidence_count_is_rejected():

    with pytest.raises(ValueError):

        IntentResult(
            intent="question",
            score=0.8,
            evidence_count=0,
        )


def test_invalid_metadata_is_rejected():

    with pytest.raises(TypeError):

        IntentResult(
            intent="question",
            score=0.8,
            evidence_count=1,
            metadata="invalid",
        )