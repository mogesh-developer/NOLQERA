import pytest

from nolqera.evaluation.models import (
    EvaluationContext,
    EvaluationRecord,
)


def test_evaluation_context_creation():
    context = EvaluationContext(
        text="Python is a programming language.",
        document_ids=["doc-1"],
        token_count=6,
    )

    assert context.text == "Python is a programming language."
    assert context.document_ids == ["doc-1"]
    assert context.token_count == 6


def test_evaluation_context_defaults():
    context = EvaluationContext(
        text="Some context."
    )

    assert context.document_ids == []
    assert context.token_count is None


def test_evaluation_context_rejects_negative_token_count():
    with pytest.raises(ValueError):
        EvaluationContext(
            text="Some context.",
            token_count=-1,
        )


def test_evaluation_context_rejects_invalid_token_count():
    with pytest.raises(TypeError):
        EvaluationContext(
            text="Some context.",
            token_count="10",
        )


def test_evaluation_record_creation():
    raw_context = EvaluationContext(
        text="Raw retrieved context.",
        document_ids=["doc-1", "doc-2"],
        token_count=100,
    )

    optimized_context = EvaluationContext(
        text="Optimized relevant context.",
        document_ids=["doc-1"],
        token_count=40,
    )

    record = EvaluationRecord(
        query="What is Python?",
        raw_context=raw_context,
        optimized_context=optimized_context,
        expected_information=[
            "Python is a programming language"
        ],
        metadata={
            "dataset": "test",
        },
    )

    assert record.query == "What is Python?"
    assert record.raw_context.token_count == 100
    assert record.optimized_context.token_count == 40
    assert len(record.expected_information) == 1
    assert record.metadata["dataset"] == "test"


def test_evaluation_record_rejects_empty_query():
    raw_context = EvaluationContext(text="Raw")
    optimized_context = EvaluationContext(text="Optimized")

    with pytest.raises(ValueError):
        EvaluationRecord(
            query="   ",
            raw_context=raw_context,
            optimized_context=optimized_context,
        )


def test_evaluation_record_requires_evaluation_context():
    with pytest.raises(TypeError):
        EvaluationRecord(
            query="Test query",
            raw_context="raw",
            optimized_context=EvaluationContext(
                text="optimized"
            ),
        )


def test_evaluation_record_defaults():
    raw_context = EvaluationContext(text="Raw")
    optimized_context = EvaluationContext(text="Optimized")

    record = EvaluationRecord(
        query="Test query",
        raw_context=raw_context,
        optimized_context=optimized_context,
    )

    assert record.expected_information == []
    assert record.metadata == {}


def test_evaluation_models_are_immutable():
    context = EvaluationContext(
        text="Some context."
    )

    with pytest.raises(Exception):
        context.text = "Modified"