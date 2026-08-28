import pytest

from nolqera.classification.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def test_accuracy_score():
    y_true = [
        "positive",
        "positive",
        "negative",
        "negative",
    ]

    y_pred = [
        "positive",
        "negative",
        "negative",
        "negative",
    ]

    assert accuracy_score(
        y_true,
        y_pred,
    ) == 0.75


def test_confusion_matrix():
    y_true = [
        "positive",
        "positive",
        "negative",
        "negative",
    ]

    y_pred = [
        "positive",
        "negative",
        "negative",
        "negative",
    ]

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    assert matrix["positive"]["positive"] == 1
    assert matrix["positive"]["negative"] == 1
    assert matrix["negative"]["negative"] == 2
    assert matrix["negative"]["positive"] == 0


def test_precision():
    y_true = [
        "positive",
        "positive",
        "negative",
        "negative",
    ]

    y_pred = [
        "positive",
        "negative",
        "positive",
        "negative",
    ]

    assert precision_score(
        y_true,
        y_pred,
        "positive",
    ) == 0.5


def test_recall():
    y_true = [
        "positive",
        "positive",
        "negative",
        "negative",
    ]

    y_pred = [
        "positive",
        "negative",
        "positive",
        "negative",
    ]

    assert recall_score(
        y_true,
        y_pred,
        "positive",
    ) == 0.5


def test_f1():
    y_true = [
        "positive",
        "positive",
        "negative",
        "negative",
    ]

    y_pred = [
        "positive",
        "negative",
        "positive",
        "negative",
    ]

    assert f1_score(
        y_true,
        y_pred,
        "positive",
    ) == 0.5


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        accuracy_score(
            ["positive"],
            [],
        )