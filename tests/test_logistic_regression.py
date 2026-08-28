import pytest

from nolqera.classification.logistic_regression import (
    LogisticRegression,
)


def test_sigmoid():
    model = LogisticRegression()

    assert model._sigmoid(0) == 0.5
    assert model._sigmoid(10) > 0.99
    assert model._sigmoid(-10) < 0.01


def test_fit_and_predict():
    model = LogisticRegression(
        learning_rate=0.1,
        epochs=1000,
    )

    X = [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ]

    y = [
        0,
        0,
        1,
        1,
    ]

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions == [
        0,
        0,
        1,
        1,
    ]


def test_probability():
    model = LogisticRegression()

    model.fit(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ],
        [0, 0, 1, 1],
    )

    probability = model.predict_probability(
        [3.0]
    )

    assert 0.5 < probability <= 1.0


def test_invalid_label():
    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(
            [[1.0]],
            [2],
        )


def test_invalid_learning_rate():
    with pytest.raises(ValueError):
        LogisticRegression(
            learning_rate=0
        )

def test_binary_cross_entropy():
    model = LogisticRegression()

    loss = model._binary_cross_entropy(
        [1, 0],
        [0.9, 0.1],
    )

    assert loss < 0.2


def test_loss_history():
    model = LogisticRegression(learning_rate=0.1, epochs=100)
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]

    model.fit(X, y)

    assert len(model.loss_history) == 100
    assert model.loss_history[-1] < model.loss_history[0]

def test_loss_decreases():
    model = LogisticRegression(
        learning_rate=0.1,
        epochs=100,
    )

    X = [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ]

    y = [
        0,
        0,
        1,
        1,
    ]

    model.fit(X, y)

    assert len(model.loss_history) == 100

    assert (
        model.loss_history[-1]
        < model.loss_history[0]
    )