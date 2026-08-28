from nolqera.utils.dataset import train_test_split

import pytest

def test_train_test_split():
    data = [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]

    labels = [
        "x",
        "x",
        "y",
        "y",
        "y",
    ]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            data,
            labels,
            test_size=0.2,
            random_state=42,
        )
    )

    assert len(X_train) == 4
    assert len(X_test) == 1

    assert len(y_train) == 4
    assert len(y_test) == 1

def test_random_state_reproducibility():
    data = ["a", "b", "c", "d", "e"]
    labels = ["x", "x", "y", "y", "y"]

    result1 = train_test_split(
        data,
        labels,
        random_state=42,
    )

    result2 = train_test_split(
        data,
        labels,
        random_state=42,
    )

    assert result1 == result2




def test_invalid_test_size():
    with pytest.raises(ValueError):
        train_test_split(
            ["a"],
            ["x"],
            test_size=1.5,
        )


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        train_test_split(
            ["a", "b"],
            ["x"],
        )