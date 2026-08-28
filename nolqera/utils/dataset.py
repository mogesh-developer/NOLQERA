import random
from typing import TypeVar


T = TypeVar("T")


def train_test_split(
    data: list[T],
    labels: list[str],
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[list[T], list[T], list[str], list[str]]:
    """Split data and labels into training and test sets."""

    if not isinstance(data, list):
        raise TypeError("data must be a list")

    if not isinstance(labels, list):
        raise TypeError("labels must be a list")

    if len(data) != len(labels):
        raise ValueError(
            "data and labels must have the same length"
        )

    if not data:
        raise ValueError("data cannot be empty")

    if not isinstance(test_size, (int, float)):
        raise TypeError(
            "test_size must be a number"
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1"
        )

    if random_state is not None:
        if not isinstance(random_state, int):
            raise TypeError(
                "random_state must be an integer"
            )

    indices = list(range(len(data)))

    rng = random.Random(random_state)
    rng.shuffle(indices)

    test_count = max(
        1,
        round(len(data) * test_size),
    )

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = [data[i] for i in train_indices]
    X_test = [data[i] for i in test_indices]

    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )