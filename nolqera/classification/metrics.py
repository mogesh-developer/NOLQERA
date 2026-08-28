def accuracy_score(
    y_true: list[str],
    y_pred: list[str],
) -> float:
    """Calculate classification accuracy."""

    _validate_inputs(y_true, y_pred)

    if not y_true:
        raise ValueError("y_true cannot be empty")

    correct = sum(
        true == pred
        for true, pred in zip(y_true, y_pred)
    )

    return correct / len(y_true)


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix."""

    _validate_inputs(y_true, y_pred)

    if labels is None:
        labels = sorted(
            set(y_true) | set(y_pred)
        )

    matrix = {
        actual: {
            predicted: 0
            for predicted in labels
        }
        for actual in labels
    }

    for actual, predicted in zip(
        y_true,
        y_pred,
    ):
        matrix[actual][predicted] += 1

    return matrix


def precision_score(
    y_true: list[str],
    y_pred: list[str],
    positive_label: str,
) -> float:
    """Calculate precision for a binary classifier."""

    _validate_inputs(y_true, y_pred)

    tp = sum(
        actual == positive_label
        and predicted == positive_label
        for actual, predicted in zip(
            y_true,
            y_pred,
        )
    )

    fp = sum(
        actual != positive_label
        and predicted == positive_label
        for actual, predicted in zip(
            y_true,
            y_pred,
        )
    )

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score(
    y_true: list[str],
    y_pred: list[str],
    positive_label: str,
) -> float:
    """Calculate recall for a binary classifier."""

    _validate_inputs(y_true, y_pred)

    tp = sum(
        actual == positive_label
        and predicted == positive_label
        for actual, predicted in zip(
            y_true,
            y_pred,
        )
    )

    fn = sum(
        actual == positive_label
        and predicted != positive_label
        for actual, predicted in zip(
            y_true,
            y_pred,
        )
    )

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(
    y_true: list[str],
    y_pred: list[str],
    positive_label: str,
) -> float:
    """Calculate F1 score for a binary classifier."""

    precision = precision_score(
        y_true,
        y_pred,
        positive_label,
    )

    recall = recall_score(
        y_true,
        y_pred,
        positive_label,
    )

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def _validate_inputs(
    y_true: list[str],
    y_pred: list[str],
) -> None:
    if not isinstance(y_true, list):
        raise TypeError(
            "y_true must be a list"
        )

    if not isinstance(y_pred, list):
        raise TypeError(
            "y_pred must be a list"
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have "
            "the same length"
        )