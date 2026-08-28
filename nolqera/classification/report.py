from .metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_report(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Generate precision, recall and F1 metrics per class."""

    if not isinstance(y_true, list):
        raise TypeError("y_true must be a list")

    if not isinstance(y_pred, list):
        raise TypeError("y_pred must be a list")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have "
            "the same length"
        )

    if not y_true:
        raise ValueError(
            "y_true cannot be empty"
        )

    if labels is None:
        labels = sorted(
            set(y_true) | set(y_pred)
        )

    report = {}

    for label in labels:
        report[label] = {
            "precision": precision_score(
                y_true,
                y_pred,
                label,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                label,
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                label,
            ),
        }

    report["accuracy"] = {
        "score": accuracy_score(
            y_true,
            y_pred,
        )
    }

    return report