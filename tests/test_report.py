from nolqera.classification.report import (
    classification_report,
)


def test_classification_report():
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

    report = classification_report(
        y_true,
        y_pred,
    )

    assert "positive" in report
    assert "negative" in report
    assert "accuracy" in report

    assert (
        report["positive"]["precision"]
        == 0.5
    )

    assert (
        report["positive"]["recall"]
        == 0.5
    )

    assert (
        report["positive"]["f1"]
        == 0.5
    )

    assert (
        report["accuracy"]["score"]
        == 0.5
    )