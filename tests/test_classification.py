import pytest

from nolqera.classification.naive_bayes import (
    MultinomialNaiveBayes,
)
from nolqera.classification.text_classifier import (
    TextClassifier,
)


def test_fit():
    classifier = MultinomialNaiveBayes()

    classifier.fit(
        [
            ["i", "love", "nlp"],
            ["i", "love", "python"],
            ["i", "hate", "bugs"],
        ],
        [
            "positive",
            "positive",
            "negative",
        ],
    )

    assert classifier.classes_ == [
        "negative",
        "positive",
    ]

    assert classifier.total_documents == 3


def test_class_prior():
    classifier = MultinomialNaiveBayes()

    classifier.fit(
        [
            ["good"],
            ["great"],
            ["bad"],
        ],
        [
            "positive",
            "positive",
            "negative",
        ],
    )

    assert classifier.class_prior("positive") == 2 / 3
    assert classifier.class_prior("negative") == 1 / 3


def test_token_probability():
    classifier = MultinomialNaiveBayes()

    classifier.fit(
        [
            ["good", "good"],
            ["good", "great"],
            ["bad"],
        ],
        [
            "positive",
            "positive",
            "negative",
        ],
    )

    probability = classifier.token_probability(
        "good",
        "positive",
    )

    assert probability > 0


def test_unknown_token_probability():
    classifier = MultinomialNaiveBayes()

    classifier.fit(
        [
            ["good"],
            ["bad"],
        ],
        [
            "positive",
            "negative",
        ],
    )

    probability = classifier.token_probability(
        "excellent",
        "positive",
    )

    assert probability > 0


def test_invalid_alpha():
    with pytest.raises(ValueError):
        MultinomialNaiveBayes(alpha=0)


def test_mismatched_data():
    classifier = MultinomialNaiveBayes()

    with pytest.raises(ValueError):
        classifier.fit(
            [["good"]],
            [],
        )

def test_predict_one():
    classifier = MultinomialNaiveBayes()

    classifier.fit(
        [
            ["good", "excellent"],
            ["good", "nice"],
            ["bad", "terrible"],
            ["bad", "awful"],
        ],
        [
            "positive",
            "positive",
            "negative",
            "negative",
        ],
    )

    assert (
        classifier.predict_one(
            ["good", "nice"]
        )
        == "positive"
    )

    assert (
        classifier.predict_one(
            ["bad", "awful"]
        )
        == "negative"
    )


def test_predict():
    classifier = MultinomialNaiveBayes()

    classifier.fit(
        [
            ["good"],
            ["great"],
            ["bad"],
            ["terrible"],
        ],
        [
            "positive",
            "positive",
            "negative",
            "negative",
        ],
    )

    predictions = classifier.predict(
        [
            ["good"],
            ["bad"],
        ]
    )

    assert predictions == [
        "positive",
        "negative",
    ]


def test_predict_unknown_words():
    classifier = MultinomialNaiveBayes()

    classifier.fit(
        [
            ["good"],
            ["bad"],
        ],
        [
            "positive",
            "negative",
        ],
    )

    prediction = classifier.predict_one(
        ["something", "unknown"]
    )

    assert prediction in [
        "positive",
        "negative",
    ]

def test_score():
    classifier = MultinomialNaiveBayes()

    documents = [
        ["good"],
        ["great"],
        ["bad"],
        ["terrible"],
    ]

    labels = [
        "positive",
        "positive",
        "negative",
        "negative",
    ]

    classifier.fit(
        documents,
        labels,
    )

    accuracy = classifier.score(
        [
            ["good"],
            ["bad"],
            ["great"],
            ["terrible"],
        ],
        [
            "positive",
            "negative",
            "positive",
            "negative",
        ],
    )

    assert accuracy == 1.0

def test_score_mismatched_labels():
    classifier = MultinomialNaiveBayes()

    classifier.fit(
        [["good"]],
        ["positive"],
    )

    with pytest.raises(ValueError):
        classifier.score(
            [["good"]],
            [],
        )


def test_score_empty_labels():
    classifier = MultinomialNaiveBayes()

    classifier.fit(
        [["good"]],
        ["positive"],
    )

    with pytest.raises(ValueError):
        classifier.score([], [])

def test_text_classifier_evaluate():
    classifier = TextClassifier()

    documents = [
        "good movie",
        "great film",
        "bad movie",
        "terrible film",
    ]

    labels = [
        "positive",
        "positive",
        "negative",
        "negative",
    ]

    classifier.fit(
        documents,
        labels,
    )

    report = classifier.evaluate(
        documents,
        labels,
    )

    assert "positive" in report
    assert "negative" in report
    assert "accuracy" in report

    assert report["accuracy"]["score"] == 1.0