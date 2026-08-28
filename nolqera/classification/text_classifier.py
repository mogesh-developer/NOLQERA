from ..tokenization.word_tokenizer import tokenize_words
from .naive_bayes import MultinomialNaiveBayes
from .report import classification_report

class TextClassifier:
    """High-level text classification pipeline."""

    def __init__(
        self,
        classifier: MultinomialNaiveBayes | None = None,
    ):
        self.classifier = (
            classifier
            if classifier is not None
            else MultinomialNaiveBayes()
        )

    @staticmethod
    def _tokenize_documents(
        documents: list[str],
    ) -> list[list[str]]:
        if not isinstance(documents, list):
            raise TypeError(
                "documents must be a list"
            )

        for document in documents:
            if not isinstance(document, str):
                raise TypeError(
                    "documents must contain strings"
                )

        return [
            tokenize_words(document)
            for document in documents
        ]

    def fit(
        self,
        documents: list[str],
        labels: list[str],
    ) -> None:
        """Train the classifier on raw text."""

        tokenized_documents = (
            self._tokenize_documents(documents)
        )

        self.classifier.fit(
            tokenized_documents,
            labels,
        )

    def predict(
        self,
        documents: list[str],
    ) -> list[str]:
        """Predict classes for raw text."""

        tokenized_documents = (
            self._tokenize_documents(documents)
        )

        return self.classifier.predict(
            tokenized_documents
        )

    def predict_one(
        self,
        document: str,
    ) -> str:
        """Predict the class for one text."""

        if not isinstance(document, str):
            raise TypeError(
                "document must be a string"
            )

        tokens = tokenize_words(document)

        return self.classifier.predict_one(
            tokens
        )

    def score(
        self,
        documents: list[str],
        labels: list[str],
    ) -> float:
        """Calculate accuracy on raw text."""

        if not labels:
            raise ValueError(
                "labels cannot be empty"
            )

        if len(documents) != len(labels):
            raise ValueError(
                "documents and labels must have "
                "the same length"
            )

        predictions = self.predict(documents)

        correct = sum(
            prediction == label
            for prediction, label in zip(
                predictions,
                labels,
            )
        )

        return correct / len(labels)
    
    def evaluate(
        self,
        documents: list[str],
        labels: list[str],
    ) -> dict:
        """Evaluate the classifier."""

        if not labels:
            raise ValueError(
                "labels cannot be empty"
            )

        if len(documents) != len(labels):
            raise ValueError(
                "documents and labels must have "
                "the same length"
            )

        predictions = self.predict(documents)

        return classification_report(
            labels,
            predictions,
        )
        