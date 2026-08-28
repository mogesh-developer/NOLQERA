from collections import Counter, defaultdict
import math


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes classifier for text."""

    def __init__(self, alpha: float = 1.0):
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")

        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")

        self.alpha = float(alpha)

        self.classes_: list[str] = []
        self.class_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = defaultdict(
            Counter
        )

        self.class_token_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

        self.total_documents = 0

    def fit(
        self,
        documents: list[list[str]],
        labels: list[str],
    ) -> None:
        """Train the classifier."""

        if not isinstance(documents, list):
            raise TypeError("documents must be a list")

        if not isinstance(labels, list):
            raise TypeError("labels must be a list")

        if len(documents) != len(labels):
            raise ValueError(
                "documents and labels must have the same length"
            )

        if not documents:
            raise ValueError(
                "documents cannot be empty"
            )

        for document in documents:
            if not isinstance(document, list):
                raise TypeError(
                    "each document must be a list"
                )

            for token in document:
                if not isinstance(token, str):
                    raise TypeError(
                        "tokens must be strings"
                    )

        for label in labels:
            if not isinstance(label, str):
                raise TypeError(
                    "labels must be strings"
                )

        self.classes_ = sorted(set(labels))
        self.total_documents = len(documents)

        self.class_counts.clear()
        self.token_counts.clear()
        self.class_token_totals.clear()
        self.vocabulary.clear()

        for document, label in zip(documents, labels):
            self.class_counts[label] += 1

            for token in document:
                self.token_counts[label][token] += 1
                self.class_token_totals[label] += 1
                self.vocabulary.add(token)

    def class_prior(self, label: str) -> float:
        """Return P(class)."""

        if self.total_documents == 0:
            raise ValueError(
                "Classifier has not been fitted."
            )

        if label not in self.class_counts:
            raise KeyError(
                f"Unknown class: {label}"
            )

        return (
            self.class_counts[label]
            / self.total_documents
        )

    def token_probability(
        self,
        token: str,
        label: str,
    ) -> float:
        """Return smoothed P(token | class)."""

        if self.total_documents == 0:
            raise ValueError(
                "Classifier has not been fitted."
            )

        if label not in self.class_counts:
            raise KeyError(
                f"Unknown class: {label}"
            )

        token_count = self.token_counts[label][token]

        vocabulary_size = len(self.vocabulary)

        denominator = (
            self.class_token_totals[label]
            + self.alpha * vocabulary_size
        )

        numerator = (
            token_count + self.alpha
        )

        return numerator / denominator
    
    def _log_probability(
        self,
        document: list[str],
        label: str,
    ) -> float:
        """Calculate log P(class, document)."""

        if self.total_documents == 0:
            raise ValueError(
                "Classifier has not been fitted."
            )

        score = math.log(
            self.class_prior(label)
        )

        for token in document:
            probability = self.token_probability(
                token,
                label,
            )

            score += math.log(probability)

        return score

    def predict_one(
        self,
        document: list[str],
    ) -> str:
        """Predict the class for one document."""

        if self.total_documents == 0:
            raise ValueError(
                "Classifier has not been fitted."
            )

        if not isinstance(document, list):
            raise TypeError(
                "document must be a list"
            )

        for token in document:
            if not isinstance(token, str):
                raise TypeError(
                    "tokens must be strings"
                )

        scores = {
            label: self._log_probability(
                document,
                label,
            )
            for label in self.classes_
        }

        return max(
            scores,
            key=scores.get,
        )

    def predict(
        self,
        documents: list[list[str]],
    ) -> list[str]:
        """Predict classes for multiple documents."""

        if not isinstance(documents, list):
            raise TypeError(
                "documents must be a list"
            )

        return [
            self.predict_one(document)
            for document in documents
        ]

    def score(
        self,
        documents: list[list[str]],
        labels: list[str],
    ) -> float:
        """Return classification accuracy."""

        if not isinstance(labels, list):
            raise TypeError("labels must be a list")

        if len(documents) != len(labels):
            raise ValueError(
                "documents and labels must have the same length"
            )

        if not labels:
            raise ValueError(
                "labels cannot be empty"
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