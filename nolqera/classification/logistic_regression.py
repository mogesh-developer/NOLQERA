import math
from typing import Union


class LogisticRegression:
    """Logistic Regression classifier using gradient descent."""

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
            raise TypeError("learning_rate must be a number")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")

        if not isinstance(epochs, int) or isinstance(epochs, bool):
            raise TypeError("epochs must be an integer")
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0")

        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def _sigmoid(self, z: float) -> float:
        """Compute the sigmoid of z."""
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    @staticmethod
    def _binary_cross_entropy(
        y_true: list[int],
        probabilities: list[float],
    ) -> float:
        """Calculate binary cross-entropy loss."""

        if len(y_true) != len(probabilities):
            raise ValueError(
                "y_true and probabilities must have "
                "the same length"
            )

        if not y_true:
            raise ValueError(
                "y_true cannot be empty"
            )

        epsilon = 1e-15
        total_loss = 0.0

        for actual, probability in zip(
            y_true,
            probabilities,
        ):
            probability = max(
                epsilon,
                min(1.0 - epsilon, probability),
            )

            loss = (
                actual * math.log(probability)
                + (1 - actual)
                * math.log(1 - probability)
            )

            total_loss -= loss

        return total_loss / len(y_true)

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        """Fit the model using gradient descent."""
        if not isinstance(X, list) or not isinstance(y, list):
            raise TypeError("X and y must be lists")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        if not X:
            raise ValueError("X and y cannot be empty")

        for label in y:
            if label not in (0, 1):
                raise ValueError("Labels must be binary (0 or 1)")

        n_samples = len(X)
        n_features = len(X[0])

        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                linear_pred = sum(self.weights[j] * X[i][j] for j in range(n_features)) + self.bias
                y_hat = self._sigmoid(linear_pred)
                error = y_hat - y[i]

                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error

            dw = [w_g / n_samples for w_g in dw]
            db /= n_samples

            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j]
            self.bias -= self.learning_rate * db

            probabilities = self.predict_probability(X)
            loss = self._binary_cross_entropy(y, probabilities)
            self.loss_history.append(loss)

    def predict_probability(self, X: Union[list[float], list[list[float]]]) -> Union[float, list[float]]:
        """Predict class probabilities."""
        if not X:
            return [] if isinstance(X, list) and len(X) == 0 else 0.5

        if isinstance(X[0], (int, float)):
            linear_pred = sum(w * x for w, x in zip(self.weights, X)) + self.bias
            return self._sigmoid(linear_pred)
        else:
            results = []
            for sample in X:
                linear_pred = sum(w * x for w, x in zip(self.weights, sample)) + self.bias
                results.append(self._sigmoid(linear_pred))
            return results

    def predict(self, X: Union[list[float], list[list[float]]], threshold: float = 0.5) -> Union[int, list[int]]:
        """Predict binary class labels."""
        if not X:
            return []

        if isinstance(X[0], (int, float)):
            prob = self.predict_probability(X)
            return 1 if prob >= threshold else 0
        else:
            probs = self.predict_probability(X)
            return [1 if p >= threshold else 0 for p in probs]
