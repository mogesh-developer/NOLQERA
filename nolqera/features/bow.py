from collections import Counter


class BagOfWords:
    """Convert text documents into Bag-of-Words vectors."""

    def __init__(self):
        self.vocabulary: list[str] = []
        self._vocabulary_index: dict[str, int] = {}

    def fit(self, documents: list[list[str]]) -> None:
        """Build vocabulary from tokenized documents."""

        vocabulary = set()

        for document in documents:
            vocabulary.update(document)

        self.vocabulary = sorted(vocabulary)

        self._vocabulary_index = {
            token: index
            for index, token in enumerate(self.vocabulary)
        }

    def transform(self, documents: list[list[str]]) -> list[list[int]]:
        """Convert documents into Bag-of-Words vectors."""

        if not self.vocabulary:
            raise ValueError("The vectorizer has not been fitted.")

        vectors = []

        for document in documents:
            counts = Counter(document)

            vector = [
                counts.get(token, 0)
                for token in self.vocabulary
            ]

            vectors.append(vector)

        return vectors

    def fit_transform(
        self,
        documents: list[list[str]],
    ) -> list[list[int]]:
        """Fit vocabulary and transform documents."""

        self.fit(documents)

        return self.transform(documents)