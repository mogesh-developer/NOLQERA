import math
from collections import Counter


class TfidfVectorizer:
    """Convert tokenized documents into TF-IDF vectors."""

    def __init__(self):
        self.vocabulary: list[str] = []
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[list[str]]) -> None:
        """Build vocabulary and calculate IDF values."""

        if not documents:
            raise ValueError("Documents cannot be empty.")

        vocabulary = sorted(
            {
                token
                for document in documents
                for token in document
            }
        )

        self.vocabulary = vocabulary

        total_documents = len(documents)

        document_frequency = {
            token: sum(
                token in document
                for document in documents
            )
            for token in vocabulary
        }

        self.idf = {
            token: math.log(
                total_documents / frequency
            )
            for token, frequency in document_frequency.items()
        }

    def transform(self, documents: list[list[str]]) -> list[list[float]]:
        """Transform documents into TF-IDF vectors."""

        if not self.vocabulary:
            raise ValueError("The vectorizer has not been fitted.")

        vectors = []

        for document in documents:
            counts = Counter(document)
            total_terms = len(document)

            vector = []

            for token in self.vocabulary:
                if total_terms == 0:
                    tf = 0.0
                else:
                    tf = counts[token] / total_terms

                tfidf = tf * self.idf[token]

                vector.append(tfidf)

            vectors.append(vector)

        return vectors

    def fit_transform(
        self,
        documents: list[list[str]],
    ) -> list[list[float]]:
        """Fit the vectorizer and transform documents."""

        self.fit(documents)

        return self.transform(documents)