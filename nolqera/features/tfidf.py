import math
from collections import Counter

from ..utils.text_utils import validate_tokens
from .vocabulary import Vocabulary


class TfidfVectorizer:
    """Convert tokenized documents into TF-IDF vectors."""

    def __init__(
        self,
        min_frequency: int = 1,
        add_unk: bool = False,
    ):
        self.vocabulary = Vocabulary(
            min_frequency=min_frequency,
            add_unk=add_unk,
        )

        self.idf: dict[str, float] = {}

    def fit(self, documents: list[list[str]]) -> None:
        """Build vocabulary and calculate IDF values."""

        if not isinstance(documents, list):
            raise TypeError("documents must be a list")

        for document in documents:
            validate_tokens(document)

        if not documents:
            raise ValueError("Documents cannot be empty.")

        self.vocabulary.fit(documents)

        total_documents = len(documents)

        document_frequency = {
            token: sum(
                token in document
                for document in documents
            )
            for token in self.vocabulary.token_to_index
            if token != self.vocabulary.UNK_TOKEN
        }

        self.idf = {}

        for token in self.vocabulary.token_to_index:
            if token == self.vocabulary.UNK_TOKEN:
                self.idf[token] = 0.0
            else:
                self.idf[token] = math.log(
                    total_documents /
                    document_frequency[token]
                )

    def transform(
        self,
        documents: list[list[str]],
    ) -> list[list[float]]:
        """Transform documents into TF-IDF vectors."""

        if self.vocabulary.size == 0:
            raise ValueError(
                "The vectorizer has not been fitted."
            )

        vectors = []

        for document in documents:
            validate_tokens(document)

            counts = Counter(document)
            total_terms = len(document)

            vector = []

            for index in range(self.vocabulary.size):
                token = self.vocabulary.get_token(index)

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

    @property
    def vocabulary_list(self) -> list[str]:
        """Return vocabulary tokens in index order."""

        return [
            self.vocabulary.get_token(index)
            for index in range(self.vocabulary.size)
        ]