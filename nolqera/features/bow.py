from ..utils.text_utils import validate_tokens
from .vocabulary import Vocabulary


class BagOfWords:
    """Convert tokenized documents into Bag-of-Words vectors."""

    def __init__(
        self,
        min_frequency: int = 1,
        add_unk: bool = False,
    ):
        self.vocabulary = Vocabulary(
            min_frequency=min_frequency,
            add_unk=add_unk,
        )

    def fit(self, documents: list[list[str]]) -> None:
        """Build the vocabulary from documents."""

        if not isinstance(documents, list):
            raise TypeError("documents must be a list")

        for document in documents:
            validate_tokens(document)

        self.vocabulary.fit(documents)

    def transform(self, documents: list[list[str]]) -> list[list[int]]:
        """Convert documents into Bag-of-Words vectors."""

        if self.vocabulary.size == 0:
            raise ValueError(
                "The vectorizer has not been fitted."
            )

        vectors = []

        for document in documents:
            validate_tokens(document)

            vector = [0] * self.vocabulary.size

            for token in document:
                try:
                    index = self.vocabulary.get_index(token)
                    vector[index] += 1
                except KeyError:
                    pass

            vectors.append(vector)

        return vectors

    def fit_transform(
        self,
        documents: list[list[str]],
    ) -> list[list[int]]:
        """Fit the vocabulary and transform documents."""

        self.fit(documents)

        return self.transform(documents)

    @property
    def vocabulary_list(self) -> list[str]:
        """Return vocabulary tokens in index order."""

        return [
            self.vocabulary.get_token(index)
            for index in range(self.vocabulary.size)
        ]