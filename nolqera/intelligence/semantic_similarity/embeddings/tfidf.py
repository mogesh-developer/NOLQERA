from ....features.tfidf import TfidfVectorizer
from .base import EmbeddingProvider


class TFIDFEmbeddingProvider(EmbeddingProvider):
    """
    EmbeddingProvider adapter for NOLQERA's TF-IDF vectorizer.

    This provider works with already-tokenized documents.
    """

    def __init__(
        self,
        min_frequency: int = 1,
        add_unk: bool = False,
    ) -> None:

        self._vectorizer = TfidfVectorizer(
            min_frequency=min_frequency,
            add_unk=add_unk,
        )

        self._fitted = False

    def fit(
        self,
        documents: list[list[str]],
    ) -> None:

        if not isinstance(documents, list):
            raise TypeError(
                "documents must be a list"
            )

        if not documents:
            raise ValueError(
                "documents cannot be empty"
            )

        self._vectorizer.fit(documents)
        self._fitted = True

    def embed(
        self,
        tokens: list[str],
    ) -> list[float]:

        if not isinstance(tokens, list):
            raise TypeError(
                "tokens must be a list"
            )

        if not self._fitted:
            raise RuntimeError(
                "embedding provider must be fitted "
                "before embedding"
            )

        return self._vectorizer.transform(
            [tokens]
        )[0]

    def embed_many(
        self,
        documents: list[list[str]],
    ) -> list[list[float]]:

        if not isinstance(documents, list):
            raise TypeError(
                "documents must be a list"
            )

        if not documents:
            raise ValueError(
                "documents cannot be empty"
            )

        if not self._fitted:
            raise RuntimeError(
                "embedding provider must be fitted "
                "before embedding"
            )

        return self._vectorizer.transform(
            documents
        )

    @property
    def vocabulary(self) -> list[str]:
        """Return vocabulary in vector index order."""

        if not self._fitted:
            raise RuntimeError(
                "embedding provider must be fitted "
                "before accessing vocabulary"
            )

        return self._vectorizer.vocabulary_list