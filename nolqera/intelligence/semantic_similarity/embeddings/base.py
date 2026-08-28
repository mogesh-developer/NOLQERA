from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Base interface for converting tokenized text
    into embedding vectors.
    """

    @abstractmethod
    def embed(
        self,
        tokens: list[str],
    ) -> list[float]:
        raise NotImplementedError

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

        return [
            self.embed(tokens)
            for tokens in documents
        ]