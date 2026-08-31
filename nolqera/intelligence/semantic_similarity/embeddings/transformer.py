from sentence_transformers import SentenceTransformer

from .base import EmbeddingProvider


class TransformerEmbeddingProvider(EmbeddingProvider):
    """Generate dense semantic embeddings using a Transformer model."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:

        if not isinstance(model_name, str):
            raise TypeError(
                "model_name must be a string"
            )

        if not model_name.strip():
            raise ValueError(
                "model_name cannot be empty"
            )

        self.model_name = model_name
        self._model = SentenceTransformer(
            model_name
        )

    def embed(
        self,
        text: str,
    ) -> list[float]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "text cannot be empty"
            )

        vector = self._model.encode(
            text,
            convert_to_numpy=True,
        )

        return vector.tolist()

    def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not isinstance(texts, list):
            raise TypeError(
                "texts must be a list"
            )

        if not texts:
            raise ValueError(
                "texts cannot be empty"
            )

        for text in texts:
            if not isinstance(text, str):
                raise TypeError(
                    "all texts must be strings"
                )

            if not text.strip():
                raise ValueError(
                    "texts cannot contain empty strings"
                )

        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
        )

        return vectors.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""

        return self._model.get_sentence_embedding_dimension()