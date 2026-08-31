from ..semantic_similarity.embeddings.base import (
    EmbeddingProvider,
)
from .models import SemanticSimilarityResult
from .similarity import cosine_similarity


class SemanticSimilarityEngine:

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
    ) -> None:

        if not isinstance(
            embedding_provider,
            EmbeddingProvider,
        ):
            raise TypeError(
                "embedding_provider must implement "
                "EmbeddingProvider"
            )

        self._embedding_provider = (
            embedding_provider
        )

    def compare(
        self,
        tokens_a: list[str],
        tokens_b: list[str],
    ) -> SemanticSimilarityResult:

        if not isinstance(tokens_a, list):
            raise TypeError(
                "tokens_a must be a list"
            )

        if not isinstance(tokens_b, list):
            raise TypeError(
                "tokens_b must be a list"
            )

        if not tokens_a:
            raise ValueError(
                "tokens_a cannot be empty"
            )

        if not tokens_b:
            raise ValueError(
                "tokens_b cannot be empty"
            )

        if self._embedding_provider.__class__.__name__ == "TransformerEmbeddingProvider":
            vector_a = self._embedding_provider.embed(" ".join(tokens_a))
            vector_b = self._embedding_provider.embed(" ".join(tokens_b))
        else:
            vector_a = self._embedding_provider.embed(
                tokens_a
            )
            vector_b = self._embedding_provider.embed(
                tokens_b
            )


        similarity = cosine_similarity(
            vector_a,
            vector_b,
        )

        return SemanticSimilarityResult(
            text_a=" ".join(tokens_a),
            text_b=" ".join(tokens_b),
            score=similarity,
        )