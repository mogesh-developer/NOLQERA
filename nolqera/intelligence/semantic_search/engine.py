from ..semantic_similarity.embeddings.base import (
    EmbeddingProvider,
)
from ..semantic_similarity.similarity import (
    cosine_similarity,
)
from ...tokenization.word_tokenizer import tokenize_words
from .models import SemanticSearchResult


class SemanticSearchEngine:

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

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SemanticSearchResult]:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not isinstance(documents, list):
            raise TypeError(
                "documents must be a list"
            )

        if not documents:
            raise ValueError(
                "documents cannot be empty"
            )

        for document in documents:
            if not isinstance(document, str):
                raise TypeError(
                    "all documents must be strings"
                )

            if not document.strip():
                raise ValueError(
                    "documents cannot contain "
                    "empty strings"
                )

        if top_k is not None:

            if not isinstance(top_k, int):
                raise TypeError(
                    "top_k must be an integer"
                )

            if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than zero"
                )

        if min_score is not None:

            if not isinstance(
                min_score,
                (int, float),
            ):
                raise TypeError(
                    "min_score must be numeric"
                )

            if not 0.0 <= min_score <= 1.0:
                raise ValueError(
                    "min_score must be between 0 and 1"
                )

        if (
            self._embedding_provider.__class__.__name__
            == "TransformerEmbeddingProvider"
        ):
            query_vector = (
                self._embedding_provider.embed(query)
            )

            document_vectors = (
                self._embedding_provider.embed_many(
                    documents
                )
            )

        else:
            query_vector = (
                self._embedding_provider.embed(
                    tokenize_words(query)
                )
            )

            document_vectors = (
                self._embedding_provider.embed_many(
                    [
                        tokenize_words(document)
                        for document in documents
                    ]
                )
            )

        results = []

        for index, (
            document,
            document_vector,
        ) in enumerate(
            zip(
                documents,
                document_vectors,
            )
        ):

            similarity = cosine_similarity(
                query_vector,
                document_vector,
            )

            if (
                min_score is not None
                and similarity < min_score
            ):
                continue

            results.append(
                SemanticSearchResult(
                    text=document,
                    score=similarity,
                    index=index,
                )
            )

        results = sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )

        if top_k is not None:
            results = results[:top_k]

        return results