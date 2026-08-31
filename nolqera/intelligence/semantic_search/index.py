from ..semantic_similarity.embeddings.base import (
    EmbeddingProvider,
)
from ..semantic_similarity.similarity import (
    cosine_similarity,
)
from ...tokenization.word_tokenizer import tokenize_words
from .models import SemanticSearchResult


class SemanticSearchIndex:

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        documents: list[str],
    ) -> None:

        if not isinstance(
            embedding_provider,
            EmbeddingProvider,
        ):
            raise TypeError(
                "embedding_provider must implement "
                "EmbeddingProvider"
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

            if not isinstance(
                document,
                str,
            ):
                raise TypeError(
                    "all documents must be strings"
                )

            if not document.strip():
                raise ValueError(
                    "documents cannot contain "
                    "empty strings"
                )

        self._embedding_provider = (
            embedding_provider
        )

        self._documents = documents.copy()

        if (
            self._embedding_provider.__class__.__name__
            == "TransformerEmbeddingProvider"
        ):
            self._document_vectors = (
                self._embedding_provider.embed_many(
                    self._documents
                )
            )

        else:
            tokenized_documents = [
                tokenize_words(document)
                for document in self._documents
            ]

            self._document_vectors = (
                self._embedding_provider.embed_many(
                    tokenized_documents
                )
            )

    @property
    def documents(self) -> list[str]:
        return self._documents.copy()

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def search(
        self,
        query: str,
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

        else:
            query_vector = (
                self._embedding_provider.embed(
                    tokenize_words(query)
                )
            )

        results = []

        for index, (
            document,
            document_vector,
        ) in enumerate(
            zip(
                self._documents,
                self._document_vectors,
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

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        if top_k is not None:
            results = results[:top_k]

        return results

    def add_document(
        self,
        document: str,
    ) -> None:

        if not isinstance(document, str):
            raise TypeError(
                "document must be a string"
            )

        if not document.strip():
            raise ValueError(
                "document cannot be empty"
            )

        self._documents.append(document)

        if (
            self._embedding_provider.__class__.__name__
            == "TransformerEmbeddingProvider"
        ):
            vector = self._embedding_provider.embed(document)
        else:
            vector = self._embedding_provider.embed(tokenize_words(document))

        self._document_vectors.append(vector)

    def add_documents(
        self,
        documents: list[str],
    ) -> None:

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
                    "documents cannot contain empty strings"
                )

        self._documents.extend(documents)

        if (
            self._embedding_provider.__class__.__name__
            == "TransformerEmbeddingProvider"
        ):
            vectors = self._embedding_provider.embed_many(documents)
        else:
            vectors = self._embedding_provider.embed_many(
                [tokenize_words(doc) for doc in documents]
            )

        self._document_vectors.extend(vectors)

    def clear(self) -> None:
        self._documents.clear()
        self._document_vectors.clear()

    def remove_document(
        self,
        index: int,
    ) -> None:

        if not isinstance(index, int):
            raise TypeError(
                "index must be an integer"
            )

        if index < 0:
            raise ValueError(
                "index cannot be negative"
            )

        if index >= len(self._documents):
            raise IndexError(
                "index out of range"
            )

        self._documents.pop(index)
        self._document_vectors.pop(index)

    def update_document(
        self,
        index: int,
        document: str,
    ) -> None:

        if not isinstance(index, int):
            raise TypeError(
                "index must be an integer"
            )

        if index < 0:
            raise ValueError(
                "index cannot be negative"
            )

        if index >= len(self._documents):
            raise IndexError(
                "document index out of range"
            )

        if not isinstance(document, str):
            raise TypeError(
                "document must be a string"
            )

        if not document.strip():
            raise ValueError(
                "document cannot be empty"
            )

        if (
            self._embedding_provider.__class__.__name__
            == "TransformerEmbeddingProvider"
        ):
            vector = self._embedding_provider.embed(
                document
            )
        else:
            vector = self._embedding_provider.embed(
                tokenize_words(document)
            )

        self._documents[index] = document
        self._document_vectors[index] = vector

    def save(
        self,
        path: str,
    ) -> None:

        from .persistence import (
            SemanticSearchPersistence,
        )

        SemanticSearchPersistence.save(
            self,
            path,
        )

    def load(
        self,
        path: str,
    ) -> None:

        from .persistence import (
            SemanticSearchPersistence,
        )

        SemanticSearchPersistence.load(
            self,
            path,
        )