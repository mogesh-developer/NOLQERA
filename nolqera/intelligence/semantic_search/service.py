from .index import SemanticSearchIndex
from .models import SemanticSearchResult


class SemanticSearchService:

    def __init__(
        self,
        index: SemanticSearchIndex,
    ) -> None:

        if not isinstance(
            index,
            SemanticSearchIndex,
        ):
            raise TypeError(
                "index must be a SemanticSearchIndex"
            )

        self._index = index

    @property
    def index(self) -> SemanticSearchIndex:
        return self._index

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SemanticSearchResult]:

        return self._index.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

    def add_document(
        self,
        document: str,
    ) -> None:

        self._index.add_document(document)

    def add_documents(
        self,
        documents: list[str],
    ) -> None:

        self._index.add_documents(documents)

    def update_document(
        self,
        index: int,
        document: str,
    ) -> None:

        self._index.update_document(
            index,
            document,
        )

    def remove_document(
        self,
        index: int,
    ) -> None:

        self._index.remove_document(index)

    def clear(self) -> None:

        self._index.clear()

    @property
    def document_count(self) -> int:
        return self._index.document_count

    @property
    def documents(self) -> list[str]:
        return self._index.documents

    def save(
        self,
        path: str,
    ) -> None:
        self._index.save(path)

    def load(
        self,
        path: str,
    ) -> "SemanticSearchService":
        self._index.load(path)
        return self