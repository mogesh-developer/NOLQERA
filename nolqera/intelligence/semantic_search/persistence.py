import pickle
from pathlib import Path

from .models import SemanticSearchResult


class SemanticSearchPersistence:

    VERSION = 1

    @staticmethod
    def save(
        index,
        path: str | Path,
    ) -> None:

        if not isinstance(path, (str, Path)):
            raise TypeError(
                "path must be a string or Path"
            )

        path = Path(path)

        if not str(path).strip():
            raise ValueError(
                "path cannot be empty"
            )

        data = {
            "version": SemanticSearchPersistence.VERSION,
            "documents": index.documents,
            "document_vectors": index._document_vectors,
        }

        with path.open("wb") as file:
            pickle.dump(
                data,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def load(
        index,
        path: str | Path,
    ) -> None:

        if not isinstance(path, (str, Path)):
            raise TypeError(
                "path must be a string or Path"
            )

        path = Path(path)

        if not str(path).strip():
            raise ValueError(
                "path cannot be empty"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"index file not found: {path}"
            )

        with path.open("rb") as file:
            data = pickle.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "invalid semantic search index"
            )

        if data.get("version") != (
            SemanticSearchPersistence.VERSION
        ):
            raise ValueError(
                "unsupported semantic search index version"
            )

        documents = data.get("documents")
        document_vectors = data.get(
            "document_vectors"
        )

        if not isinstance(documents, list):
            raise ValueError(
                "invalid documents in index"
            )

        if not isinstance(
            document_vectors,
            list,
        ):
            raise ValueError(
                "invalid document vectors in index"
            )

        if len(documents) != len(
            document_vectors
        ):
            raise ValueError(
                "documents and vectors length mismatch"
            )

        index._documents = documents.copy()
        index._document_vectors = (
            document_vectors.copy()
        )
