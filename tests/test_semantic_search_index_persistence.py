import pytest

from nolqera.intelligence.semantic_search.index import (
    SemanticSearchIndex,
)
from nolqera.intelligence.semantic_similarity.embeddings.tfidf import (
    TFIDFEmbeddingProvider,
)


@pytest.fixture
def index():

    documents = [
        "fastapi python backend api",
        "python machine learning",
        "mongodb database storage",
    ]

    provider = TFIDFEmbeddingProvider()
    provider.fit(
        [
            document.split()
            for document in documents
        ]
    )

    return SemanticSearchIndex(
        embedding_provider=provider,
        documents=documents,
    )


def test_save_index(index, tmp_path):

    path = tmp_path / "semantic_index.pkl"

    index.save(path)

    assert path.exists()


def test_load_index(index, tmp_path):

    path = tmp_path / "semantic_index.pkl"

    original_documents = index.documents

    index.save(path)
    index.clear()

    assert index.document_count == 0

    index.load(path)

    assert index.documents == original_documents


def test_loaded_index_can_search(
    index,
    tmp_path,
):

    path = tmp_path / "semantic_index.pkl"

    index.save(path)

    index.clear()
    index.load(path)

    results = index.search(
        "python backend",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].score >= 0.0


def test_loaded_index_preserves_document_order(
    index,
    tmp_path,
):

    path = tmp_path / "semantic_index.pkl"

    original = index.documents

    index.save(path)
    index.clear()
    index.load(path)

    assert index.documents == original


def test_save_rejects_invalid_path(index):

    import pytest

    with pytest.raises(TypeError):
        index.save(123)


def test_load_rejects_invalid_path(index):

    import pytest

    with pytest.raises(TypeError):
        index.load(123)


def test_load_rejects_missing_file(
    index,
    tmp_path,
):

    import pytest

    path = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError):
        index.load(path)


def test_save_load_preserves_document_count(
    index,
    tmp_path,
):

    path = tmp_path / "semantic_index.pkl"

    count = index.document_count

    index.save(path)

    index.clear()
    index.load(path)

    assert index.document_count == count