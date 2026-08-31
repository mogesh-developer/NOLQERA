import pytest

from nolqera.intelligence.semantic_search.service import (
    SemanticSearchService,
)


@pytest.fixture
def service(index):

    return SemanticSearchService(index)


def test_service_save_index(service, tmp_path):

    path = tmp_path / "semantic_index.pkl"

    service.save(path)

    assert path.exists()


def test_service_load_index(service, tmp_path):

    path = tmp_path / "semantic_index.pkl"

    service.save(path)

    loaded_service = service.load(path)

    assert loaded_service is not None


def test_loaded_service_can_search(service, tmp_path):

    path = tmp_path / "semantic_index.pkl"

    service.save(path)

    loaded_service = service.load(path)

    results = loaded_service.search("python")

    assert results