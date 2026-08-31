from nolqera.intelligence.semantic_search import (
    SemanticSearchResult,
    SemanticSearchEngine,
)


def test_semantic_search_public_api_exports():

    assert SemanticSearchResult is not None
    assert SemanticSearchEngine is not None