def test_semantic_similarity_public_api_exports():

    from nolqera.intelligence.semantic_similarity import (
        cosine_similarity,
        SemanticSimilarityResult,
        SemanticSimilarityScorer,
        SemanticScore,
        SemanticSimilarityRanker,
        SemanticSimilarityEngine,
    )

    assert callable(cosine_similarity)

    assert SemanticSimilarityResult is not None
    assert SemanticSimilarityScorer is not None
    assert SemanticScore is not None
    assert SemanticSimilarityRanker is not None
    assert SemanticSimilarityEngine is not None
