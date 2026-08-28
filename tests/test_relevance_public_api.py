from nolqera.intelligence.relevance import (
    RelevanceEngine,
    RelevanceResult,
    RelevanceRanker,
    RelevanceScorer,
    cosine_similarity,
)


def test_relevance_public_api_exports():
    assert callable(cosine_similarity)
    assert RelevanceEngine is not None
    assert RelevanceResult is not None
    assert RelevanceRanker is not None
    assert RelevanceScorer is not None