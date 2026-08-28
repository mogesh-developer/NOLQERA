from nolqera.intelligence.relevance.models import RelevanceResult


def test_relevance_result_stores_analysis():
    result = RelevanceResult(
        sentence="The application uses MongoDB.",
        score=0.85,
        label="relevant",
        rank=1,
    )

    assert result.sentence == (
        "The application uses MongoDB."
    )

    assert result.score == 0.85
    assert result.label == "relevant"
    assert result.rank == 1