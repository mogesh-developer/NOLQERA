from nolqera.intelligence.importance.models import (
    ImportanceResult,
)


def test_importance_result_stores_analysis():
    result = ImportanceResult(
        sentence="The application uses MongoDB.",
        score=0.84,
        rank=1,
    )

    assert result.sentence == (
        "The application uses MongoDB."
    )

    assert result.score == 0.84
    assert result.rank == 1