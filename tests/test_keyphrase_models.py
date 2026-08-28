from nolqera.intelligence.keyphrase.models import (
    KeyphraseResult,
)


def test_keyphrase_result_stores_analysis():
    result = KeyphraseResult(
        phrase="FastAPI REST API",
        score=0.91,
        rank=1,
    )

    assert result.phrase == "FastAPI REST API"
    assert result.score == 0.91
    assert result.rank == 1