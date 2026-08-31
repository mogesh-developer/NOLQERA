import pytest

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)
from nolqera.intelligence.context_optimization.final_context_scoring import (
    FinalContextScorer,
    FinalContextScore,
)


def make_result(
    text: str,
    score: float,
    index: int = 0,
):
    return SemanticSearchResult(
        text=text,
        score=score,
        index=index,
    )


def test_scorer_accepts_result():

    scorer = FinalContextScorer()

    result = make_result(
        "python backend",
        0.9,
    )

    scored = scorer.score(result)

    assert isinstance(scored, FinalContextScore)


def test_final_score_is_between_zero_and_one():

    scorer = FinalContextScorer()

    result = make_result(
        "python backend",
        0.9,
    )

    scored = scorer.score(result)

    assert 0.0 <= scored.score <= 1.0


def test_relevance_contributes_to_final_score():

    scorer = FinalContextScorer(
        relevance_weight=1.0,
        diversity_weight=0.0,
        redundancy_weight=0.0,
    )

    result = make_result(
        "python backend",
        0.8,
    )

    scored = scorer.score(result)

    assert scored.score == pytest.approx(0.8)


def test_diversity_increases_final_score():

    scorer = FinalContextScorer(
        relevance_weight=0.5,
        diversity_weight=0.5,
        redundancy_weight=0.0,
    )

    result = make_result(
        "python backend",
        0.8,
    )

    low_diversity = scorer.score(
        result,
        diversity=0.0,
    )

    high_diversity = scorer.score(
        result,
        diversity=1.0,
    )

    assert high_diversity.score > low_diversity.score


def test_redundancy_decreases_final_score():

    scorer = FinalContextScorer(
        relevance_weight=0.7,
        diversity_weight=0.3,
        redundancy_weight=0.5,
    )

    result = make_result(
        "python backend",
        0.8,
    )

    low_redundancy = scorer.score(
        result,
        redundancy=0.0,
    )

    high_redundancy = scorer.score(
        result,
        redundancy=1.0,
    )

    assert low_redundancy.score > high_redundancy.score


def test_rank_orders_highest_final_score_first():

    scorer = FinalContextScorer()

    results = [
        make_result("low relevance", 0.2, 0),
        make_result("high relevance", 0.9, 1),
        make_result("medium relevance", 0.5, 2),
    ]

    ranked = scorer.rank(results)

    assert ranked[0].result.text == "high relevance"
    assert ranked[1].result.text == "medium relevance"
    assert ranked[2].result.text == "low relevance"


def test_rank_uses_diversity_scores():

    scorer = FinalContextScorer(
        relevance_weight=0.5,
        diversity_weight=0.5,
        redundancy_weight=0.0,
    )

    results = [
        make_result("result one", 0.7, 0),
        make_result("result two", 0.7, 1),
    ]

    ranked = scorer.rank(
        results,
        diversity_scores=[0.1, 0.9],
    )

    assert ranked[0].result.text == "result two"


def test_rank_uses_redundancy_scores():

    scorer = FinalContextScorer(
        relevance_weight=0.7,
        diversity_weight=0.0,
        redundancy_weight=0.3,
    )

    results = [
        make_result("result one", 0.8, 0),
        make_result("result two", 0.8, 1),
    ]

    ranked = scorer.rank(
        results,
        redundancy_scores=[0.9, 0.1],
    )

    assert ranked[0].result.text == "result two"


def test_rank_preserves_result_objects():

    scorer = FinalContextScorer()

    result = make_result(
        "python backend",
        0.9,
    )

    ranked = scorer.rank([result])

    assert ranked[0].result is result


def test_empty_results_return_empty_list():

    scorer = FinalContextScorer()

    assert scorer.rank([]) == []


def test_rejects_non_list():

    scorer = FinalContextScorer()

    with pytest.raises(TypeError):
        scorer.rank(None)


def test_rejects_invalid_result():

    scorer = FinalContextScorer()

    with pytest.raises(TypeError):
        scorer.score("invalid")


def test_rejects_invalid_signal():

    scorer = FinalContextScorer()

    result = make_result(
        "python backend",
        0.8,
    )

    with pytest.raises(ValueError):
        scorer.score(
            result,
            diversity=1.5,
        )


def test_rejects_negative_signal():

    scorer = FinalContextScorer()

    result = make_result(
        "python backend",
        0.8,
    )

    with pytest.raises(ValueError):
        scorer.score(
            result,
            redundancy=-0.1,
        )


def test_rejects_negative_weight():

    with pytest.raises(ValueError):
        FinalContextScorer(
            relevance_weight=-0.1,
        )


def test_rejects_zero_weights():

    with pytest.raises(ValueError):
        FinalContextScorer(
            relevance_weight=0.0,
            diversity_weight=0.0,
            redundancy_weight=0.0,
        )


def test_rank_rejects_mismatched_diversity_scores():

    scorer = FinalContextScorer()

    results = [
        make_result("one", 0.8),
        make_result("two", 0.7),
    ]

    with pytest.raises(ValueError):
        scorer.rank(
            results,
            diversity_scores=[1.0],
        )


def test_rank_rejects_mismatched_redundancy_scores():

    scorer = FinalContextScorer()

    results = [
        make_result("one", 0.8),
        make_result("two", 0.7),
    ]

    with pytest.raises(ValueError):
        scorer.rank(
            results,
            redundancy_scores=[0.0],
        )