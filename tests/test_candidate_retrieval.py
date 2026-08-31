import pytest

from nolqera.intelligence.retrieval_quality.candidate_retrieval import (
    CandidateRetriever,
    CandidateRetrievalResult,
)
from nolqera.intelligence.semantic_search.service import (
    SemanticSearchService,
)


def test_candidate_retriever_accepts_service(index):

    service = SemanticSearchService(index)

    retriever = CandidateRetriever(service)

    assert retriever.service is service


def test_candidate_retriever_rejects_invalid_service():

    with pytest.raises(TypeError):
        CandidateRetriever(None)


def test_candidate_retrieval_returns_result(index):

    service = SemanticSearchService(index)

    retriever = CandidateRetriever(service)

    result = retriever.retrieve(
        "python backend",
        candidate_k=2,
    )

    assert isinstance(
        result,
        CandidateRetrievalResult,
    )


def test_candidate_retrieval_returns_requested_candidate_count(
    index,
):

    service = SemanticSearchService(index)

    retriever = CandidateRetriever(service)

    result = retriever.retrieve(
        "python backend",
        candidate_k=2,
    )

    assert result.count == 2


def test_candidate_retrieval_preserves_search_order(index):

    service = SemanticSearchService(index)

    retriever = CandidateRetriever(service)

    result = retriever.retrieve(
        "python backend",
        candidate_k=3,
    )

    scores = [
        item.score
        for item in result.results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_candidate_retrieval_rejects_invalid_query(
    index,
):

    service = SemanticSearchService(index)

    retriever = CandidateRetriever(service)

    with pytest.raises(TypeError):
        retriever.retrieve(
            None,
            candidate_k=2,
        )


def test_candidate_retrieval_rejects_empty_query(
    index,
):

    service = SemanticSearchService(index)

    retriever = CandidateRetriever(service)

    with pytest.raises(ValueError):
        retriever.retrieve(
            "",
            candidate_k=2,
        )


def test_candidate_retrieval_rejects_invalid_candidate_k(
    index,
):

    service = SemanticSearchService(index)

    retriever = CandidateRetriever(service)

    with pytest.raises(TypeError):
        retriever.retrieve(
            "python",
            candidate_k="5",
        )


def test_candidate_retrieval_rejects_zero_candidate_k(
    index,
):

    service = SemanticSearchService(index)

    retriever = CandidateRetriever(service)

    with pytest.raises(ValueError):
        retriever.retrieve(
            "python",
            candidate_k=0,
        )


def test_candidate_retrieval_rejects_negative_candidate_k(
    index,
):

    service = SemanticSearchService(index)

    retriever = CandidateRetriever(service)

    with pytest.raises(ValueError):
        retriever.retrieve(
            "python",
            candidate_k=-1,
        )


def test_candidate_retrieval_count_property(
    index,
):

    service = SemanticSearchService(index)

    retriever = CandidateRetriever(service)

    result = retriever.retrieve(
        "python",
        candidate_k=2,
    )

    assert result.count == len(
        result.results
    )